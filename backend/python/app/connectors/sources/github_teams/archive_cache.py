"""Repository archives for indexing code without one API call per file.

A full sync lists a repository's files from one Git Tree call, but indexing
then fetched each file with its own Git Blob call. A personal access token has
5,000 ``core`` calls an hour, so a 20,000-file sync took four hours of rate
limit waits. GitHub serves the whole tree at a commit as one tarball for one
call; this module downloads it once per repository and commit, unpacks it to
local disk and answers file reads from there.

A file is served from an archive only when its git blob sha matches the sha
the record carries, so the bytes indexed are exactly the blob the record
describes. Anything an archive cannot vouch for (``export-ignore`` or
``export-subst`` attributes, a symlink, an oversized file, a commit that is not
unpacked) returns ``None`` and the caller falls back to the per-file API.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

_ROOT_ENV = "GITHUB_ARCHIVE_CACHE_DIR"
_DISABLED_ENV = "GITHUB_ARCHIVE_CACHE_DISABLED"
_MAX_ARCHIVE_BYTES_ENV = "GITHUB_ARCHIVE_MAX_BYTES"

DEFAULT_IDLE_TTL_SECONDS = 3600.0
DEFAULT_FAILURE_BACKOFF_SECONDS = 900.0
DEFAULT_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_PER_REPO = 2

# Downloads the tarball for (repo, commit) into the given path; returns bytes
# written, or raises. The caller logs and backs off on any failure.
Downloader = Callable[[str], Awaitable[int]]


def git_blob_sha(content: bytes) -> str:
    """The sha1 git names a blob by: ``sha1("blob <len>\\0" + content)``."""
    return hashlib.sha1(b"blob %d\x00" % len(content) + content, usedforsecurity=False).hexdigest()


def _safe_relative(name: str) -> str | None:
    """``name`` without the tarball's ``owner-repo-sha/`` top directory, or
    None when it is the top directory itself or would escape the tree."""
    parts = PurePosixPath(name).parts[1:]
    if not parts or any(p in ("", ".", "..") for p in parts) or name.startswith("/"):
        return None
    return "/".join(parts)


def extract_tarball(tar_path: str, dest: str, *, max_file_bytes: int) -> int:
    """Unpack regular files of at most ``max_file_bytes`` into ``dest``.

    Links, devices and oversized files are skipped: none can be served (a
    symlink's blob is its target path, which the per-file API returns), and
    skipping them keeps disk use to what indexing reads. Returns the number of
    files written.
    """
    count = 0
    with tarfile.open(tar_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isreg() or member.size > max_file_bytes:
                continue
            rel = _safe_relative(member.name)
            if rel is None:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


@dataclass
class _Entry:
    path: str
    last_used: float = field(default_factory=time.monotonic)


class RepoArchiveCache:
    """Unpacked repository trees keyed by ``(repo_id, commit_sha)``.

    One process-wide instance (``shared_archive_cache``) serves every GitHub
    connector in the process. At most ``max_per_repo`` commits per repository
    stay on disk; an archive unused for ``idle_ttl`` seconds is deleted, so a
    finished sync leaves nothing behind. A failed download is not retried for
    ``failure_backoff`` seconds; reads fall back to the per-file API meanwhile.
    """

    def __init__(
        self,
        root: str,
        *,
        max_file_bytes: int,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        idle_ttl: float = DEFAULT_IDLE_TTL_SECONDS,
        failure_backoff: float = DEFAULT_FAILURE_BACKOFF_SECONDS,
        max_per_repo: int = DEFAULT_MAX_PER_REPO,
        logger: logging.Logger | None = None,
    ) -> None:
        self.root = root
        self.max_file_bytes = max_file_bytes
        self.max_archive_bytes = max_archive_bytes
        self.idle_ttl = idle_ttl
        self.failure_backoff = failure_backoff
        self.max_per_repo = max_per_repo
        self.logger = logger or logging.getLogger(__name__)
        self._entries: dict[tuple[int, str], _Entry] = {}
        self._failed: dict[tuple[int, str], float] = {}
        self._locks: dict[tuple[int, str], asyncio.Lock] = {}
        # Archives left by an earlier process are not trusted: the process may
        # have died mid-extract. Start empty.
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------------

    async def read(
        self,
        repo_id: int,
        commit_sha: str | None,
        path: str,
        blob_sha: str,
        *,
        download: Downloader | None,
    ) -> bytes | None:
        """The file's bytes when an archive holds this exact blob, else None.

        Archives already unpacked for the repository are tried first, newest
        use first: a file unchanged since another commit has the same blob
        sha. When none holds it and ``download`` is given, the tarball at
        ``commit_sha`` is fetched once (concurrent readers wait for it).
        """
        await self._evict_idle()
        content = await self._read_any(repo_id, path, blob_sha, prefer=commit_sha)
        if content is not None or download is None or not commit_sha:
            return content
        key = (repo_id, commit_sha)
        if key in self._entries or self._backing_off(key):
            return None
        if not await self._ensure(key, download):
            return None
        return await self._read_entry(key, path, blob_sha)

    def stats(self) -> dict[str, int]:
        return {"archives": len(self._entries), "failed": len(self._failed)}

    # ------------------------------------------------------------------

    async def _read_any(
        self, repo_id: int, path: str, blob_sha: str, *, prefer: str | None,
    ) -> bytes | None:
        keys = [k for k in self._entries if k[0] == repo_id]
        keys.sort(key=lambda k: (k[1] != prefer, -self._entries[k].last_used))
        for key in keys:
            content = await self._read_entry(key, path, blob_sha)
            if content is not None:
                return content
        return None

    async def _read_entry(self, key: tuple[int, str], path: str, blob_sha: str) -> bytes | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        rel = _safe_relative(f"top/{path}")
        if rel is None:
            return None
        target = os.path.join(entry.path, rel)
        try:
            content = await asyncio.to_thread(_read_file, target)
        except OSError:
            return None
        entry.last_used = time.monotonic()
        if git_blob_sha(content) != blob_sha:
            return None
        return content

    def _backing_off(self, key: tuple[int, str]) -> bool:
        failed_at = self._failed.get(key)
        if failed_at is None:
            return False
        if time.monotonic() - failed_at < self.failure_backoff:
            return True
        del self._failed[key]
        return False

    async def _ensure(self, key: tuple[int, str], download: Downloader) -> bool:
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._entries:
                return True
            if self._backing_off(key):
                return False
            work = os.path.join(self.root, f".work-{uuid.uuid4().hex}")
            final = os.path.join(self.root, f"{key[0]}-{key[1]}")
            os.makedirs(work)
            tar_path = os.path.join(work, "archive.tar.gz")
            tree = os.path.join(work, "tree")
            started = time.monotonic()
            try:
                size = await download(tar_path)
                files = await asyncio.to_thread(
                    extract_tarball, tar_path, tree, max_file_bytes=self.max_file_bytes,
                )
                shutil.rmtree(final, ignore_errors=True)
                os.replace(tree, final)
            except Exception as e:
                self._failed[key] = time.monotonic()
                self.logger.warning(
                    "GitHub archive for repo %s at %s unavailable (%s); "
                    "files fall back to per-file API calls for %ss",
                    key[0], key[1][:12], e, int(self.failure_backoff),
                )
                return False
            finally:
                await asyncio.to_thread(shutil.rmtree, work, True)
            self._entries[key] = _Entry(final)
            self.logger.info(
                "GitHub archive for repo %s at %s: %s bytes, %s files unpacked in %.1fs",
                key[0], key[1][:12], size, files, time.monotonic() - started,
            )
            await self._trim_repo(key[0], keep=key)
            return True

    async def _trim_repo(self, repo_id: int, *, keep: tuple[int, str]) -> None:
        keys = sorted(
            (k for k in self._entries if k[0] == repo_id and k != keep),
            key=lambda k: self._entries[k].last_used,
        )
        while len(keys) + 1 > self.max_per_repo:
            await self._drop(keys.pop(0))

    async def _evict_idle(self) -> None:
        now = time.monotonic()
        for key in [k for k, e in self._entries.items() if now - e.last_used > self.idle_ttl]:
            lock = self._locks.get(key)
            if lock is not None and lock.locked():
                continue
            await self._drop(key)

    async def _drop(self, key: tuple[int, str]) -> None:
        entry = self._entries.pop(key, None)
        self._locks.pop(key, None)
        if entry is not None:
            await asyncio.to_thread(shutil.rmtree, entry.path, True)
            self.logger.info("GitHub archive for repo %s at %s removed", key[0], key[1][:12])


def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


_shared: RepoArchiveCache | None = None


def shared_archive_cache(*, max_file_bytes: int, logger: logging.Logger | None = None) -> RepoArchiveCache | None:
    """The process-wide cache, or None when ``GITHUB_ARCHIVE_CACHE_DISABLED`` is set."""
    global _shared  # noqa: PLW0603 - one cache per process, built on first use
    if os.environ.get(_DISABLED_ENV, "").strip().lower() in ("1", "true", "yes"):
        return None
    if _shared is None:
        root = os.environ.get(_ROOT_ENV) or os.path.join(tempfile.gettempdir(), "pipeshub-github-archives")
        max_archive = int(os.environ.get(_MAX_ARCHIVE_BYTES_ENV) or DEFAULT_MAX_ARCHIVE_BYTES)
        _shared = RepoArchiveCache(
            root, max_file_bytes=max_file_bytes, max_archive_bytes=max_archive, logger=logger,
        )
    return _shared
