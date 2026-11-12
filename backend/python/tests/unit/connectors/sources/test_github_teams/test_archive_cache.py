"""Repository tarballs serve full-sync file content in one API call per repo.

Covers the archive cache itself (blob-sha verification, safe extraction,
single download under concurrency, failure backoff, cleanup) and how
``ReposSync.fetch_code_file_content`` uses it: archive first, the per-file
Git Blob call only for files the archive cannot vouch for.
"""
from __future__ import annotations

import asyncio
import io
import os
import tarfile
import time
from types import SimpleNamespace

import pytest

from app.connectors.sources.github_teams.archive_cache import (
    RepoArchiveCache,
    extract_tarball,
    git_blob_sha,
)
from app.connectors.sources.github_teams.models import blob_external_id
from app.connectors.sources.github_teams.repos import ReposSync
from app.models.entities import CodeFileRecord

from tests.unit.connectors.sources.test_github_teams.conftest import (
    failed_response,
    make_mock_connector,
    make_repo,
    ok_response,
)

pytestmark = pytest.mark.anyio

MAX_FILE = 1024


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def write_tarball(dest: str, files: dict[str, bytes], *, top: str = "acme-widgets-abc123",
                  extra: list[tarfile.TarInfo] | None = None) -> int:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        root = tarfile.TarInfo(top)
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        for path, body in files.items():
            info = tarfile.TarInfo(f"{top}/{path}")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        for info in extra or []:
            tar.addfile(info)
    data = buf.getvalue()
    with open(dest, "wb") as fh:
        fh.write(data)
    return len(data)


def make_cache(tmp_path, **kwargs) -> RepoArchiveCache:
    return RepoArchiveCache(str(tmp_path / "archives"), max_file_bytes=MAX_FILE, **kwargs)


class TestGitBlobSha:
    def test_matches_git(self) -> None:
        # `git hash-object` of an empty file and of "hello\n".
        assert git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


class TestExtractTarball:
    def test_only_safe_regular_files_within_the_size_cap(self, tmp_path) -> None:
        link = tarfile.TarInfo("acme-widgets-abc123/link.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        escape = tarfile.TarInfo("acme-widgets-abc123/../../evil.py")
        escape.size = 0
        tar_path = str(tmp_path / "a.tar.gz")
        write_tarball(
            tar_path,
            {"src/app.py": b"print(1)\n", "big.bin": b"x" * (MAX_FILE + 1)},
            extra=[link, escape],
        )
        dest = tmp_path / "tree"

        count = extract_tarball(tar_path, str(dest), max_file_bytes=MAX_FILE)

        assert count == 1
        assert (dest / "src/app.py").read_bytes() == b"print(1)\n"
        assert not (dest / "big.bin").exists()
        assert not (dest / "link.py").exists()
        assert not (tmp_path / "evil.py").exists()


class TestRepoArchiveCache:
    async def test_concurrent_reads_download_once_and_verify_each_blob(self, tmp_path) -> None:
        cache = make_cache(tmp_path)
        files = {f"pkg/m{i}.py": f"x = {i}\n".encode() for i in range(10)}
        downloads: list[str] = []

        async def download(dest: str) -> int:
            downloads.append(dest)
            await asyncio.sleep(0.01)
            return write_tarball(dest, files)

        results = await asyncio.gather(*(
            cache.read(7, "abc123", path, git_blob_sha(body), download=download)
            for path, body in files.items()
        ))

        assert results == list(files.values())
        assert len(downloads) == 1
        # The tarball itself is gone once unpacked.
        assert not os.path.exists(downloads[0])

    async def test_a_blob_the_archive_does_not_hold_is_not_served(self, tmp_path) -> None:
        cache = make_cache(tmp_path)

        async def download(dest: str) -> int:
            return write_tarball(dest, {"a.py": b"new body\n"})

        assert await cache.read(7, "abc123", "a.py", git_blob_sha(b"old body\n"), download=download) is None
        assert await cache.read(7, "abc123", "missing.py", git_blob_sha(b""), download=download) is None

    async def test_an_unchanged_file_is_served_from_another_commit_without_a_download(self, tmp_path) -> None:
        cache = make_cache(tmp_path)
        calls: list[str] = []

        async def download(dest: str) -> int:
            calls.append(dest)
            return write_tarball(dest, {"a.py": b"same\n"})

        await cache.read(7, "c1", "a.py", git_blob_sha(b"same\n"), download=download)
        assert await cache.read(7, "c2", "a.py", git_blob_sha(b"same\n"), download=download) == b"same\n"
        assert len(calls) == 1

    async def test_no_download_without_a_downloader(self, tmp_path) -> None:
        cache = make_cache(tmp_path)
        assert await cache.read(7, "abc123", "a.py", git_blob_sha(b"x"), download=None) is None
        assert cache.stats() == {"archives": 0, "failed": 0}

    async def test_a_failed_download_backs_off(self, tmp_path) -> None:
        cache = make_cache(tmp_path, failure_backoff=60)
        calls: list[str] = []

        async def download(dest: str) -> int:
            calls.append(dest)
            raise RuntimeError("HTTP 429: rate limited")

        for _ in range(3):
            assert await cache.read(7, "abc123", "a.py", git_blob_sha(b"x"), download=download) is None
        assert len(calls) == 1
        assert cache.stats()["failed"] == 1
        # Nothing half-written is left behind.
        assert [p for p in os.listdir(cache.root)] == []

    async def test_idle_archives_are_removed(self, tmp_path) -> None:
        cache = make_cache(tmp_path, idle_ttl=0.05)

        async def download(dest: str) -> int:
            return write_tarball(dest, {"a.py": b"x"})

        await cache.read(7, "abc123", "a.py", git_blob_sha(b"x"), download=download)
        assert len(os.listdir(cache.root)) == 1
        time.sleep(0.1)
        await cache.read(8, "zzz", "b.py", git_blob_sha(b"y"), download=None)

        assert os.listdir(cache.root) == []
        assert cache.stats()["archives"] == 0

    async def test_only_the_newest_commits_of_a_repo_are_kept(self, tmp_path) -> None:
        cache = make_cache(tmp_path, max_per_repo=1)

        def downloader(body: bytes):
            async def download(dest: str) -> int:
                return write_tarball(dest, {"a.py": body})
            return download

        await cache.read(7, "c1", "a.py", git_blob_sha(b"one"), download=downloader(b"one"))
        await cache.read(7, "c2", "a.py", git_blob_sha(b"two"), download=downloader(b"two"))

        assert sorted(os.listdir(cache.root)) == ["7-c2"]

    def test_leftovers_from_an_earlier_process_are_cleared(self, tmp_path) -> None:
        stale = tmp_path / "archives" / "7-old"
        stale.mkdir(parents=True)
        (stale / "a.py").write_bytes(b"x")

        make_cache(tmp_path)

        assert not stale.exists()


class TestFetchThroughArchive:
    """A full-sync record reads the tarball once per repo and commit; files
    the archive cannot vouch for still get their own Git Blob call."""

    FILES = {"a.py": b"a = 1\n", "pkg/b.py": b"b = 2\n", "pkg/deep/c.py": b"c = 3\n"}

    def _record(self, path: str, body: bytes, *, commit: str | None = "abc123",
                size: int | None = -1, sha: str | None = None) -> CodeFileRecord:
        return CodeFileRecord(
            id=f"rec-{path}", org_id="org-1", record_name=path.rsplit("/", 1)[-1],
            record_type="CODE_FILE", version=0, origin="CONNECTOR",
            connector_name="GITHUB TEAMS", connector_id="github-conn-1",
            external_record_id=blob_external_id(1, path), external_record_group_id="1-code-repository",
            file_path=path, file_hash=sha or git_blob_sha(body),
            size_in_bytes=len(body) if size == -1 else size, commit_sha=commit,
        )

    def _wire(self, c, *, tarball=None, files: dict[str, bytes] | None = None) -> list[str]:
        import base64

        repo = make_repo(repo_id=1)
        calls: list[str] = []
        files = self.FILES if files is None else files

        def dispatch(method, *args, **kwargs):
            calls.append(method._mock_name)
            if method is c.data_source.download_repo_tarball:
                owner, name, ref, dest = args
                assert (owner, name, ref) == ("acme", "widgets", "abc123")
                if tarball is not None:
                    return tarball
                return ok_response(write_tarball(dest, files))
            if method is c.data_source.get_git_blob:
                return ok_response(SimpleNamespace(size=3, content=base64.b64encode(b"api").decode(), encoding="base64"))
            if method is c.data_source.get_file_contents:
                return ok_response(SimpleNamespace(size=3, decoded_content=b"contents-api", content=None))
            raise AssertionError(f"unexpected ds_call for {method!r}")

        async def ds_call(method, *args, _github_timeout=None, **kwargs):
            return dispatch(method, *args, **kwargs)

        async def get_repo_by_id(repo_id: int):
            return ok_response(repo)

        c.runtime.ds_call = ds_call
        c.runtime.get_repo_by_id = get_repo_by_id
        c.data_source.rate_limit_budget.return_value = (5000, 4321, int(time.time()) + 600)
        return calls

    async def test_every_file_of_a_commit_costs_one_tarball_call(self, tmp_path) -> None:
        c = make_mock_connector()
        calls = self._wire(c)
        sync = ReposSync(c, archive_cache=make_cache(tmp_path))

        for path, body in self.FILES.items():
            assert await sync.fetch_code_file_content(self._record(path, body)) == body

        assert calls == ["download_repo_tarball"]

    async def test_path_and_ids_are_those_of_the_api_path(self, tmp_path) -> None:
        # Records are built by the tree listing, untouched by the archive: the
        # same external ids and paths whichever way content is fetched.
        c = make_mock_connector()
        self._wire(c)
        sync = ReposSync(c, archive_cache=make_cache(tmp_path))
        record = self._record("pkg/deep/c.py", self.FILES["pkg/deep/c.py"])
        before = (record.external_record_id, record.file_path, record.external_record_group_id)

        await sync.fetch_code_file_content(record)

        assert (record.external_record_id, record.file_path, record.external_record_group_id) == before
        assert record.external_record_id == blob_external_id(1, "pkg/deep/c.py")

    async def test_a_blob_the_archive_does_not_match_uses_the_git_blob_api(self, tmp_path) -> None:
        c = make_mock_connector()
        calls = self._wire(c)
        sync = ReposSync(c, archive_cache=make_cache(tmp_path))
        # export-subst / eol attributes change the archived bytes.
        record = self._record("a.py", b"a = 1\n", sha=git_blob_sha(b"api"), size=3)

        assert await sync.fetch_code_file_content(record) == b"api"
        assert calls == ["download_repo_tarball", "get_git_blob"]

    async def test_a_failed_tarball_falls_back_to_per_file_calls_once(self, tmp_path) -> None:
        c = make_mock_connector()
        calls = self._wire(c, tarball=failed_response("rate limited", status_code=429))
        sync = ReposSync(c, archive_cache=make_cache(tmp_path))

        for path, body in self.FILES.items():
            assert await sync.fetch_code_file_content(self._record(path, body)) == b"api"

        assert calls == ["download_repo_tarball", "get_git_blob", "get_git_blob", "get_git_blob"]

    async def test_an_incremental_record_never_downloads_a_tarball(self, tmp_path) -> None:
        c = make_mock_connector()
        calls = self._wire(c)
        sync = ReposSync(c, archive_cache=make_cache(tmp_path))

        record = self._record("a.py", b"a = 1\n", size=None)

        assert await sync.fetch_code_file_content(record) == b"contents-api"
        assert calls == ["get_file_contents"]

    async def test_an_incremental_record_reads_an_archive_already_on_disk(self, tmp_path) -> None:
        c = make_mock_connector()
        calls = self._wire(c)
        sync = ReposSync(c, archive_cache=make_cache(tmp_path))
        await sync.fetch_code_file_content(self._record("a.py", b"a = 1\n"))

        record = self._record("pkg/b.py", b"b = 2\n", size=None, commit="def456")

        assert await sync.fetch_code_file_content(record) == b"b = 2\n"
        assert calls == ["download_repo_tarball"]

    async def test_without_a_cache_files_are_read_by_api(self) -> None:
        c = make_mock_connector()
        calls = self._wire(c)

        assert await ReposSync(c).fetch_code_file_content(self._record("a.py", b"a = 1\n")) == b"api"
        assert calls == ["get_git_blob"]


class TestDownloadRepoTarball:
    """The API answers with a redirect to codeload; the token stays on the API hop."""

    async def test_follows_the_redirect_without_the_token_and_records_the_budget(self, tmp_path) -> None:
        import httpx

        from app.sources.external.github.github_async import GitHubAsyncDataSource

        seen: list[tuple[str, str | None]] = []
        reset = int(time.time()) + 900

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), request.headers.get("authorization")))
            if request.url.host == "api.github.com":
                return httpx.Response(302, headers={
                    "location": "https://codeload.github.com/acme/widgets/legacy.tar.gz/abc?token=t",
                    "x-ratelimit-resource": "core", "x-ratelimit-limit": "5000",
                    "x-ratelimit-remaining": "4999", "x-ratelimit-reset": str(reset),
                })
            return httpx.Response(200, content=b"tarball-bytes")

        client = SimpleNamespace(get_client=lambda: SimpleNamespace(get_token=lambda: "secret"))
        ds = GitHubAsyncDataSource(client, transport=httpx.MockTransport(handler))
        dest = tmp_path / "a.tar.gz"

        res = await ds.download_repo_tarball("acme", "widgets", "abc", str(dest), max_bytes=1024)

        assert res.success and res.data == len(b"tarball-bytes")
        assert dest.read_bytes() == b"tarball-bytes"
        assert seen[0] == ("https://api.github.com/repos/acme/widgets/tarball/abc", "Bearer secret")
        assert seen[1][0].startswith("https://codeload.github.com/") and seen[1][1] is None
        assert ds.rate_limit_budget("core") == (5000, 4999, reset)
        await ds.aclose()

    async def test_an_archive_over_the_cap_is_refused(self, tmp_path) -> None:
        import httpx

        from app.sources.external.github.github_async import GitHubAsyncDataSource

        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 64))
        client = SimpleNamespace(get_client=lambda: SimpleNamespace(get_token=lambda: "secret"))
        ds = GitHubAsyncDataSource(client, transport=transport)

        res = await ds.download_repo_tarball("acme", "widgets", "abc", str(tmp_path / "a"), max_bytes=10)

        assert not res.success
        await ds.aclose()
