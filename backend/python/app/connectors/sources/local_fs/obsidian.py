"""Obsidian vault support for Local FS.

When a Local FS connector is marked as an Obsidian vault, notes keep the links
between them: each ``[[wikilink]]`` becomes a LINKED_TO edge between the two
note records, and each ``![[embed]]`` a RELATED edge to the embedded file. The
knowledge-graph navigator follows both.

This module is pure: it parses note text and resolves link targets against the
vault's paths the way Obsidian does. Reading notes and writing edges is the
connector's job.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import unquote

if TYPE_CHECKING:
    from collections.abc import Iterable

# Folders Obsidian keeps inside the vault that are not notes: app settings and
# plugins, and the vault's own trash.
VAULT_INTERNAL_DIRS = (".obsidian", ".trash")

NOTE_EXTENSION = "md"

# ``[[target]]``, ``[[target|alias]]``, ``[[target#heading]]``,
# ``[[target#^block]]``, each optionally embedded with a leading ``!``.
_WIKILINK = re.compile(r"(!?)\[\[([^\[\]\n]+?)\]\]")
# ``[text](target)`` and ``![alt](target)``. The target may be wrapped in
# angle brackets when it has spaces: ``[text](<my note.md>)``.
_MD_LINK = re.compile(r"(!?)\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+\"[^\"\n]*\")?\s*\)")
_FENCED_CODE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_OBSIDIAN_COMMENT = re.compile(r"%%.*?%%", re.DOTALL)
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


@dataclass(frozen=True)
class NoteLink:
    """One link as written in a note: the target before resolution."""

    target: str
    embed: bool


def is_vault_internal(rel_path: str) -> bool:
    """Whether a vault-relative path is inside ``.obsidian/`` or ``.trash/``."""
    parts = rel_path.replace("\\", "/").split("/")
    return any(part in VAULT_INTERNAL_DIRS for part in parts[:-1]) or (
        len(parts) == 1 and parts[0] in VAULT_INTERNAL_DIRS
    )


def is_note(rel_path: str) -> bool:
    return posixpath.splitext(rel_path)[1].lower() == f".{NOTE_EXTENSION}"


def _strip_code_and_comments(text: str) -> str:
    text = _FENCED_CODE.sub("", text)
    text = _INLINE_CODE.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    return _OBSIDIAN_COMMENT.sub("", text)


def _wikilink_target(inner: str) -> str:
    # Alias first: ``[[a#b|c]]`` names ``a``. Then the heading or block part.
    # Inside a table the pipe is escaped (``[[a\|c]]``); the backslash is not
    # part of the name.
    target = inner.split("|", 1)[0].rstrip("\\")
    target = target.split("#", 1)[0]
    return target.strip()


def parse_links(text: str) -> list[NoteLink]:
    """The links a note's text makes, in order, without duplicates.

    Links inside code blocks, inline code and comments are not links, as in
    Obsidian. Frontmatter is kept: Obsidian counts ``related: "[[Note]]"``.
    External URLs and same-note heading links (``[[#Heading]]``) are dropped.
    """
    body = _strip_code_and_comments(text)
    found: list[NoteLink] = []
    seen: set[tuple[str, bool]] = set()

    def add(target: str, *, embed: bool) -> None:
        if not target:
            return
        key = (target, embed)
        if key in seen:
            return
        seen.add(key)
        found.append(NoteLink(target=target, embed=embed))

    for match in _WIKILINK.finditer(body):
        add(_wikilink_target(match.group(2)), embed=bool(match.group(1)))

    for match in _MD_LINK.finditer(body):
        raw = match.group(2).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        if _URL_SCHEME.match(raw) or raw.startswith("#"):
            continue
        add(unquote(raw.split("#", 1)[0]).strip(), embed=bool(match.group(1)))

    return found


def _norm(path: str) -> str:
    return unicodedata.normalize("NFC", path.replace("\\", "/")).strip("/")


class VaultIndex:
    """The vault's file paths, for resolving link targets as Obsidian does.

    ``[[Note]]`` names ``Note.md`` anywhere in the vault. When several files
    share the name, the one in the linking note's own folder wins, then the
    one with the shortest path. ``[[folder/Note]]`` names a path relative to
    the vault root, or to the linking note's folder. Matching ignores case,
    like Obsidian on macOS and Windows.
    """

    def __init__(self, paths: Iterable[str]) -> None:
        self._exact: dict[str, str] = {}
        self._by_name: dict[str, list[str]] = {}
        for path in paths:
            norm = _norm(path)
            if not norm:
                continue
            self._exact.setdefault(norm.lower(), norm)
            name = posixpath.basename(norm).lower()
            self._by_name.setdefault(name, []).append(norm)

    def __len__(self) -> int:
        return len(self._exact)

    def resolve(self, target: str, source_path: str) -> str | None:
        """The vault-relative path ``target`` names from ``source_path``, or None."""
        target = _norm(target)
        if not target:
            return None
        has_ext = bool(posixpath.splitext(target)[1])
        candidates = [target] if has_ext else [f"{target}.{NOTE_EXTENSION}", target]
        source_dir = posixpath.dirname(_norm(source_path))

        if "/" in target:
            for candidate in candidates:
                for base in (source_dir, ""):
                    joined = posixpath.normpath(posixpath.join(base, candidate))
                    if joined.startswith(".."):
                        continue
                    hit = self._exact.get(joined.lower())
                    if hit:
                        return hit
            # Obsidian also resolves a partial path by its tail: ``[[b/Note]]``
            # finds ``a/b/Note.md``.
            for candidate in candidates:
                tail = "/" + candidate.lower()
                matches = [
                    p for p in self._by_name.get(posixpath.basename(candidate).lower(), [])
                    if ("/" + p.lower()).endswith(tail)
                ]
                if matches:
                    return self._pick(matches, source_dir)
            return None

        for candidate in candidates:
            matches = self._by_name.get(candidate.lower())
            if matches:
                return self._pick(matches, source_dir)
        return None

    @staticmethod
    def _pick(matches: list[str], source_dir: str) -> str:
        same_folder = [p for p in matches if posixpath.dirname(p) == source_dir]
        if same_folder:
            return min(same_folder)
        return min(matches, key=lambda p: (p.count("/"), len(p), p))
