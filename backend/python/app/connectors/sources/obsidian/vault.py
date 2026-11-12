"""Reading an Obsidian vault the way Obsidian does.

Pure functions over note text and vault paths: frontmatter, tags, links and
how a link target resolves to a file. Fetching notes and writing records and
edges is the connector's job (``connector.py``).
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Folders inside a vault that are not vault content: Obsidian's settings and
# plugins, its trash, and a git checkout of the vault.
VAULT_INTERNAL_DIRS = (".obsidian", ".trash", ".git")

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
# Frontmatter: a ``---`` line at the very top, YAML, a closing ``---``/``...``.
_FRONTMATTER = re.compile(
    r"\A﻿?---[ \t]*\r?\n(.*?)^(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
# ``#tag`` and ``#nested/tag``: at line start or after whitespace, and not all
# digits (``#123`` is not a tag in Obsidian).
_TAG = re.compile(r"(?:^|(?<=[\s(\[,]))#([\w\-/]*[^\W\d][\w\-/]*)", re.MULTILINE)

_PROPERTY_KEYS_SHOWN_ELSEWHERE = {"title", "aliases", "alias", "tags", "tag"}


@dataclass(frozen=True)
class NoteLink:
    """One link as written in a note, before resolution.

    ``subpath`` is the heading (``Heading``) or block (``^id``) after ``#``.
    """

    target: str
    embed: bool
    subpath: str = ""


@dataclass
class ParsedNote:
    title: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    links: list[NoteLink] = field(default_factory=list)
    body: str = ""


def is_vault_internal(rel_path: str) -> bool:
    """Whether a vault-relative path is inside ``.obsidian/``, ``.trash/`` or ``.git/``."""
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    return any(part in VAULT_INTERNAL_DIRS for part in parts)


def is_note(rel_path: str) -> bool:
    return posixpath.splitext(rel_path)[1].lower() == f".{NOTE_EXTENSION}"


def note_title_from_path(rel_path: str) -> str:
    name = posixpath.basename(rel_path.replace("\\", "/"))
    return name[: -len(NOTE_EXTENSION) - 1] if is_note(name) else name


def _norm(path: str) -> str:
    return unicodedata.normalize("NFC", path.replace("\\", "/")).strip("/")


class ExcludeFilters:
    """Obsidian's "Excluded files" (``userIgnoreFilters`` in ``.obsidian/app.json``).

    An entry is a path prefix (``Templates/``, ``Archive``) or, written as
    ``/pattern/``, a regular expression searched in the path.
    """

    def __init__(self, filters: Iterable[str] = ()) -> None:
        self._prefixes: list[str] = []
        self._patterns: list[re.Pattern[str]] = []
        for raw in filters or ():
            if not isinstance(raw, str) or not raw.strip():
                continue
            entry = raw.strip()
            if len(entry) > 2 and entry.startswith("/") and entry.endswith("/"):
                try:
                    self._patterns.append(re.compile(entry[1:-1]))
                except re.error:
                    continue
            else:
                prefix = _norm(entry)
                if prefix:
                    self._prefixes.append(prefix)

    def __bool__(self) -> bool:
        return bool(self._prefixes or self._patterns)

    def excludes(self, rel_path: str) -> bool:
        path = _norm(rel_path)
        if any(path.startswith(prefix) for prefix in self._prefixes):
            return True
        return any(p.search(path) for p in self._patterns)


def _strip_code_and_comments(text: str) -> str:
    text = _FENCED_CODE.sub("", text)
    text = _INLINE_CODE.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    return _OBSIDIAN_COMMENT.sub("", text)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """The note's YAML frontmatter (``{}`` when absent or invalid) and its body."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    body = text[match.end():]
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, body
    if not isinstance(data, dict):
        return {}, body
    return {str(k): v for k, v in data.items()}, body


def _as_list(value: Any) -> list[str]:  # noqa: ANN401
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in re.split(r"[,\s]+", value) if part]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_list(item))
        return out
    return [str(value)]


def _aliases(value: Any) -> list[str]:  # noqa: ANN401
    # An alias may contain spaces, so a string is one alias (or comma-separated).
    if value is None:
        return []
    if isinstance(value, str):
        return [a.strip() for a in value.split(",") if a.strip()]
    if isinstance(value, (list, tuple)):
        return [str(a).strip() for a in value if a is not None and str(a).strip()]
    return [str(value)]


def _wikilink_parts(inner: str) -> tuple[str, str]:
    # Alias first: ``[[a#b|c]]`` names ``a``. Inside a table the pipe is
    # escaped (``[[a\|c]]``); the backslash is not part of the name.
    target = inner.split("|", 1)[0].rstrip("\\")
    name, _, subpath = target.partition("#")
    return name.strip(), subpath.strip()


def parse_links(text: str) -> list[NoteLink]:
    """The links a note's text makes, in order, without duplicates.

    Links inside code blocks, inline code and comments are not links, as in
    Obsidian. Frontmatter is kept: Obsidian counts ``related: "[[Note]]"``.
    External URLs and same-note heading links (``[[#Heading]]``) are dropped.
    """
    body = _strip_code_and_comments(text)
    found: list[NoteLink] = []
    seen: set[tuple[str, bool, str]] = set()

    def add(target: str, subpath: str, *, embed: bool) -> None:
        if not target:
            return
        key = (target, embed, subpath)
        if key in seen:
            return
        seen.add(key)
        found.append(NoteLink(target=target, embed=embed, subpath=subpath))

    for match in _WIKILINK.finditer(body):
        name, subpath = _wikilink_parts(match.group(2))
        add(name, subpath, embed=bool(match.group(1)))

    for match in _MD_LINK.finditer(body):
        raw = match.group(2).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        if _URL_SCHEME.match(raw) or raw.startswith("#"):
            continue
        name, _, subpath = unquote(raw).partition("#")
        add(name.strip(), subpath.strip(), embed=bool(match.group(1)))

    return found


def parse_tags(body: str, frontmatter: Mapping[str, Any]) -> list[str]:
    """Frontmatter ``tags``/``tag`` and inline ``#tags``, without ``#``.

    Nested tags (``#a/b``) are kept whole. Tags differing only in case are one
    tag, as in Obsidian; the first spelling is kept.
    """
    tags: list[str] = []
    seen: set[str] = set()

    def add(tag: str) -> None:
        tag = tag.strip().lstrip("#").strip("/")
        if not tag or tag.isdigit() or tag.lower() in seen:
            return
        seen.add(tag.lower())
        tags.append(tag)

    for key in ("tags", "tag"):
        for tag in _as_list(frontmatter.get(key)):
            add(tag)
    for match in _TAG.finditer(_strip_code_and_comments(body)):
        add(match.group(1))
    return tags


def parse_note(rel_path: str, text: str) -> ParsedNote:
    frontmatter, body = split_frontmatter(text)
    title = frontmatter.get("title")
    aliases = _aliases(frontmatter.get("aliases")) + _aliases(frontmatter.get("alias"))
    return ParsedNote(
        title=str(title).strip() if title not in (None, "") else note_title_from_path(rel_path),
        aliases=list(dict.fromkeys(aliases)),
        tags=parse_tags(body, frontmatter),
        frontmatter=frontmatter,
        # Frontmatter links count, as in Obsidian.
        links=parse_links(text),
        body=body,
    )


def _property_text(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, (list, tuple)):
        return ", ".join(_property_text(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {_property_text(v)}" for k, v in value.items())
    return str(value)


def render_for_index(note: ParsedNote) -> str:
    """The text indexed for a note: a short properties header, then the body.

    The raw YAML block is dropped; the title, aliases, tags and every other
    property are kept as plain lines, so searching for a tag or a property
    value still finds the note and chat sees them as the note's metadata.
    """
    lines = [f"# {note.title}"]
    if note.aliases:
        lines.append(f"Aliases: {', '.join(note.aliases)}")
    if note.tags:
        lines.append(f"Tags: {' '.join('#' + t for t in note.tags)}")
    for key, value in note.frontmatter.items():
        if key.lower() in _PROPERTY_KEYS_SHOWN_ELSEWHERE or value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {_property_text(value)}")
    return "\n".join(lines) + "\n\n" + note.body.lstrip("\r\n")


def new_note_path(target: str, source_path: str, settings: Mapping[str, Any] | None) -> str:
    """Where Obsidian creates the note an unresolved ``[[target]]`` names.

    Follows "Default location for new notes" in ``.obsidian/app.json``: the
    vault root (default), the linking note's folder (``current``), or a fixed
    folder (``folder`` with ``newFileFolderPath``). A target with a folder in
    it is a path from the vault root.
    """
    target = _norm(target)
    if not posixpath.splitext(target)[1]:
        target = f"{target}.{NOTE_EXTENSION}"
    if "/" in target:
        return posixpath.normpath(target)
    settings = settings or {}
    location = settings.get("newFileLocation")
    if location == "current":
        folder = posixpath.dirname(_norm(source_path))
    elif location == "folder":
        folder = _norm(str(settings.get("newFileFolderPath") or ""))
    else:
        folder = ""
    return posixpath.join(folder, target) if folder else target


class VaultIndex:
    """The vault's file paths, for resolving link targets as Obsidian does.

    ``[[Note]]`` names ``Note.md`` anywhere in the vault. When several files
    share the name, the one in the linking note's own folder wins, then the
    one with the shortest path. ``[[folder/Note]]`` names a path relative to
    the vault root or the linking note's folder, or any path ending so. A
    note's frontmatter aliases name it too. Matching ignores case, like
    Obsidian on macOS and Windows.
    """

    def __init__(
        self,
        paths: Iterable[str],
        aliases: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._exact: dict[str, str] = {}
        self._by_name: dict[str, list[str]] = {}
        self._by_alias: dict[str, list[str]] = {}
        for path in paths:
            norm = _norm(path)
            if not norm:
                continue
            self._exact.setdefault(norm.lower(), norm)
            name = posixpath.basename(norm).lower()
            self._by_name.setdefault(name, []).append(norm)
        for path, names in (aliases or {}).items():
            norm = self._exact.get(_norm(path).lower())
            if norm is None:
                continue
            for alias in names:
                self._by_alias.setdefault(alias.strip().lower(), []).append(norm)

    def __len__(self) -> int:
        return len(self._exact)

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and _norm(path).lower() in self._exact

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
        matches = self._by_alias.get(target.lower())
        if matches:
            return self._pick(matches, source_dir)
        return None

    @staticmethod
    def _pick(matches: list[str], source_dir: str) -> str:
        same_folder = [p for p in matches if posixpath.dirname(p) == source_dir]
        if same_folder:
            return min(same_folder)
        return min(matches, key=lambda p: (p.count("/"), len(p), p))
