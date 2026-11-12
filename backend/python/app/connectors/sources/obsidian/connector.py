"""Obsidian: sync an Obsidian vault and keep its knowledge graph.

The vault is a folder on the user's machine, read by the PipesHub desktop app
exactly as Local FS reads one: the server pulls file events and file bytes
through the Node desktop relay. This connector reuses that transport whole
(it subclasses ``LocalFsConnector``) and adds what makes a vault a graph:

* Notes are records named by their title; folders are the PARENT_CHILD tree.
* ``[[links]]`` and markdown links between notes become LINKED_TO edges, and
  ``![[embeds]]`` (images, PDFs, other notes) RELATED edges to the embedded
  file's own record, resolved the way Obsidian resolves them (path, same
  folder, shortest path, frontmatter aliases, any case).
* A link to a note that does not exist yet points at a placeholder record at
  the path Obsidian would create it; when the note appears there, the record
  takes the placeholder's place and keeps the edge. Created anywhere else, the
  linking notes are re-read and re-linked in the same run.
* Indexed note text drops the YAML block and leads with the title, aliases,
  tags (inline and frontmatter) and other properties as plain lines.
* ``.obsidian/``, ``.trash/``, ``.git/`` and the vault's own "Excluded files"
  are never synced.
"""

from __future__ import annotations

import json
import unicodedata
from typing import TYPE_CHECKING, Any, Optional

from app.config.constants.arangodb import (
    AppGroups,
    Connectors,
    OriginTypes,
    PermissionModel,
    ProgressStatus,
    RecordRelations,
)
from app.connectors.core.base.sync_point.sync_point import (
    generate_record_sync_point_key,
)
from app.connectors.core.registry.connector_builder import (
    CommonFields,
    ConnectorBuilder,
    ConnectorScope,
    CustomField,
    DocumentationLink,
    SyncStrategy,
)
from app.connectors.sources.local_fs.connector import (
    FULL_SYNC_RESET_BATCH_SIZE,
    INCLUDE_SUBFOLDERS_KEY,
    LOCAL_FS_STORAGE_PATH_PREFIX,
    SYNC_ROOT_PATH_KEY,
    LocalFsConnector,
    LocalFsDesktopOfflineError,
)
from app.connectors.sources.obsidian.vault import (
    ExcludeFilters,
    NoteLink,
    VaultIndex,
    is_note,
    is_vault_internal,
    new_note_path,
    note_title_from_path,
    parse_note,
    render_for_index,
)
from app.models.entities import FileRecord, Record, RecordType, RelatedExternalRecord

if TYPE_CHECKING:
    from app.connectors.sources.obsidian.vault import ParsedNote

OBSIDIAN_CONNECTOR_NAME = "Obsidian"
OBSIDIAN_ICON_PATH = "/icons/connectors/obsidian.svg"
OBSIDIAN_APP_SETTINGS_PATH = ".obsidian/app.json"
# Largest note the link pass reads. The pass fetches each changed note over the
# desktop relay, and nothing written by hand is this big.
OBSIDIAN_MAX_NOTE_BYTES = 2 * 1024 * 1024
# A link pass re-reads notes whose links a run's new or deleted notes affect;
# this caps how many rounds of that one run may do.
_MAX_RELINK_ROUNDS = 3


def _norm(path: str) -> str:
    return unicodedata.normalize("NFC", path.strip().replace("\\", "/")).strip("/")


@(
    ConnectorBuilder(OBSIDIAN_CONNECTOR_NAME)
    .in_group(AppGroups.LOCAL_STORAGE.value)
    .with_supported_auth_types("NONE")
    .with_description(
        "Index an Obsidian vault on your machine with its links: notes, "
        "[[wikilinks]], embeds, tags and properties."
    )
    .with_info(
        "Syncs through the PipesHub desktop app, which must be running and "
        "connected. Links between notes become edges in the knowledge graph. "
        "The .obsidian, .trash and .git folders and the vault's Excluded files "
        "are skipped."
    )
    .with_categories(["Storage", "Local", "Notes"])
    .with_scopes([ConnectorScope.PERSONAL.value])
    .with_permission_model(PermissionModel.APP_LEVEL)
    .configure(
        lambda builder: builder.with_icon(OBSIDIAN_ICON_PATH)
        .with_realtime_support(False)
        .add_documentation_link(
            DocumentationLink(
                "Pipeshub documentation",
                "https://docs.pipeshub.com",
                "pipeshub",
            )
        )
        .with_sync_strategies([SyncStrategy.SCHEDULED, SyncStrategy.MANUAL], selected=SyncStrategy.SCHEDULED)
        .with_scheduled_config(True, 60)
        .with_sync_support(True)
        .with_agent_support(False)
        .with_hide_connector(False)
        .add_sync_custom_field(
            CustomField(
                name=SYNC_ROOT_PATH_KEY,
                display_name="Vault folder",
                field_type="FOLDER",
                required=True,
                description="Choose the Obsidian vault folder on this machine.",
            )
        )
        .add_sync_custom_field(
            CustomField(
                name=INCLUDE_SUBFOLDERS_KEY,
                display_name="Include subfolders",
                field_type="BOOLEAN",
                required=False,
                default_value="true",
                description="Recurse into the vault's folders.",
            )
        )
        .add_sync_custom_field(CommonFields.batch_size_field())
        .add_filter_field(
            CommonFields.modified_date_filter(
                "Only sync files modified within this range (optional)."
            )
        )
        .add_filter_field(CommonFields.enable_manual_sync_filter())
        .add_filter_field(CommonFields.file_extension_filter())
    )
    .build_decorator()
)
class ObsidianConnector(LocalFsConnector):
    CONNECTOR = Connectors.OBSIDIAN
    CONNECTOR_KEY = "obsidian"
    RECORD_GROUP_PREFIX = "obsidian"

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self._vault_settings: dict[str, Any] = {}
        self._exclude = ExcludeFilters()
        self._reset_run_state()

    # -- what a run collects -------------------------------------------------

    def _reset_run_state(self) -> None:
        self._changed_notes: set[str] = set()
        self._created_paths: set[str] = set()
        self._removed_paths: set[str] = set()

    def _on_run_restarted(self) -> None:
        self._reset_run_state()

    def _skip_rel_path(self, rel_path: str) -> bool:
        return is_vault_internal(rel_path) or self._exclude.excludes(rel_path)

    def _record_name_for(self, rel_path: str) -> str:
        return note_title_from_path(rel_path) or "note"

    def _on_file_upserted(self, rel_path: str, old_rel_path: str) -> None:
        path = _norm(rel_path)
        self._created_paths.add(path)
        self._removed_paths.discard(path)
        if is_note(path):
            self._changed_notes.add(path)
        if old_rel_path and _norm(old_rel_path) != path:
            self._removed_paths.add(_norm(old_rel_path))
            self._changed_notes.discard(_norm(old_rel_path))

    def _on_file_deleted(self, rel_path: str) -> None:
        path = _norm(rel_path)
        self._removed_paths.add(path)
        self._changed_notes.discard(path)
        self._created_paths.discard(path)

    # -- sync ----------------------------------------------------------------

    async def run_sync(self) -> None:
        self._reset_run_state()
        await self._reload_sync_settings()
        await self._load_vault_settings()
        await super().run_sync()

    async def _load_vault_settings(self) -> None:
        """Read ``.obsidian/app.json``: excluded files and where new notes go.

        Missing or unreadable settings mean Obsidian's defaults. An offline
        desktop is left for the run itself to report.
        """
        probe = FileRecord(
            record_name="app.json",
            record_type=RecordType.FILE,
            external_record_id=self._external_record_id_for_rel_path(OBSIDIAN_APP_SETTINGS_PATH),
            version=0,
            origin=OriginTypes.CONNECTOR,
            connector_name=self.CONNECTOR,
            connector_id=self.connector_id,
            is_file=True,
            path=OBSIDIAN_APP_SETTINGS_PATH,
            local_fs_relative_path=OBSIDIAN_APP_SETTINGS_PATH,
        )
        try:
            raw = await self._fetch_desktop_content(probe)
            settings = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self.logger.info("Obsidian: using default vault settings (%s)", exc)
            settings = {}
        if not isinstance(settings, dict):
            settings = {}
        self._vault_settings = settings
        self._exclude = ExcludeFilters(settings.get("userIgnoreFilters") or [])

    async def _transform_content(self, record: FileRecord, content: bytes) -> bytes:
        rel_path = record.local_fs_relative_path or record.path or ""
        if not is_note(rel_path):
            return content
        note = parse_note(rel_path, content.decode("utf-8", errors="replace"))
        return render_for_index(note).encode("utf-8")

    # -- the link graph ------------------------------------------------------

    def _graph_state_key(self) -> str:
        return generate_record_sync_point_key(
            RecordType.FILE.value, f"{self.CONNECTOR_KEY}-graph", self.connector_id
        )

    @staticmethod
    def _vault_rel_path(record: Record) -> Optional[str]:
        if not isinstance(record, FileRecord) or not record.is_file:
            return None
        raw = record.local_fs_relative_path or record.path
        if not raw or raw.startswith(LOCAL_FS_STORAGE_PATH_PREFIX):
            return None
        return _norm(raw)

    async def _list_vault_records(self) -> dict[str, Record]:
        status_filters = [status.value for status in ProgressStatus]
        by_path: dict[str, Record] = {}
        offset = 0
        while True:
            records = await self.data_entities_processor.get_records_by_status(
                self.connector_id,
                status_filters,
                limit=FULL_SYNC_RESET_BATCH_SIZE,
                offset=offset,
            )
            if not records:
                break
            for record in records:
                rel_path = self._vault_rel_path(record)
                if rel_path and not getattr(record, "is_placeholder", False):
                    by_path[rel_path] = record
            offset += len(records)
        return by_path

    async def _on_run_complete(self, mode: str) -> None:
        """Store the links of the notes this run touched as edges.

        Never fails the sync: the files are indexed either way, and a note whose
        links could not be read keeps the edges it had until it changes again
        or a full sync reads it.
        """
        try:
            await self._apply_links(mode)
        except LocalFsDesktopOfflineError:
            self.logger.warning("Obsidian: desktop went offline during the link pass")
        except Exception as exc:
            self.logger.warning("Obsidian: link pass failed: %s", exc, exc_info=True)

    async def _apply_links(self, mode: str) -> None:
        if not (self._changed_notes or self._removed_paths or self._created_paths):
            return
        by_path = await self._list_vault_records()

        state = {} if mode == "FULL" else await self.record_sync_point.read_sync_point(self._graph_state_key())
        aliases: dict[str, list[str]] = dict(state.get("aliases") or {})
        # source note -> link targets as written that resolved to nothing
        unresolved: dict[str, list[str]] = dict(state.get("unresolved") or {})
        # source note -> vault paths its links resolved to
        links: dict[str, list[str]] = dict(state.get("links") or {})

        for path in list(aliases):
            if path not in by_path:
                aliases.pop(path)
        for table in (unresolved, links):
            for path in list(table):
                if path not in by_path:
                    table.pop(path)

        parsed: dict[str, ParsedNote] = {}

        async def read(paths: set[str]) -> None:
            for rel_path in sorted(paths):
                note = by_path.get(rel_path)
                if note is None or rel_path in parsed:
                    continue
                if (getattr(note, "size_in_bytes", 0) or 0) > OBSIDIAN_MAX_NOTE_BYTES:
                    continue
                try:
                    content = await self._fetch_desktop_content(note)
                except LocalFsDesktopOfflineError:
                    raise
                except Exception as exc:
                    self.logger.warning("Obsidian: could not read %s for its links: %s", rel_path, exc)
                    continue
                parsed[rel_path] = parse_note(rel_path, content.decode("utf-8", errors="replace"))
                aliases[rel_path] = parsed[rel_path].aliases
                if not aliases[rel_path]:
                    aliases.pop(rel_path)

        to_read = {p for p in self._changed_notes if p in by_path}
        await read(to_read)

        # Notes elsewhere whose links this run's changes affect: a link that
        # named nothing may name a note created now (or a new alias), and a
        # link that named a removed file now names nothing.
        for _round in range(_MAX_RELINK_ROUNDS):
            index = VaultIndex(by_path.keys(), aliases)
            affected: set[str] = set()
            for source, targets in unresolved.items():
                if source in parsed:
                    continue
                if any(index.resolve(t, source) for t in targets):
                    affected.add(source)
            if self._removed_paths:
                for source, targets in links.items():
                    if source not in parsed and any(t in self._removed_paths for t in targets):
                        affected.add(source)
            affected &= by_path.keys()
            if not affected:
                break
            await read(affected)

        index = VaultIndex(by_path.keys(), aliases)
        updates: list[tuple[Record, list[RelatedExternalRecord]]] = []
        edge_count = 0
        placeholder_count = 0
        for rel_path, note in sorted(parsed.items()):
            source_record = by_path[rel_path]
            # One edge per target and kind; a link into a heading or block of
            # the target adds that subpath to the edge.
            edges: dict[tuple[str, bool], RelatedExternalRecord] = {}
            missing: list[str] = []
            resolved_paths: list[str] = []

            def add_edge(
                ext_id: str, name: str, link: NoteLink,
                edges: dict[tuple[str, bool], RelatedExternalRecord] = edges,
            ) -> None:
                key = (ext_id, link.embed)
                edge = edges.get(key)
                if edge is None:
                    edge = edges[key] = RelatedExternalRecord(
                        external_record_id=ext_id,
                        record_type=RecordType.FILE,
                        record_name=name,
                        relation_type=RecordRelations.RELATED if link.embed else RecordRelations.LINKED_TO,
                    )
                if link.subpath:
                    subpaths = edge.link_subpaths or []
                    if link.subpath not in subpaths:
                        edge.link_subpaths = [*subpaths, link.subpath]

            for link in note.links:
                target_path = index.resolve(link.target, rel_path)
                if target_path is not None:
                    target = by_path[target_path]
                    if target.id == source_record.id:
                        continue
                    resolved_paths.append(target_path)
                    add_edge(target.external_record_id, target.record_name, link)
                    continue
                missing.append(link.target)
                if link.embed:
                    # Obsidian creates no file for a missing embed.
                    continue
                future_path = new_note_path(link.target, rel_path, self._vault_settings)
                if self._skip_rel_path(future_path):
                    continue
                ext_id = self._external_record_id_for_rel_path(future_path)
                if (ext_id, False) not in edges:
                    placeholder_count += 1
                add_edge(ext_id, note_title_from_path(future_path), link)
            related = list(edges.values())
            edge_count += len(related)
            updates.append((source_record, related))
            links[rel_path] = sorted(set(resolved_paths))
            if missing:
                unresolved[rel_path] = list(dict.fromkeys(missing))
            else:
                unresolved.pop(rel_path, None)
            if not links[rel_path]:
                links.pop(rel_path)

        if updates:
            await self.data_entities_processor.on_related_records_update(updates)
        await self.record_sync_point.update_sync_point(
            self._graph_state_key(),
            {"aliases": aliases, "unresolved": unresolved, "links": links},
        )
        self.logger.info(
            "Obsidian: links stored for %d note(s): %d edge(s), %d to notes not created yet",
            len(updates),
            edge_count,
            placeholder_count,
        )

