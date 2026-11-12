"""Obsidian vault mode for Local FS: link parsing, resolution, and the link pass."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.config.constants.arangodb import Connectors, OriginTypes, RecordRelations
from app.connectors.core.registry.filters import FilterCollection
from app.connectors.sources.local_fs.connector import (
    OBSIDIAN_VAULT_KEY,
    SYNC_ROOT_PATH_KEY,
    LocalFsConnector,
    LocalFsDesktopOfflineError,
    _parse_obsidian_vault,
)
from app.connectors.sources.local_fs.models import (
    LocalFsFileEvent,
    LocalFsFileEventBatchStats,
)
from app.connectors.sources.local_fs.obsidian import (
    NoteLink,
    VaultIndex,
    is_note,
    is_vault_internal,
    parse_links,
)
from app.models.entities import FileRecord, RecordGroupType, RecordType, User

# The connector tests install the import shims the connector module needs.
from tests.unit.connectors.sources.local_fs.test_connector import (
    folder_connector as folder_connector,
)


class TestParseLinks:
    def test_wikilinks_aliases_headings_and_blocks_name_the_note(self) -> None:
        text = "[[Alpha]] [[Beta|the beta]] [[Gamma#Section]] [[Delta#^block|d]]"
        assert [link.target for link in parse_links(text)] == ["Alpha", "Beta", "Gamma", "Delta"]

    def test_embeds_are_marked(self) -> None:
        assert parse_links("![[diagram.png]] [[Note]]") == [
            NoteLink("diagram.png", True),
            NoteLink("Note", False),
        ]

    def test_escaped_pipe_in_a_table(self) -> None:
        assert parse_links("| [[Note\\|alias]] |") == [NoteLink("Note", False)]

    def test_code_and_comments_are_not_links(self) -> None:
        text = "```\n[[InFence]]\n```\n`[[Inline]]` <!-- [[Html]] --> %% [[Obs]] %% [[Real]]"
        assert parse_links(text) == [NoteLink("Real", False)]

    def test_frontmatter_links_count(self) -> None:
        text = '---\nrelated: "[[Other]]"\n---\nbody'
        assert parse_links(text) == [NoteLink("Other", False)]

    def test_markdown_links_to_vault_files(self) -> None:
        text = "[a](Folder/My%20Note.md) [b](<Other Note.md>) [c](https://x.com) [d](#heading) ![i](img.png)"
        assert parse_links(text) == [
            NoteLink("Folder/My Note.md", False),
            NoteLink("Other Note.md", False),
            NoteLink("img.png", True),
        ]

    def test_same_note_heading_link_is_dropped_and_duplicates_collapse(self) -> None:
        assert parse_links("[[#Heading]] [[A]] [[A|again]]") == [NoteLink("A", False)]


class TestVaultIndex:
    index = VaultIndex(
        [
            "Inbox.md",
            "Projects/Plan.md",
            "Archive/Plan.md",
            "Projects/Deep/Spec.md",
            "assets/diagram.png",
        ]
    )

    def test_bare_name_finds_the_note_anywhere(self) -> None:
        assert self.index.resolve("Spec", "Inbox.md") == "Projects/Deep/Spec.md"

    def test_case_is_ignored(self) -> None:
        assert self.index.resolve("inbox", "Projects/Plan.md") == "Inbox.md"

    def test_same_folder_wins_among_duplicates(self) -> None:
        assert self.index.resolve("Plan", "Archive/Other.md") == "Archive/Plan.md"
        assert self.index.resolve("Plan", "Projects/Other.md") == "Projects/Plan.md"

    def test_explicit_and_partial_paths(self) -> None:
        assert self.index.resolve("Archive/Plan", "Inbox.md") == "Archive/Plan.md"
        assert self.index.resolve("Deep/Spec", "Inbox.md") == "Projects/Deep/Spec.md"
        assert self.index.resolve("../Inbox.md", "Projects/Plan.md") == "Inbox.md"

    def test_attachments_by_name(self) -> None:
        assert self.index.resolve("diagram.png", "Inbox.md") == "assets/diagram.png"

    def test_unresolved(self) -> None:
        assert self.index.resolve("Missing", "Inbox.md") is None
        assert self.index.resolve("Nope/Plan", "Inbox.md") is None


@pytest.mark.parametrize(
    ("path", "internal"),
    [
        (".obsidian/app.json", True),
        (".obsidian/plugins/x/main.js", True),
        (".trash/old.md", True),
        ("sub/.obsidian/workspace.json", True),
        ("notes/obsidian.md", False),
        ("Inbox.md", False),
    ],
)
def test_is_vault_internal(path: str, internal: bool) -> None:
    assert is_vault_internal(path) is internal


def test_is_note() -> None:
    assert is_note("a/B.MD") and not is_note("a/b.png")


def test_setting_defaults_off_and_parses_strings() -> None:
    assert _parse_obsidian_vault({}) is False
    assert _parse_obsidian_vault({"sync": {OBSIDIAN_VAULT_KEY: "true"}}) is True
    assert _parse_obsidian_vault({"sync": {"values": {OBSIDIAN_VAULT_KEY: True}}}) is True


def _note(rec_id: str, path: str, *, size: int = 10) -> FileRecord:
    return FileRecord(
        id=rec_id,
        record_name=Path(path).name,
        record_type=RecordType.FILE,
        external_record_id=f"ext-{rec_id}",
        version=0,
        origin=OriginTypes.CONNECTOR,
        connector_name=Connectors.LOCAL_FS,
        connector_id="connector-instance-1",
        is_file=True,
        path=path,
        local_fs_relative_path=path,
        mime_type="text/markdown",
        record_group_type=RecordGroupType.DRIVE,
        size_in_bytes=size,
    )


def _event(path: str, event_type: str = "CREATED") -> LocalFsFileEvent:
    return LocalFsFileEvent(
        type=event_type, path=path, timestamp=1, isDirectory=False, sha256="abc", size=5
    )


class TestObsidianBatch:
    @staticmethod
    def _setup(connector: LocalFsConnector, tmp_path: Path, vault: bool) -> None:
        sync = {SYNC_ROOT_PATH_KEY: str(tmp_path)}
        if vault:
            sync[OBSIDIAN_VAULT_KEY] = True
        connector.config_service.get_config = AsyncMock(return_value={"sync": sync})
        connector.data_entities_processor.get_user_by_user_id = AsyncMock(
            return_value=User(email="u@x.com", id="u1", org_id="org-1")
        )

    async def _apply(
        self,
        connector: LocalFsConnector,
        events: list[LocalFsFileEvent],
        changed: set[str] | None = None,
    ) -> LocalFsFileEventBatchStats:
        empty = AsyncMock(return_value=(FilterCollection(filters=[]), FilterCollection(filters=[])))
        with patch("app.connectors.sources.local_fs.connector.load_connector_filters", new=empty):
            await connector._reload_sync_settings()
            root = Path(connector.sync_root_path)
            owner, rg = await connector._ensure_owner_and_record_group(root)
            return await connector._apply_file_event_batch(
                events,
                owner=owner,
                sync_filters=FilterCollection(filters=[]),
                indexing_filters=FilterCollection(filters=[]),
                external_record_group_id=rg,
                root_for_display=root,
                emitted_folder_paths=set(),
                changed_notes=changed,
            )

    async def test_vault_internal_files_are_skipped_and_notes_collected(
        self, folder_connector: LocalFsConnector, tmp_path: Path
    ) -> None:
        self._setup(folder_connector, tmp_path, vault=True)
        changed: set[str] = set()
        stats = await self._apply(
            folder_connector,
            [_event(".obsidian/app.json"), _event(".trash/x.md"), _event("Note.md"), _event("img.png")],
            changed,
        )
        assert stats.skipped == 2
        assert changed == {"Note.md"}
        written = [
            r.record_name
            for call in folder_connector.data_entities_processor.on_new_records.await_args_list
            for r, _ in call.args[0]
        ]
        assert "app.json" not in written and "Note.md" in written

    async def test_vault_mode_off_keeps_obsidian_folder(
        self, folder_connector: LocalFsConnector, tmp_path: Path
    ) -> None:
        self._setup(folder_connector, tmp_path, vault=False)
        stats = await self._apply(folder_connector, [_event(".obsidian/app.json")])
        assert stats.skipped == 0


class TestApplyObsidianLinks:
    def _vault(
        self, connector: LocalFsConnector, contents: dict[str, bytes]
    ) -> dict[str, FileRecord]:
        records = [_note(f"r{i}", path) for i, path in enumerate([*sorted(contents), "assets/d.png"])]
        by_path = {r.path: r for r in records}
        proc = connector.data_entities_processor
        proc.get_records_by_status = AsyncMock(side_effect=[records, []])
        proc.on_related_records_update = AsyncMock()

        async def fetch(record: FileRecord) -> bytes:
            return contents[record.path]

        connector._fetch_desktop_content = AsyncMock(side_effect=fetch)
        return by_path

    async def test_links_and_embeds_become_edges(self, folder_connector: LocalFsConnector) -> None:
        by_path = self._vault(
            folder_connector,
            {
                "A.md": b"See [[B]] and [[Sub/C|c]], ![[d.png]], [[Missing]], [[A]]",
                "B.md": b"back to [[A]]",
                "Sub/C.md": b"",
            },
        )
        await folder_connector._apply_obsidian_links({"A.md", "Sub/C.md"})

        updates = folder_connector.data_entities_processor.on_related_records_update.await_args.args[0]
        by_note = {note.path: related for note, related in updates}
        assert set(by_note) == {"A.md", "Sub/C.md"}
        edges = {(r.external_record_id, r.relation_type) for r in by_note["A.md"]}
        assert edges == {
            (by_path["B.md"].external_record_id, RecordRelations.LINKED_TO),
            (by_path["Sub/C.md"].external_record_id, RecordRelations.LINKED_TO),
            (by_path["assets/d.png"].external_record_id, RecordRelations.RELATED),
        }
        # A note with no links still gets an update, which clears stale edges.
        assert by_note["Sub/C.md"] == []

    async def test_unreadable_note_keeps_its_edges(self, folder_connector: LocalFsConnector) -> None:
        self._vault(folder_connector, {"A.md": b"[[B]]", "B.md": b""})
        folder_connector._fetch_desktop_content = AsyncMock(side_effect=RuntimeError("gone"))
        await folder_connector._apply_obsidian_links({"A.md"})
        folder_connector.data_entities_processor.on_related_records_update.assert_not_awaited()

    async def test_desktop_offline_stops_the_pass_without_raising(
        self, folder_connector: LocalFsConnector
    ) -> None:
        self._vault(folder_connector, {"A.md": b"[[B]]", "B.md": b""})
        folder_connector._fetch_desktop_content = AsyncMock(
            side_effect=LocalFsDesktopOfflineError("offline")
        )
        await folder_connector._apply_obsidian_links({"A.md", "B.md"})
        assert folder_connector._fetch_desktop_content.await_count == 1

    async def test_listing_failure_skips_the_pass(self, folder_connector: LocalFsConnector) -> None:
        proc = folder_connector.data_entities_processor
        proc.get_records_by_status = AsyncMock(side_effect=RuntimeError("db"))
        proc.on_related_records_update = AsyncMock()
        await folder_connector._apply_obsidian_links({"A.md"})
        proc.on_related_records_update.assert_not_awaited()
