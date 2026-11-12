"""Obsidian connector: vault parsing, link resolution, and the link pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.constants.arangodb import Connectors, OriginTypes, RecordRelations
from app.connectors.api.router import _is_local_fs_connector_type
from app.connectors.core.registry.filters import FilterCollection
from app.connectors.sources.local_fs.connector import (
    SYNC_ROOT_PATH_KEY,
    LocalFsDesktopOfflineError,
)
from app.connectors.sources.local_fs.models import LocalFsFileEvent
from app.connectors.sources.obsidian.connector import ObsidianConnector, _decode_graph_state
from app.connectors.sources.obsidian.vault import (
    ExcludeFilters,
    NoteLink,
    VaultIndex,
    is_note,
    is_vault_internal,
    new_note_path,
    parse_links,
    parse_note,
    parse_tags,
    render_for_index,
    split_frontmatter,
    tags_from_indexed_text,
)
from app.models.entities import FileRecord, RecordGroupType, RecordType, User

# Importing the Local FS connector tests installs the import shims it needs.
from tests.unit.connectors.sources.local_fs.test_connector import (
    OWNER_DEVICE_ID,
    _app_metadata,
)

CONNECTOR_ID = "connector-instance-1"


# -- parsing -------------------------------------------------------------------


class TestParseLinks:
    def test_wikilinks_aliases_headings_and_blocks(self) -> None:
        text = "[[Alpha]] [[Beta|the beta]] [[Gamma#Section]] [[Delta#^block|d]]"
        assert parse_links(text) == [
            NoteLink("Alpha", False),
            NoteLink("Beta", False),
            NoteLink("Gamma", False, "Section"),
            NoteLink("Delta", False, "^block"),
        ]

    def test_embeds_are_marked(self) -> None:
        links = parse_links("![[diagram.png]] [[Note]]")
        assert [(link.target, link.embed) for link in links] == [
            ("diagram.png", True),
            ("Note", False),
        ]

    def test_escaped_pipe_in_a_table(self) -> None:
        assert [link.target for link in parse_links("| [[Note\\|alias]] |")] == ["Note"]

    def test_code_and_comments_are_not_links(self) -> None:
        text = "```\n[[InFence]]\n```\n`[[Inline]]` <!-- [[Html]] --> %% [[Obs]] %% [[Real]]"
        assert [link.target for link in parse_links(text)] == ["Real"]

    def test_markdown_links_to_vault_files(self) -> None:
        text = "[a](Folder/My%20Note.md) [b](<Other Note.md>) [c](https://x.com) [d](#heading) ![i](img.png)"
        assert [(link.target, link.embed) for link in parse_links(text)] == [
            ("Folder/My Note.md", False),
            ("Other Note.md", False),
            ("img.png", True),
        ]

    def test_same_note_heading_link_is_dropped(self) -> None:
        assert [link.target for link in parse_links("[[#Heading]] [[A]]")] == ["A"]


class TestNoteParsing:
    TEXT = (
        "---\n"
        "title: Real Title\n"
        "aliases: [Alt, Other]\n"
        "tags: [project/alpha]\n"
        "status: draft\n"
        "---\n"
        "Body with #inline and #nested/tag and #123 and [[Link]].\n"
    )

    def test_frontmatter_is_split_off(self) -> None:
        meta, body = split_frontmatter(self.TEXT)
        assert meta["status"] == "draft"
        assert body.startswith("Body with")

    def test_no_or_broken_frontmatter_keeps_text(self) -> None:
        assert split_frontmatter("plain") == ({}, "plain")
        meta, _body = split_frontmatter("---\n: [bad\n---\nbody")
        assert meta == {}

    def test_title_aliases_tags(self) -> None:
        note = parse_note("Folder/file name.md", self.TEXT)
        assert note.title == "Real Title"
        assert note.aliases == ["Alt", "Other"]
        assert note.tags == ["project/alpha", "inline", "nested/tag"]
        assert [link.target for link in note.links] == ["Link"]

    def test_title_falls_back_to_file_name(self) -> None:
        assert parse_note("Folder/My Note.md", "x").title == "My Note"

    def test_tags_dedupe_ignoring_case(self) -> None:
        assert parse_tags("#Tag #tag", {"tags": "tag, other"}) == ["tag", "other"]

    def test_indexed_text_drops_yaml_keeps_properties(self) -> None:
        rendered = render_for_index(parse_note("a.md", self.TEXT))
        assert rendered.startswith("# Real Title\n")
        assert "Aliases: Alt, Other" in rendered
        assert "#project/alpha" in rendered
        assert "status: draft" in rendered
        assert "---" not in rendered
        assert rendered.rstrip().endswith("[[Link]].")


def test_tags_round_trip_through_the_indexed_header() -> None:
    rendered = render_for_index(parse_note("a.md", "---\ntags: [a/b, C]\n---\nbody #d"))
    assert tags_from_indexed_text(rendered) == ["a/b", "C", "d"]
    # A markdown parser may fold the header lines into one paragraph.
    assert tags_from_indexed_text("A Aliases: x Tags: #one #two") == ["one", "two"]
    assert tags_from_indexed_text("no header") == []


class TestVaultIndex:
    index = VaultIndex(
        [
            "Inbox.md",
            "Projects/Plan.md",
            "Archive/Plan.md",
            "Projects/Deep/Spec.md",
            "assets/diagram.png",
        ],
        aliases={"Projects/Deep/Spec.md": ["The Spec"]},
    )

    def test_bare_name_finds_the_note_anywhere(self) -> None:
        assert self.index.resolve("Spec", "Inbox.md") == "Projects/Deep/Spec.md"

    def test_case_is_ignored(self) -> None:
        assert self.index.resolve("inbox", "Projects/Plan.md") == "Inbox.md"

    def test_same_folder_wins(self) -> None:
        assert self.index.resolve("Plan", "Archive/Other.md") == "Archive/Plan.md"
        assert self.index.resolve("Plan", "Projects/Other.md") == "Projects/Plan.md"

    def test_explicit_and_partial_paths(self) -> None:
        assert self.index.resolve("Archive/Plan", "Inbox.md") == "Archive/Plan.md"
        assert self.index.resolve("Deep/Spec", "Inbox.md") == "Projects/Deep/Spec.md"
        assert self.index.resolve("../Inbox.md", "Projects/Plan.md") == "Inbox.md"

    def test_aliases(self) -> None:
        assert self.index.resolve("the spec", "Inbox.md") == "Projects/Deep/Spec.md"

    def test_attachments_by_name(self) -> None:
        assert self.index.resolve("diagram.png", "Inbox.md") == "assets/diagram.png"

    def test_unresolved(self) -> None:
        assert self.index.resolve("Missing", "Inbox.md") is None
        assert self.index.resolve("Nope/Plan", "Inbox.md") is None


@pytest.mark.parametrize(
    ("path", "internal"),
    [
        (".obsidian/app.json", True),
        (".trash/old.md", True),
        (".git/HEAD", True),
        ("sub/.obsidian/workspace.json", True),
        ("notes/obsidian.md", False),
    ],
)
def test_is_vault_internal(path: str, internal: bool) -> None:
    assert is_vault_internal(path) is internal


def test_is_note() -> None:
    assert is_note("a/B.MD") and not is_note("a/b.png")


def test_exclude_filters_prefixes_and_regex() -> None:
    filters = ExcludeFilters(["Templates/", "/\\.excalidraw\\.md$/", "  "])
    assert filters.excludes("Templates/Daily.md")
    assert filters.excludes("x/drawing.excalidraw.md")
    assert not filters.excludes("Notes/Templates.md")
    assert not ExcludeFilters()


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        (None, "New.md"),
        ({"newFileLocation": "current"}, "Projects/New.md"),
        ({"newFileLocation": "folder", "newFileFolderPath": "Inbox"}, "Inbox/New.md"),
    ],
)
def test_new_note_path(settings: dict[str, Any] | None, expected: str) -> None:
    assert new_note_path("New", "Projects/Source.md", settings) == expected
    assert new_note_path("Deep/New", "Projects/Source.md", settings) == "Deep/New.md"


def test_router_treats_obsidian_as_a_desktop_folder_connector() -> None:
    assert _is_local_fs_connector_type(Connectors.OBSIDIAN.value)
    assert _is_local_fs_connector_type(Connectors.LOCAL_FS.value)
    assert not _is_local_fs_connector_type("Drive")


# -- the connector ---------------------------------------------------------------


@pytest.fixture
def vault() -> ObsidianConnector:
    proc = MagicMock()
    proc.org_id = "org-1"
    proc.get_user_by_user_id = AsyncMock(
        return_value=User(email="u@x.com", id="u1", org_id="org-1")
    )
    proc.get_record_by_external_id = AsyncMock(return_value=None)
    proc.get_file_record_by_id = AsyncMock(return_value=None)
    proc.get_records_by_status = AsyncMock(return_value=[])
    proc.on_record_deleted = AsyncMock(return_value=False)
    proc.on_new_app_users = AsyncMock()
    proc.on_new_record_groups = AsyncMock()
    proc.on_new_records = AsyncMock()
    proc.on_records_moved = AsyncMock()
    proc.reindex_existing_records = AsyncMock()
    proc.on_related_records_update = AsyncMock()
    proc.get_app_by_id = AsyncMock(return_value=_app_metadata(OWNER_DEVICE_ID))
    connector = ObsidianConnector(
        MagicMock(), proc, MagicMock(), MagicMock(), CONNECTOR_ID, "personal", "test-user"
    )
    store: dict[str, dict[str, Any]] = {}

    async def read(key: str) -> dict[str, Any]:
        return dict(store.get(key, {}))

    async def update(key: str, data: dict[str, Any]) -> None:
        # Neo4j rejects map-valued properties, as the real sync point store does.
        for name, value in data.items():
            items = value if isinstance(value, list) else [value]
            assert all(v is None or isinstance(v, (str, int, float, bool)) for v in items), (
                f"sync point property {name!r} is not a Neo4j primitive"
            )
        store.setdefault(key, {}).update(data)

    connector.record_sync_point = MagicMock()
    connector.record_sync_point.read_sync_point = AsyncMock(side_effect=read)
    connector.record_sync_point.update_sync_point = AsyncMock(side_effect=update)
    connector.graph_store = store  # type: ignore[attr-defined]
    return connector


def _file(connector: ObsidianConnector, rec_id: str, path: str) -> FileRecord:
    return FileRecord(
        id=rec_id,
        record_name=Path(path).stem,
        record_type=RecordType.FILE,
        external_record_id=connector._external_record_id_for_rel_path(path),
        version=0,
        origin=OriginTypes.CONNECTOR,
        connector_name=Connectors.OBSIDIAN,
        connector_id=CONNECTOR_ID,
        is_file=True,
        path=path,
        local_fs_relative_path=path,
        mime_type="text/markdown",
        record_group_type=RecordGroupType.DRIVE,
        size_in_bytes=10,
    )


class Vault:
    """A fake vault: the records the graph holds and the bytes on disk."""

    def __init__(self, connector: ObsidianConnector, files: dict[str, str]) -> None:
        self.connector = connector
        self.files = dict(files)
        connector._fetch_desktop_content = AsyncMock(side_effect=self._fetch)

    async def _fetch(self, record: FileRecord) -> bytes:
        return self.files[record.local_fs_relative_path].encode()

    async def run(
        self,
        *,
        changed: set[str] | None = None,
        removed: set[str] | None = None,
        moved: dict[str, str] | None = None,
        mode: str = "INCREMENTAL",
    ) -> dict[str, list]:
        c = self.connector
        records = [_file(c, f"r{i}", p) for i, p in enumerate(sorted(self.files))]
        proc = c.data_entities_processor
        proc.get_records_by_status = AsyncMock(side_effect=[records, []])
        proc.on_related_records_update.reset_mock()
        c._reset_run_state()
        for path in changed or ():
            c._on_file_upserted(path, "")
        for path in removed or ():
            c._on_file_deleted(path)
        for old, new in (moved or {}).items():
            c._on_file_upserted(new, old)
        await c._on_run_complete(mode)
        call = proc.on_related_records_update.await_args
        if call is None:
            return {}
        return {note.local_fs_relative_path: related for note, related in call.args[0]}


def _edges(related: list) -> set[tuple[str, RecordRelations]]:
    return {(r.record_name, r.relation_type) for r in related}


def _ext(connector: ObsidianConnector, path: str) -> str:
    return connector._external_record_id_for_rel_path(path)


class TestLinkPass:
    async def test_links_and_embeds_become_edges(self, vault: ObsidianConnector) -> None:
        fake = Vault(
            vault,
            {
                "A.md": "See [[B#Part]] and [[Sub/C|c]], ![[d.png]], [[A]]",
                "B.md": "back to [[A]]",
                "Sub/C.md": "",
                "assets/d.png": "",
            },
        )
        updates = await fake.run(changed={"A.md", "Sub/C.md"}, mode="FULL")
        assert set(updates) == {"A.md", "Sub/C.md"}
        assert _edges(updates["A.md"]) == {
            ("B", RecordRelations.LINKED_TO),
            ("C", RecordRelations.LINKED_TO),
            ("d", RecordRelations.RELATED),
        }
        # A note with no links still gets an update, which clears stale edges.
        assert updates["Sub/C.md"] == []

    async def test_heading_and_block_links_are_kept_on_one_edge(
        self, vault: ObsidianConnector
    ) -> None:
        fake = Vault(vault, {"A.md": "[[B#Intro]] [[B#^abc]] [[B]] ![[B]]", "B.md": ""})
        updates = await fake.run(changed={"A.md"}, mode="FULL")
        by_kind = {r.relation_type: r for r in updates["A.md"]}
        assert set(by_kind) == {RecordRelations.LINKED_TO, RecordRelations.RELATED}
        assert by_kind[RecordRelations.LINKED_TO].link_subpaths == ["Intro", "^abc"]
        assert by_kind[RecordRelations.RELATED].link_subpaths is None

    async def test_alias_links_resolve(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[Bee]]", "B.md": "---\naliases: [Bee]\n---\n"})
        updates = await fake.run(changed={"A.md", "B.md"}, mode="FULL")
        assert _edges(updates["A.md"]) == {("B", RecordRelations.LINKED_TO)}

    async def test_unresolved_link_points_at_a_placeholder(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"Notes/A.md": "[[Later]] ![[missing.png]]"})
        updates = await fake.run(changed={"Notes/A.md"}, mode="FULL")
        (edge,) = updates["Notes/A.md"]
        assert edge.relation_type == RecordRelations.LINKED_TO
        assert edge.external_record_id == _ext(vault, "Later.md")
        assert edge.record_name == "Later"

    async def test_placeholder_follows_new_note_location(self, vault: ObsidianConnector) -> None:
        vault._vault_settings = {"newFileLocation": "current"}
        fake = Vault(vault, {"Notes/A.md": "[[Later]]"})
        updates = await fake.run(changed={"Notes/A.md"}, mode="FULL")
        assert updates["Notes/A.md"][0].external_record_id == _ext(vault, "Notes/Later.md")

    async def test_note_created_elsewhere_relinks_its_sources(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[Later]]"})
        await fake.run(changed={"A.md"}, mode="FULL")
        # Later is created in a folder, not where the placeholder was.
        fake.files["Deep/Later.md"] = "hi"
        updates = await fake.run(changed={"Deep/Later.md"})
        assert set(updates) == {"A.md", "Deep/Later.md"}
        assert updates["A.md"][0].external_record_id == _ext(vault, "Deep/Later.md")

    async def test_deleted_target_relinks_its_sources(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[B]]", "B.md": ""})
        await fake.run(changed={"A.md", "B.md"}, mode="FULL")
        del fake.files["B.md"]
        updates = await fake.run(removed={"B.md"})
        # A's edge now points at the placeholder where B would be recreated.
        (edge,) = updates["A.md"]
        assert edge.external_record_id == _ext(vault, "B.md")
        state = _decode_graph_state(vault.graph_store[vault._graph_state_key()])
        assert state["unresolved"] == {"A.md": ["B"]}

    async def test_moved_target_relinks_its_sources(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[B]]", "B.md": "", "C.md": "[[Other]]"})
        await fake.run(changed={"A.md", "B.md", "C.md"}, mode="FULL")
        fake.files["Moved/B.md"] = fake.files.pop("B.md")
        updates = await fake.run(moved={"B.md": "Moved/B.md"})
        assert "A.md" in updates and "C.md" not in updates
        assert updates["A.md"][0].external_record_id == _ext(vault, "Moved/B.md")

    async def test_unchanged_run_does_nothing(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[B]]"})
        assert await fake.run() == {}
        vault.data_entities_processor.get_records_by_status.assert_not_awaited()

    async def test_unreadable_note_keeps_its_edges(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[B]]", "B.md": ""})
        vault._fetch_desktop_content = AsyncMock(side_effect=RuntimeError("gone"))
        assert await fake.run(changed={"A.md"}) == {}

    async def test_desktop_offline_ends_the_pass_quietly(self, vault: ObsidianConnector) -> None:
        fake = Vault(vault, {"A.md": "[[B]]", "B.md": ""})
        vault._fetch_desktop_content = AsyncMock(side_effect=LocalFsDesktopOfflineError("offline"))
        assert await fake.run(changed={"A.md", "B.md"}) == {}
        assert vault._fetch_desktop_content.await_count == 1

    async def test_listing_failure_never_fails_the_sync(self, vault: ObsidianConnector) -> None:
        vault._on_file_upserted("A.md", "")
        vault.data_entities_processor.get_records_by_status = AsyncMock(side_effect=RuntimeError("db"))
        await vault._on_run_complete("INCREMENTAL")
        vault.data_entities_processor.on_related_records_update.assert_not_awaited()


class TestConnectorHooks:
    def test_identity(self, vault: ObsidianConnector) -> None:
        assert vault.connector_name == Connectors.OBSIDIAN
        assert vault._record_group_external_id() == f"obsidian:{CONNECTOR_ID}"
        assert vault._record_name_for("Folder/My Note.md") == "My Note"
        assert vault._record_name_for("img.png") == "img.png"

    def test_skips_internal_and_excluded(self, vault: ObsidianConnector) -> None:
        vault._exclude = ExcludeFilters(["Templates"])
        assert vault._skip_rel_path(".obsidian/app.json")
        assert vault._skip_rel_path(".trash/a.md")
        assert vault._skip_rel_path("Templates/Daily.md")
        assert not vault._skip_rel_path("Notes/a.md")

    def test_move_tracks_old_path_as_removed(self, vault: ObsidianConnector) -> None:
        vault._on_file_upserted("New/A.md", "A.md")
        assert vault._changed_notes == {"New/A.md"}
        assert vault._removed_paths == {"A.md"}

    async def test_vault_settings_load_and_tolerate_failure(self, vault: ObsidianConnector) -> None:
        vault._fetch_desktop_content = AsyncMock(
            return_value=b'{"userIgnoreFilters": ["Templates/"], "newFileLocation": "current"}'
        )
        await vault._load_vault_settings()
        assert vault._exclude.excludes("Templates/x.md")
        assert vault._vault_settings["newFileLocation"] == "current"
        probe = vault._fetch_desktop_content.await_args.args[0]
        assert probe.local_fs_relative_path == ".obsidian/app.json"

        vault._fetch_desktop_content = AsyncMock(side_effect=RuntimeError("missing"))
        await vault._load_vault_settings()
        assert not vault._exclude and vault._vault_settings == {}

    async def test_indexed_content_is_rendered_for_notes_only(self, vault: ObsidianConnector) -> None:
        note = _file(vault, "r1", "A.md")
        out = await vault._transform_content(note, b"---\ntags: [x]\n---\nbody")
        assert out.decode().startswith("# A\nTags: #x")
        png = _file(vault, "r2", "a.png")
        assert await vault._transform_content(png, b"\x89PNG") == b"\x89PNG"

    async def test_batch_skips_vault_internal_files(
        self, vault: ObsidianConnector, tmp_path: Path
    ) -> None:
        vault.config_service.get_config = AsyncMock(
            return_value={"sync": {SYNC_ROOT_PATH_KEY: str(tmp_path)}}
        )
        empty = AsyncMock(return_value=(FilterCollection(filters=[]), FilterCollection(filters=[])))
        events = [
            LocalFsFileEvent(type="CREATED", path=p, timestamp=1, isDirectory=False, sha256="abc", size=5)
            for p in (".obsidian/app.json", ".git/HEAD", "Note.md", "img.png")
        ]
        with patch("app.connectors.sources.local_fs.connector.load_connector_filters", new=empty):
            await vault._reload_sync_settings()
            root = Path(vault.sync_root_path)
            owner, rg = await vault._ensure_owner_and_record_group(root)
            stats = await vault._apply_file_event_batch(
                events,
                owner=owner,
                sync_filters=FilterCollection(filters=[]),
                indexing_filters=FilterCollection(filters=[]),
                external_record_group_id=rg,
                root_for_display=root,
                emitted_folder_paths=set(),
            )
        assert stats.skipped == 2
        written = {
            r.record_name
            for call in vault.data_entities_processor.on_new_records.await_args_list
            for r, _ in call.args[0]
        }
        assert written == {"Note", "img.png"}
        assert vault._changed_notes == {"Note.md"}


class TestTagsBecomeTopics:
    @staticmethod
    def _record(connector: Connectors, text: str, topics: list[str] | None) -> Any:
        from app.models.blocks import Block, BlockType, SemanticMetadata

        record = MagicMock()
        record.connector_name = connector
        record.block_containers.blocks = [Block(type=BlockType.TEXT, data=text)]
        record.semantic_metadata = SemanticMetadata(topics=topics)
        return record

    def test_note_tags_are_added_to_extracted_topics(self) -> None:
        from app.modules.transformers.document_extraction import _merge_source_tags

        text = render_for_index(parse_note("a.md", "---\ntags: [Project/X]\n---\n#ml body"))
        record = self._record(Connectors.OBSIDIAN, text, ["Machine Learning", "ml"])
        _merge_source_tags(record)
        assert record.semantic_metadata.topics == ["Machine Learning", "ml", "Project/X"]

    def test_other_connectors_are_untouched(self) -> None:
        from app.modules.transformers.document_extraction import _merge_source_tags

        record = self._record(Connectors.LOCAL_FS, "# a\nTags: #x\n\nbody", ["t"])
        _merge_source_tags(record)
        assert record.semantic_metadata.topics == ["t"]
