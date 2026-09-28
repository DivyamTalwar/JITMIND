"""Independent-review regressions against the real SQLite authority."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from unittest.mock import patch

import pytest
from test_durable_migration import legacy_source
from test_durable_storage import FakeGenerator, proposal, write

from jitmind.agents.memory_agent import MemoryAgent
from jitmind.ingestion.documents import Document
from jitmind.ingestion.pipeline import IngestionPipeline
from jitmind.storage import (
    DurableMemoryAdapter,
    DurablePageAdapter,
    IngestRequest,
    InvalidRequest,
    MigrationError,
    SQLiteDurableStore,
    StagedLegacy,
    StorageFailure,
    stage_legacy,
)
from jitmind.storage.migration import MAX_SOURCE_BYTES

MAX_INT = 2**63 - 1
MAY = "2026-05-01T00:00:00+00:00"
JUNE = "2026-06-01T00:00:00+00:00"
JULY = "2026-07-01T00:00:00+00:00"


def put(store, key, operation="add", target=None, **times):
    value = proposal(operation, target, content="SENSITIVE")
    value.decision = value.decision.model_copy(update=times)
    return store.ingest(
        IngestRequest.create("a", key, "SENSITIVE", {"private": "SENSITIVE"}),
        lambda _: value,
    )


@pytest.mark.parametrize("retirement", ["delete", "update"])
def test_all_page_surfaces_require_active_fact_and_receipts_survive(
    tmp_path, retirement
):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JUNE)
    added = put(store, "add")
    targeted = put(store, "targeted", "noop", added.memory_id)
    untargeted = put(store, "untargeted", "noop")
    other = write(store, "other", namespace="b")
    adapter = DurablePageAdapter(store, "a")
    assert adapter.get(added.page_id).content == "SENSITIVE"
    for administrative in (targeted, untargeted):
        assert store.get_page("a", administrative.page_id) is None
        result = store.receipt_content(administrative)
        assert result.status == "unavailable" and result.page is None
        assert result.receipt == administrative
    assert [p.meta["page_id"] for p in adapter.load()] == [added.page_id]
    retired = put(store, "retire", retirement, added.memory_id)
    reopened = SQLiteDurableStore(store.path)
    for receipt in (added, targeted, untargeted):
        assert reopened.get_page("a", receipt.page_id) is None
        result = reopened.receipt_content(receipt)
        assert result.page is None
        assert result.status == ("retired" if receipt == added else "unavailable")
        update = reopened.memory_update(receipt)
        assert update.new_page.content == update.new_page.header == ""
        assert update.new_page.meta == {
            "page_id": receipt.page_id,
            "redacted": True,
            "content_status": result.status,
        }
        assert update.debug["decorated_page"] == ""
        assert update.debug["content_status"] == result.status
        assert update.debug["durable_receipt"] == receipt.model_dump()
        request = IngestRequest.create(
            "a", receipt.idempotency_key, "SENSITIVE", {"private": "SENSITIVE"}
        )
        assert (
            reopened.ingest(request, lambda _: pytest.fail("retry generated"))
            == receipt
        )
    expected = [] if retirement == "delete" else [retired.page_id]
    assert [p.meta["page_id"] for p in adapter.load()] == expected
    assert [p.meta["page_id"] for p in adapter.list_pages(limit=1)] == expected
    if retirement == "delete":
        assert reopened.receipt_content(retired).status == "unavailable"
        assert reopened.memory_update(retired).new_state.abstracts == []
    assert reopened.get_page("b", other.page_id) is not None
    assert reopened.status("a")["pages"] == reopened.status("a")["operations"] == 4
    with sqlite3.connect(store.path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM pages WHERE payload LIKE '%SENSITIVE%'"
            ).fetchone()[0]
            == 4
        )


@pytest.mark.parametrize("operation", ["noop", "delete"])
def test_default_durable_memorize_and_pipeline_acknowledge_hidden_content(
    tmp_path, operation
):
    store = SQLiteDurableStore(tmp_path / "db")
    generator = FakeGenerator()
    agent = MemoryAgent(generator=generator, durable_store=store, namespace_id="a")
    added = agent.memorize_durable("SENSITIVE", idempotency_key="first")
    generator.operation, generator.target = operation, added.memory_id
    update = agent.memorize("forget SENSITIVE", idempotency_key="operation")
    assert update.new_page.content == update.new_page.header == ""
    assert update.debug["content_status"] == "unavailable"
    receipt = agent.memorize_durable("forget SENSITIVE", idempotency_key="operation")
    calls = generator.calls
    assert agent.memorize("forget SENSITIVE", idempotency_key="operation") == update
    assert generator.calls == calls
    assert update.debug["durable_receipt"] == receipt.model_dump()
    generator.operation, generator.target = "add", None
    pipeline_target = agent.memorize_durable(
        "pipeline target", idempotency_key="pipeline-target"
    )
    generator.operation, generator.target = operation, pipeline_target.memory_id
    updates = IngestionPipeline(deduplicate=False).ingest_documents(
        [Document(content="SENSITIVE", source="test")], agent
    )
    assert len(updates) == 1
    assert updates[0].debug["content_status"] == "unavailable"
    assert updates[0].new_page.content == ""


@pytest.mark.parametrize("new_start", ["2026-04-01T00:00:00+00:00", None])
def test_inverting_update_rejected_before_any_write(tmp_path, new_start):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JUNE)
    original = put(store, "add", t_valid=JULY)
    before = store.status("a")
    traced = []
    original_connection = store._connection
    from contextlib import contextmanager

    @contextmanager
    def traced_connection(**kwargs):
        with original_connection(**kwargs) as conn:
            conn.set_trace_callback(traced.append)
            yield conn

    store._connection = traced_connection
    with pytest.raises(InvalidRequest):
        put(store, "update", "update", original.memory_id, t_valid=new_start)
    assert not [sql for sql in traced if sql.startswith(("INSERT", "UPDATE", "DELETE"))]
    assert SQLiteDurableStore(store.path).status("a") == before
    assert store.get_entry("a", original.memory_id).t_invalid is None


@pytest.mark.parametrize("new_start", [MAY, "2026-06-01T01:00:00+01:00"])
def test_ordered_update_control(tmp_path, new_start):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JUNE)
    original = put(store, "add", t_valid=MAY)
    updated = put(store, "update", "update", original.memory_id, t_valid=new_start)
    old = store.get_entry("a", original.memory_id, include_inactive=True)
    assert old.t_invalid == new_start and old.t_expired == JUNE
    assert store.get_entry("a", updated.memory_id).version_of == old.id


@pytest.mark.parametrize("prior_end", [None, "2026-08-01T00:00:00+00:00"])
def test_future_delete_is_explicit_zero_length_retraction(tmp_path, prior_end):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JUNE)
    original = put(store, "add", t_valid=JULY, t_invalid=prior_end)
    deleted = put(store, "delete", "delete", original.memory_id)
    old = store.get_entry("a", original.memory_id, include_inactive=True)
    assert old.t_valid == old.t_invalid == JULY
    assert old.t_expired == JUNE and old.status == "deleted"
    assert store.memory_update(deleted).debug["content_status"] == "unavailable"
    assert store.get_entry("a", old.id) is None


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_corrupt_target_interval_is_rejected_without_repair(tmp_path, operation):
    store = SQLiteDurableStore(tmp_path / "db", clock=lambda: JUNE)
    original = put(store, "add", t_valid=JULY)
    with sqlite3.connect(store.path) as conn:
        data = json.loads(conn.execute("SELECT payload FROM facts").fetchone()[0])
        data["t_invalid"] = MAY
        conn.execute("UPDATE facts SET payload=?", (json.dumps(data),))
    before = store.path.read_bytes()
    value = proposal(operation, original.memory_id)
    value.decision.t_valid = JULY
    with pytest.raises(StorageFailure):
        store.commit_proposal(IngestRequest.create("a", "mutation", "m"), 1, value)
    assert store.path.read_bytes() == before
    assert store.status("a")["revision"] == 1


@pytest.mark.parametrize("bad", [2**63, -(2**63) - 1, True, 1.0])
@pytest.mark.parametrize(
    "surface",
    [
        "events",
        "expected_revision",
        "projection_cursor",
        "memory_cursor",
        "page_cursor",
        "inactive_flag",
    ],
)
def test_public_integer_and_cursor_inputs_reject_before_binding(tmp_path, bad, surface):
    store = SQLiteDurableStore(tmp_path / "db")
    operations = {
        "events": lambda: store.pending_events("a", after_revision=bad),
        "expected_revision": lambda: store.commit_proposal(
            IngestRequest.create("a", "k", "m"), bad, proposal()
        ),
        "projection_cursor": lambda: store.projected_entries("a", after_id=bad),
        "memory_cursor": lambda: DurableMemoryAdapter(store, "a").get_entries(
            after_id=bad
        ),
        "page_cursor": lambda: DurablePageAdapter(store, "a").list_pages(after_id=bad),
        "inactive_flag": lambda: DurableMemoryAdapter(store, "a").get_entries(
            include_inactive=bad
        ),
    }
    if surface == "inactive_flag" and bad is True:
        assert operations[surface]() == []
    else:
        with pytest.raises(InvalidRequest, match="^invalid_request$"):
            operations[surface]()


@pytest.mark.parametrize(
    "surface",
    [
        "snapshot",
        "events",
        "drain",
        "projection",
        "memory",
        "pages",
        "memory_update",
        "context",
        "replans",
        "timeout",
        "retries",
    ],
)
def test_all_public_counts_are_bounded_even_on_replay(tmp_path, surface):
    store = SQLiteDurableStore(tmp_path / "db")
    receipt = write(store)
    request = IngestRequest.create("a", "key", "message", {"custom": [1, {"x": True}]})
    methods = {
        "snapshot": lambda: store.snapshot("a", limit=2**63),
        "events": lambda: store.pending_events("a", limit=2**63),
        "drain": lambda: store.drain_outbox("a", limit=2**63),
        "projection": lambda: store.projected_entries("a", limit=2**63),
        "memory": lambda: DurableMemoryAdapter(store, "a").get_entries(limit=2**63),
        "pages": lambda: DurablePageAdapter(store, "a").list_pages(limit=2**63),
        "memory_update": lambda: store.memory_update(receipt, limit=2**63),
        "context": lambda: store.ingest(
            request, lambda _: proposal(), context_limit=2**63
        ),
        "replans": lambda: store.ingest(
            request, lambda _: proposal(), max_replans=2**63
        ),
        "timeout": lambda: SQLiteDurableStore(store.path, busy_timeout_ms=2**63),
        "retries": lambda: SQLiteDurableStore(store.path, commit_retries=2**63),
    }
    with pytest.raises(InvalidRequest):
        methods[surface]()


def test_signed_64_boundary_and_revision_exhaustion(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db")
    receipt = write(store)
    assert store.pending_events("a", after_revision=MAX_INT) == []
    assert len(store.pending_events("a", after_revision=0)) == 1
    assert store.projected_entries("a", after_id="") == []
    assert len(DurableMemoryAdapter(store, "a").get_entries(after_id="")) == 1
    assert len(DurablePageAdapter(store, "a").list_pages(after_id="")) == 1
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE namespaces SET revision=?", (MAX_INT - 1,))
    last = store.commit_proposal(
        IngestRequest.create("a", "last", "m"), MAX_INT - 1, proposal()
    )
    assert last.revision == MAX_INT
    assert store.snapshot("a").revision == MAX_INT
    assert write(store) == receipt  # replay at exhaustion still reconciles
    before = store.path.read_bytes()
    with pytest.raises(InvalidRequest):
        store.commit_proposal(
            IngestRequest.create("a", "new", "m"), MAX_INT, proposal()
        )
    assert store.path.read_bytes() == before


def staged_document(source):
    value = json.dumps(source)
    return StagedLegacy(
        hashlib.sha256(value.encode()).hexdigest(),
        value,
        len(source["pages"]),
        len(source["memory"]["entries"]),
    )


@pytest.mark.parametrize(
    "damage",
    ["late_retirement", "early_child", "retired_before_created", "validity_overlap"],
)
def test_contradictory_staged_chain_rejected_without_writes(tmp_path, damage):
    legacy_source(tmp_path)
    source = {
        "memory": json.loads((tmp_path / "advanced_memory_state.json").read_text()),
        "pages": json.loads((tmp_path / "pages.json").read_text()),
    }
    old, new = source["memory"]["entries"]
    if damage == "late_retirement":
        old["t_expired"] = JULY
    elif damage == "early_child":
        new["t_created"] = "2025-01-01T00:00:00+00:00"
    elif damage == "retired_before_created":
        old["t_expired"] = "2025-01-01T00:00:00+00:00"
    else:
        old["t_invalid"], new["t_valid"] = JULY, JUNE
    store = SQLiteDurableStore(tmp_path / "db")
    before = store.path.read_bytes()
    with pytest.raises(MigrationError):
        store.import_staged(staged_document(source), "a")
    assert store.path.read_bytes() == before
    (tmp_path / "advanced_memory_state.json").write_text(json.dumps(source["memory"]))
    with pytest.raises(MigrationError):
        stage_legacy(tmp_path, quiesced=True)


@pytest.mark.parametrize(
    "variant", ["missing_optional_times", "ordered_gap", "zero_length", "offset_equal"]
)
def test_legacy_temporal_controls_preserve_explicit_values(tmp_path, variant):
    legacy_source(tmp_path)
    source = {
        "memory": json.loads((tmp_path / "advanced_memory_state.json").read_text()),
        "pages": json.loads((tmp_path / "pages.json").read_text()),
    }
    old, new = source["memory"]["entries"]
    if variant == "missing_optional_times":
        old["t_expired"] = old["t_invalid"] = new["t_valid"] = None
    elif variant == "ordered_gap":
        new["t_created"] = new["t_valid"] = JULY
    elif variant == "zero_length":
        old["status"] = "deleted"
        old["t_valid"] = old["t_invalid"] = JULY
        new["version_of"] = None
    else:
        new["t_created"] = "2026-02-01T01:00:00+01:00"
    store = SQLiteDurableStore(tmp_path / "db")
    store.import_staged(staged_document(source), "a")
    for raw in (old, new):
        actual = store.get_entry("a", raw["id"], include_inactive=True).model_dump()
        assert actual == raw | {
            "source_page_id": store.resolve_page_alias("a", raw["source_page_id"])
        }


@pytest.mark.parametrize("kind", ["characters", "utf8_bytes"])
def test_manual_staging_envelope_is_checked_before_json_parse(tmp_path, kind):
    raw = (
        " " * (MAX_SOURCE_BYTES + 1)
        if kind == "characters"
        else "é" * (MAX_SOURCE_BYTES // 2 + 1)
    )
    staged = StagedLegacy(hashlib.sha256(raw.encode()).hexdigest(), raw, 0, 0)
    store = SQLiteDurableStore(tmp_path / "db")
    before = store.path.read_bytes()
    with (
        patch(
            "jitmind.storage.migration.json.loads",
            side_effect=AssertionError("parsed oversized input"),
        ),
        pytest.raises(MigrationError),
    ):
        store.import_staged(staged, "a")
    assert store.path.read_bytes() == before


def test_envelope_limit_valid_control_and_strict_manual_counts(tmp_path):
    source = {"memory": {"entries": []}, "pages": []}
    staged = staged_document(source)
    raw = staged.source_json + " " * (
        MAX_SOURCE_BYTES - len(staged.source_json.encode())
    )
    staged = replace(
        staged, source_json=raw, source_digest=hashlib.sha256(raw.encode()).hexdigest()
    )
    store = SQLiteDurableStore(tmp_path / "db")
    assert store.import_staged(staged, "a")["fact_count"] == 0
    for bad in (2**63, -(2**63) - 1, False, 0.0):
        for field in ("page_count", "fact_count"):
            with pytest.raises(MigrationError):
                store.import_staged(replace(staged, **{field: bad}), "b")
    assert store.status("b")["revision"] == 0


def test_receipt_content_available_and_forged_receipts_fail_closed(tmp_path):
    store = SQLiteDurableStore(tmp_path / "db")
    receipt = write(store)
    result = store.receipt_content(receipt)
    assert result.status == "available" and result.receipt == receipt
    assert result.page == store.get_page("a", receipt.page_id)
    assert store.memory_update(receipt).debug["content_status"] == "available"
    for fields in (
        {"revision": 2**63},
        {"revision": True},
        {"page_id": "other"},
        {"namespace_id": "b"},
    ):
        forged = receipt.model_copy(update=fields)
        for method in (store.receipt_content, store.memory_update):
            with pytest.raises(InvalidRequest):
                method(forged)


def test_imported_unlinked_pages_have_no_ordinary_visibility(tmp_path):
    page = {
        "header": "SENSITIVE",
        "content": "SENSITIVE",
        "meta": {"page_id": 0, "custom": "SENSITIVE"},
    }
    staged = staged_document({"memory": {"entries": []}, "pages": [page]})
    store = SQLiteDurableStore(tmp_path / "db")
    store.import_staged(staged, "a")
    page_id = store.resolve_page_alias("a", "0")
    assert page_id is not None
    assert store.get_page("a", page_id) is None
    adapter = DurablePageAdapter(store, "a")
    assert adapter.load() == adapter.list_pages() == []
    assert store.status("a")["pages"] == 1


@pytest.mark.parametrize("source", [None, b"{}", "\ud800"])
def test_malformed_manual_staging_envelope_is_domain_error(tmp_path, source):
    store = SQLiteDurableStore(tmp_path / "db")
    with pytest.raises(MigrationError, match="^invalid_legacy_source$"):
        store.import_staged(StagedLegacy("digest", source, 0, 0), "a")
    assert store.status("a")["revision"] == 0


def test_real_sqlite_full_rolls_back_and_foreign_database_is_unchanged(tmp_path):
    # Preserve the independent review's positive controls as acceptance tests.
    from contextlib import contextmanager

    from jitmind.storage import SchemaMismatch

    store = SQLiteDurableStore(tmp_path / "db")
    original = store._connection

    @contextmanager
    def capped(**kwargs):
        with original(**kwargs) as conn:
            count = conn.execute("PRAGMA page_count").fetchone()[0]
            conn.execute(f"PRAGMA max_page_count={count}")
            yield conn

    store._connection = capped
    with pytest.raises(StorageFailure):
        store.ingest(
            IngestRequest.create("a", "full", "x" * 100000), lambda _: proposal()
        )
    fresh = SQLiteDurableStore(store.path)
    assert fresh.status("a") == {
        "revision": 0,
        "pending_events": 0,
        "delivery_watermark": 0,
        "pages": 0,
        "facts": 0,
        "operations": 0,
    }
    foreign = tmp_path / "foreign"
    with sqlite3.connect(foreign) as conn:
        conn.execute("CREATE TABLE unrelated(secret TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('preserve')")
    before = foreign.read_bytes()
    with pytest.raises(SchemaMismatch):
        SQLiteDurableStore(foreign)
    assert foreign.read_bytes() == before
