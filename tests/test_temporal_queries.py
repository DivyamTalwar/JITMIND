from datetime import datetime, timezone

from jitmind.schemas import AdvancedMemoryStore, MemoryEntry


UTC = timezone.utc


def _ts(month: int, day: int) -> str:
    return datetime(2026, month, day, tzinfo=UTC).isoformat()


def test_valid_time_and_transaction_time_queries() -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    old = MemoryEntry(
        id="old",
        content="office is in Delhi",
        t_created=_ts(1, 2),
        t_observed=_ts(1, 2),
        t_valid=_ts(1, 1),
    )
    store.add_entry(old)
    replacement = MemoryEntry(
        id="new",
        content="office is in Bengaluru",
        version_of="old",
        t_created=_ts(2, 2),
        t_observed=_ts(2, 2),
        t_valid=_ts(2, 1),
    )
    store.update_entry("old", replacement)

    assert [e.id for e in store.query_as_of(_ts(1, 15))] == ["old"]
    assert [e.id for e in store.query_as_of(_ts(2, 15))] == ["new"]
    assert [
        e.id
        for e in store.query_as_of(_ts(1, 15), transaction_at=_ts(1, 15))
    ] == ["old"]
    assert store.query_as_of(_ts(2, 15), transaction_at=_ts(1, 15)) == []


def test_history_and_diff_follow_version_chain() -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    first = MemoryEntry(id="v1", content="plan A", t_valid=_ts(1, 1))
    second = MemoryEntry(
        id="v2", content="plan B", version_of="v1", t_valid=_ts(2, 1)
    )
    store.add_entry(first)
    store.update_entry("v1", second)

    assert [entry.id for entry in store.get_version_history("v2")] == ["v1", "v2"]
    diff = store.diff_as_of(_ts(1, 15), _ts(2, 15))
    assert [entry.id for entry in diff["changed"]] == ["v2"]
    assert diff["added"] == []
    assert diff["removed"] == []


def test_cleanup_retains_auditable_history_by_default() -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False, ttl_seconds=0)
    store.add_entry(MemoryEntry(id="expired", content="historical fact"))

    assert store.cleanup_expired() == 1
    assert store.get_entry_by_id("expired").status == "expired"


def test_invalid_timestamp_fails_loudly() -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    try:
        store.query_as_of("not-a-time")
    except ValueError as exc:
        assert "valid_at" in str(exc)
    else:
        raise AssertionError("invalid time should fail")
