from datetime import datetime, timezone

import pytest

from jitmind.agents.research_agent import ResearchAgent
from jitmind.scoping import namespace_matches, normalize_namespace
from jitmind.schemas import AdvancedMemoryStore, Hit, InMemoryPageStore, MemoryEntry, Page


def test_namespace_components_are_validated_and_not_string_prefixed() -> None:
    assert normalize_namespace(user_id="alice") == ("users", "alice")
    assert namespace_matches(("org", "red"), ("org", "red"))
    assert not namespace_matches(("org", "red-team"), ("org", "red"))
    assert namespace_matches(
        ("org", "red", "agent", "planner"),
        ("org", "red"),
        include_descendants=True,
    )
    with pytest.raises(ValueError):
        normalize_namespace(("org", ""))


def test_store_reads_and_temporal_queries_stay_inside_namespace() -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    store.add_entry(MemoryEntry(id="red", content="red secret", namespace=("org", "red")))
    store.add_entry(MemoryEntry(id="blue", content="blue secret", namespace=("org", "blue")))

    assert store.load(namespace=("org", "red")).abstracts == ["red secret"]
    assert [e.id for e in store.get_entries(namespace=("org", "blue"))] == ["blue"]
    now = datetime.now(timezone.utc)
    assert [e.id for e in store.query_as_of(now, namespace=("org", "red"))] == ["red"]


def test_cross_namespace_update_fails_closed() -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    store.add_entry(MemoryEntry(id="red", content="secret", namespace=("org", "red")))
    replacement = MemoryEntry(
        id="replacement", content="tampered", namespace=("org", "blue")
    )

    with pytest.raises(LookupError):
        store.update_entry("red", replacement, namespace=("org", "blue"))

    assert store.get_entry_by_id("red").status == "active"
    assert store.get_entry_by_id("replacement") is None


def test_retrieval_filter_rejects_hits_from_other_namespaces() -> None:
    pages = InMemoryPageStore()
    pages.add(Page(header="red", content="red", meta={"namespace": ["org", "red"]}))
    pages.add(Page(header="blue", content="blue", meta={"namespace": ["org", "blue"]}))
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.page_store = pages
    agent._current_namespace = ("org", "red")

    hits = [
        Hit(page_id="0", snippet="red", source="keyword"),
        Hit(page_id="1", snippet="blue", source="keyword"),
    ]
    scoped = agent._filter_scope_hits(hits)

    assert [hit.page_id for hit in scoped] == ["0"]
    assert scoped[0].meta["namespace"] == ["org", "red"]
