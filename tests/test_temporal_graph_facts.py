from datetime import datetime, timezone

import pytest

from jitmind.graph.graph_store import GraphMemoryStore
from jitmind.retriever.graph_retriever import GraphRetriever
from jitmind.schemas import GraphFact


class _Result(list):
    pass


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.params = None

    def run(self, query, **params):
        self.query = query
        self.params = params
        return _Result(self.rows)

    def close(self):
        return None


class _Driver:
    def __init__(self, session):
        self._session = session

    def session(self, database):
        assert database == "neo4j"
        return self._session


def _store(rows):
    session = _Session(rows)
    store = GraphMemoryStore.__new__(GraphMemoryStore)
    store._database = "neo4j"
    store._driver = _Driver(session)
    return store, session


def test_query_facts_builds_a_bitemporal_snapshot() -> None:
    rows = [
        {
            "id": "fact-1",
            "head": "Alice",
            "relation": "WORKS_AT",
            "tail": "Acme",
            "source_memory_id": "memory-1",
            "source_page_id": "7",
            "t_observed": "2026-02-02T00:00:00+00:00",
            "t_valid": "2026-02-01T00:00:00+00:00",
            "t_invalid": None,
        }
    ]
    store, session = _store(rows)

    facts = store.query_facts(
        ["Alice"],
        relation_types=["works at"],
        valid_at="2026-02-03T00:00:00Z",
        observed_at=datetime(2026, 2, 4, tzinfo=timezone.utc),
        namespace=("tenant", "red"),
    )

    assert [fact.statement for fact in facts] == ["Alice WORKS_AT Acme"]
    assert session.params == {
        "names": ["Alice"],
        "types": ["WORKS_AT"],
        "valid_at": "2026-02-03T00:00:00+00:00",
        "observed_at": "2026-02-04T00:00:00+00:00",
        "namespace": ["tenant", "red"],
        "limit": 20,
    }
    assert "r.t_invalid" in session.query
    assert "m.namespace" in session.query


def test_each_relation_observation_has_a_distinct_stable_id() -> None:
    store, _ = _store([])
    first = store._relation_id("memory-1", "Person:Alice", "WORKS_AT", "Org:Acme")
    replay = store._relation_id("memory-1", "Person:Alice", "WORKS_AT", "Org:Acme")
    later = store._relation_id("memory-2", "Person:Alice", "WORKS_AT", "Org:Acme")

    assert first == replay
    assert first != later


def test_invalid_temporal_query_fails_before_hitting_neo4j() -> None:
    store, session = _store([])
    with pytest.raises(ValueError, match="valid_at"):
        store.query_facts(valid_at="not-a-time")
    assert session.query is None


def test_graph_retriever_preserves_fact_provenance() -> None:
    class FakeStore:
        def query_facts(self, names, **kwargs):
            assert names == ["Alice"]
            assert kwargs["valid_at"] == "2026-02-03T00:00:00Z"
            return [
                GraphFact(
                    id="fact-1",
                    head="Alice",
                    relation="WORKS_AT",
                    tail="Acme",
                    source_memory_id="memory-1",
                    source_page_id="7",
                    t_valid="2026-02-01T00:00:00Z",
                )
            ]

    retriever = GraphRetriever({"graph_store": FakeStore()})
    hits = retriever.search_temporal(
        ["Alice"], valid_at="2026-02-03T00:00:00Z"
    )[0]

    assert hits[0].page_id == "7"
    assert hits[0].snippet == "Alice WORKS_AT Acme"
    assert hits[0].meta["fact_id"] == "fact-1"
    assert hits[0].meta["memory_id"] == "memory-1"

