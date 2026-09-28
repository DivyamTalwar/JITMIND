"""Runs when J02's optional storage package is installed/in the integration tree."""

import json

import pytest

from jitmind.schemas import MemoryEntry, Page

storage = pytest.importorskip("jitmind.storage")
from jitmind.scope import ScopeAuthority, ScopeDenied
from jitmind.scoped_research import (
    DurableScopedBackend,
    ScopedBackendError,
    ScopedResearchFacade,
)


def add(store, namespace, key, content):
    request = storage.IngestRequest.create(namespace, key, content)
    proposal = storage.Proposal(
        abstract=content,
        header="same header",
        decorated=content,
        decision={"operation": "add"},
    )
    return store.commit_proposal(request, store.snapshot(namespace).revision, proposal)


def test_real_sqlite_isolation_expansion_revocation_and_snapshot(tmp_path):
    store = storage.SQLiteDurableStore(tmp_path / "memory.db")
    add(store, "red", "same", "shared authorized")
    add(store, "red-sibling", "same", "shared FOREIGN_SENTINEL")
    authority = ScopeAuthority()
    authority.grant("p", "red", [])
    scope = authority.context("p", "red", "r")
    facade = ScopedResearchFacade(authority, DurableScopedBackend(store))
    response = facade.research(scope, "shared")
    result = response.serialize()
    assert len(result["raw_memory"]["hits"]) == 2
    assert "FOREIGN_SENTINEL" not in str(result)
    add(store, "red", "next", "shared new")
    with pytest.raises(ScopeDenied):
        response.serialize()
    assert len(facade.research(scope, "shared").serialize()["raw_memory"]["hits"]) == 4
    authority.revoke("p", "red")
    with pytest.raises(ScopeDenied):
        facade.research(scope, "shared")


def test_truncated_snapshot_fails_closed(tmp_path):
    store = storage.SQLiteDurableStore(tmp_path / "memory.db")
    add(store, "red", "one", "shared one")
    add(store, "red", "two", "shared two")
    authority = ScopeAuthority()
    authority.grant("p", "red", [])
    scope = authority.context("p", "red", "r")
    with pytest.raises(ScopedBackendError):
        ScopedResearchFacade(authority, DurableScopedBackend(store, limit=1)).research(
            scope, "shared"
        )


def test_durable_update_does_not_expand_superseded_source(tmp_path):
    store = storage.SQLiteDurableStore(tmp_path / "memory.db")
    old = add(store, "red", "one", "shared old secret")
    request = storage.IngestRequest.create("red", "two", "shared replacement")
    proposal = storage.Proposal(
        abstract="shared replacement",
        header="same header",
        decorated="shared replacement",
        decision={"operation": "update", "target_id": old.memory_id},
    )
    store.commit_proposal(request, old.revision, proposal)
    authority = ScopeAuthority()
    authority.grant("p", "red", [])
    facade = ScopedResearchFacade(authority, DurableScopedBackend(store))
    result = facade.research(authority.context("p", "red", "r"), "shared").serialize()
    assert "shared old secret" not in str(result)
    assert "shared replacement" in str(result)


def test_durable_revocation_during_snapshot_read_drops_response(tmp_path):
    store = storage.SQLiteDurableStore(tmp_path / "memory.db")
    add(store, "red", "one", "shared one")
    authority = ScopeAuthority()
    authority.grant("p", "red", [])
    scope = authority.context("p", "red", "r")
    original = store.snapshot

    def revoking(*args, **kwargs):
        result = original(*args, **kwargs)
        authority.revoke("p", "red")
        return result

    store.snapshot = revoking
    with pytest.raises(ScopeDenied):
        ScopedResearchFacade(authority, DurableScopedBackend(store)).research(
            scope, "shared"
        )


def migrated_pair(tmp_path, fact_meta, page_meta):
    source = tmp_path / "legacy"
    source.mkdir()
    fact = MemoryEntry(
        id="m", content="shared abstract", source_page_id="0", meta=fact_meta
    )
    page = Page(
        header="source",
        content="shared PAGE_SENTINEL",
        meta={"page_id": 0, "memory_id": "m", **page_meta},
    )
    (source / "advanced_memory_state.json").write_text(
        json.dumps({"entries": [fact.model_dump()]})
    )
    (source / "pages.json").write_text(json.dumps([page.model_dump()]))
    store = storage.SQLiteDurableStore(tmp_path / "memory.db")
    store.import_staged(storage.stage_legacy(source, quiesced=True), "n")
    authority = ScopeAuthority()
    authority.grant("p", "n", ["repo"])
    return store, authority, authority.context("p", "n", "r")


@pytest.mark.parametrize(
    "page_meta",
    [
        {"repo_id": "other", "snapshot_id": "main"},
        {"repo_id": "repo", "snapshot_id": "other"},
    ],
)
def test_durable_conflicting_provenance_denied_before_index(tmp_path, page_meta):
    store, authority, scope = migrated_pair(
        tmp_path, {"repo_id": "repo", "snapshot_id": "main"}, page_meta
    )
    from jitmind.scoped_research import LexicalScopedIndex

    class NeverBuild(LexicalScopedIndex):
        def build_scoped(self, *args):
            pytest.fail("conflicting provenance reached index")

    facade = ScopedResearchFacade(
        authority, DurableScopedBackend(store), index_factory=NeverBuild()
    )
    with pytest.raises(ScopeDenied):
        facade.research(scope, "shared", snapshots=(("repo", "main"),))


@pytest.mark.parametrize(
    "fact_meta,page_meta",
    [
        ({"repo_id": "repo", "snapshot_id": "main"}, {}),
        ({}, {"repo_id": "repo", "snapshot_id": "main"}),
        ({"repo_id": "repo"}, {"snapshot_id": "main"}),
    ],
)
def test_durable_missing_provenance_inherits_without_widening(
    tmp_path, fact_meta, page_meta
):
    store, authority, scope = migrated_pair(tmp_path, fact_meta, page_meta)
    facade = ScopedResearchFacade(authority, DurableScopedBackend(store))
    assert facade.research(scope, "shared").serialize()["raw_memory"]["hits"] == []
    hits = facade.research(scope, "shared", snapshots=(("repo", "main"),)).serialize()[
        "raw_memory"
    ]["hits"]
    assert len(hits) == 2
    assert all(
        h["meta"]["repo_id"] == "repo" and h["meta"]["snapshot_id"] == "main"
        for h in hits
    )
    authority.grant("p", "n", [])
    assert (
        facade.research(authority.context("p", "n", "ungranted"), "shared").serialize()[
            "raw_memory"
        ]["hits"]
        == []
    )


@pytest.mark.parametrize(
    "page_meta,visible",
    [
        ({"expires_at": "2000-01-01T00:00:00Z"}, False),
        ({"t_valid": "2999-01-01T00:00:00Z"}, False),
        ({"t_invalid": "2000-01-01T00:00:00Z"}, False),
        (
            {
                "t_valid": "2000-01-01T00:00:00Z",
                "t_invalid": "2999-01-01T00:00:00Z",
                "expires_at": "2999-01-01T00:00:00Z",
            },
            True,
        ),
    ],
)
def test_durable_page_eligibility_is_independent_of_active_fact(
    tmp_path, page_meta, visible
):
    store, authority, scope = migrated_pair(tmp_path, {}, page_meta)
    hits = (
        ScopedResearchFacade(authority, DurableScopedBackend(store))
        .research(scope, "shared")
        .serialize()["raw_memory"]["hits"]
    )
    assert [h["snippet"] for h in hits if h["source"] == "memory"] == [
        "shared abstract"
    ]
    assert [h["snippet"] for h in hits if h["source"] == "page"] == (
        ["shared PAGE_SENTINEL"] if visible else []
    )
    assert (
        store.get_page("n", store.snapshot("n").entries[0].source_page_id).content
        == "shared PAGE_SENTINEL"
    )


def test_real_durable_noop_delete_and_midread_mutation_controls(tmp_path):
    store = storage.SQLiteDurableStore(tmp_path / "memory.db")
    original = add(store, "n", "active", "shared active")
    request = storage.IngestRequest.create("n", "noop", "shared NOOP_SENTINEL")
    store.commit_proposal(
        request,
        store.snapshot("n").revision,
        storage.Proposal(
            abstract="shared NOOP_SENTINEL",
            header="h",
            decorated="d",
            decision={"operation": "noop"},
        ),
    )
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    scope = authority.context("p", "n", "r")
    facade = ScopedResearchFacade(authority, DurableScopedBackend(store))
    response = facade.research(scope, "shared")
    result = response.serialize()
    assert len(result["raw_memory"]["hits"]) == 2 and "NOOP_SENTINEL" not in str(result)
    get_page = store.get_page
    mutated = False

    def mutate_after_read(*args):
        nonlocal mutated
        page = get_page(*args)
        if not mutated:
            mutated = True
            request = storage.IngestRequest.create("n", "delete", "shared delete")
            store.commit_proposal(
                request,
                store.snapshot("n").revision,
                storage.Proposal(
                    abstract="shared delete",
                    header="h",
                    decorated="d",
                    decision={"operation": "delete", "target_id": original.memory_id},
                ),
            )
        return page

    store.get_page = mutate_after_read
    with pytest.raises(ScopeDenied):
        facade.research(scope, "shared")
    with pytest.raises(ScopeDenied):
        response.serialize()
    assert not facade.research(scope, "shared").serialize()["raw_memory"]["hits"]
