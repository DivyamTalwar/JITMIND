from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from jitmind.schemas import (
    AdvancedMemoryStore,
    InMemoryMemoryStore,
    InMemoryPageStore,
    MemoryEntry,
    Page,
)
from jitmind.scope import ScopeAuthority, ScopeDenied
from jitmind.scoped_research import (
    LegacyScopedBackend,
    LegacyStoreBinding,
    LexicalScopedIndex,
    ScopedBackendError,
    ScopedDocument,
    ScopedResearchFacade,
    ScopedView,
)


class ObservedFactory(LexicalScopedIndex):
    def __init__(self):
        self.inputs = []

    def build_scoped(self, view, authority, scope):
        self.inputs.append(tuple(doc.content for doc in view.documents))
        return super().build_scoped(view, authority, scope)


class Generator:
    def __init__(self, callback=None):
        self.prompts = []
        self.callback = callback

    def generate_single(self, *, prompt, schema):
        self.prompts.append(prompt)
        if self.callback:
            self.callback()
        return {"json": {"content": prompt, "sources": ["invented"]}}


def setup(tmp_path=None):
    authority = ScopeAuthority()
    bindings = []
    stores = []
    for namespace in ("red", "red-sibling"):
        authority.grant("alice", namespace, ["root-one", "root-two"])
        pages = InMemoryPageStore(str(tmp_path / namespace) if tmp_path else None)
        pages.add(
            Page(
                header="same name",
                content=f"shared {namespace} fact",
                meta={"namespace_id": namespace},
            )
        )
        memory = InMemoryMemoryStore()
        memory.add(f"shared {namespace} memory")
        bindings.append(LegacyStoreBinding(namespace, pages, memory))
        stores.append(pages)
    return authority, LegacyScopedBackend(tuple(bindings)), stores


def test_real_stores_are_isolated_before_build_rerank_prompt_and_logs(tmp_path, capsys):
    authority, backend, stores = setup(tmp_path)
    stores[0].add(
        Page(
            header="same name",
            content="FOREIGN_SENTINEL shared",
            meta={"namespace_id": "red-sibling"},
        )
    )
    factory, generator = ObservedFactory(), Generator()

    class Reranker:
        def rerank(self, query, docs, top_k):
            assert all(
                "FOREIGN_SENTINEL" not in doc and "red-sibling" not in doc
                for doc in docs
            )
            return [(index, 1.0) for index in range(len(docs))]

    facade = ScopedResearchFacade(
        authority,
        backend,
        index_factory=factory,
        generator=generator,
        reranker=Reranker(),
    )
    scope = authority.context("alice", "red", "r")
    result = facade.research(scope, "shared").serialize()
    assert len(result["raw_memory"]["hits"]) == 2
    assert factory.inputs == [("shared red fact", "shared red memory")]
    assert "red-sibling" not in str(generator.prompts)
    assert "FOREIGN_SENTINEL" not in str(result) + str(capsys.readouterr())
    assert "invented" not in str(result)


def test_unsupported_backend_fails_before_any_access():
    class Unscoped:
        def load(self):
            pytest.fail("unscoped load")

        def search(self, *args):
            pytest.fail("unscoped search")

    with pytest.raises(ScopedBackendError):
        ScopedResearchFacade(ScopeAuthority(), Unscoped())
    with pytest.raises(ScopedBackendError):
        LegacyScopedBackend((LegacyStoreBinding("red", Unscoped()),))
    authority, backend, _ = setup()
    with pytest.raises(ScopedBackendError):
        ScopedResearchFacade(authority, backend, index_factory=Unscoped())


def test_untagged_stores_cannot_be_bound_to_two_namespaces(tmp_path):
    for first, second in [
        (InMemoryPageStore(), None),
        (InMemoryPageStore(str(tmp_path)), InMemoryPageStore(str(tmp_path))),
    ]:
        with pytest.raises(ScopedBackendError):
            LegacyScopedBackend(
                (
                    LegacyStoreBinding("a", first),
                    LegacyStoreBinding("b", second or first),
                )
            )


def test_revoked_before_access_midflight_and_before_serialization():
    authority, backend, _ = setup()
    scope = authority.context("alice", "red", "r")
    factory = ObservedFactory()
    facade = ScopedResearchFacade(authority, backend, index_factory=factory)
    response = facade.research(scope, "shared")
    authority.revoke("alice", "red")
    with pytest.raises(ScopeDenied):
        response.serialize()
    with pytest.raises(ScopeDenied):
        facade.research(scope, "shared")
    assert len(factory.inputs) == 1
    authority.grant("alice", "red", ["root-one"])
    scope = authority.context("alice", "red", "r2")
    facade.generator = Generator(lambda: authority.revoke("alice", "red"))
    with pytest.raises(ScopeDenied):
        facade.research(scope, "shared")
    assert len(factory.inputs) == 2
    assert len(facade._cache) == 0


def test_cache_rechecks_snapshot_and_regrant_and_canonical_metadata():
    authority, backend, stores = setup()
    scope = authority.context("alice", "red", "r")
    factory = ObservedFactory()
    facade = ScopedResearchFacade(authority, backend, index_factory=factory)
    response = facade.research(scope, "shared")
    facade.research(scope, "shared")
    assert len(factory.inputs) == 1
    stores[0].save([Page(header="changed", content="shared replacement")])
    with pytest.raises(ScopeDenied):
        response.serialize()
    result = facade.research(scope, "shared").serialize()
    assert "shared replacement" in str(result)
    assert len(factory.inputs) == 2
    authority.grant("alice", "red", ["root-one"])
    scope = authority.context("alice", "red", "r2")
    facade.research(scope, "shared")
    assert len(factory.inputs) == 3


def test_concurrent_requests_do_not_share_current_scope():
    authority, backend, _ = setup()
    barrier = Barrier(2)
    generator = Generator(barrier.wait)
    facade = ScopedResearchFacade(authority, backend, generator=generator)

    def run(namespace):
        scope = authority.context("alice", namespace, namespace)
        return facade.research(scope, "shared").serialize()

    with ThreadPoolExecutor(max_workers=2) as pool:
        red, sibling = list(pool.map(run, ("red", "red-sibling")))
    assert "red-sibling" not in str(red)
    assert "shared red fact" not in str(sibling)


def test_repository_ids_same_basename_and_branch_snapshots_stay_distinct():
    authority, backend, stores = setup()
    stores[0].save(
        [
            Page(
                header="project",
                content="shared one main",
                meta={
                    "repo_id": "root-one",
                    "snapshot_id": "main@1",
                    "source_digest": "digest",
                    "validation": {"checked": True},
                },
            ),
            Page(
                header="project",
                content="shared one other",
                meta={"repo_id": "root-one", "snapshot_id": "other@1"},
            ),
            Page(
                header="project",
                content="shared two",
                meta={"repo_id": "root-two", "snapshot_id": "main@1"},
            ),
        ]
    )
    scope = authority.context("alice", "red", "r")
    facade = ScopedResearchFacade(authority, backend)
    result = facade.research(
        scope, "shared", snapshots=(("root-one", "main@1"),)
    ).serialize()
    assert "shared one main" in str(result)
    assert "shared one other" not in str(result) and "shared two" not in str(result)
    hit = next(hit for hit in result["raw_memory"]["hits"] if hit["source"] == "page")
    assert hit["meta"]["source_digest"] == "digest"
    assert hit["meta"]["validation"] == {"checked": True}
    default = facade.research(scope, "shared").serialize()
    assert all(hit["source"] == "memory" for hit in default["raw_memory"]["hits"])


def test_optional_adapter_failure_leaves_authorized_primary_only(capsys):
    authority, backend, _ = setup()

    class Broken:
        scope_protocol = "jitmind.scoped.v1"

        def open_view(self, authority, scope, snapshots):
            authority.require(scope)
            raise ScopedBackendError()

        def validate_view(self, *args):
            pytest.fail("failed backend has no view")

    facade = ScopedResearchFacade(authority, backend, optional_backends=(Broken(),))
    result = facade.research(
        authority.context("alice", "red", "r"), "shared"
    ).serialize()
    assert "red-sibling" not in str(result)
    assert len(result["raw_memory"]["hits"]) == 2
    assert "FOREIGN" not in str(capsys.readouterr())


def test_foreign_derived_hit_is_rejected_before_prompt(capsys):
    authority, backend, _ = setup()

    class Poisoned(LexicalScopedIndex):
        def build_scoped(self, view, authority, scope):
            class Index:
                def search(self, *args):
                    return [("FOREIGN_SENTINEL", 1.0)]

            return Index()

    generator = Generator()
    facade = ScopedResearchFacade(
        authority, backend, index_factory=Poisoned(), generator=generator
    )
    with pytest.raises(ScopeDenied, match="^Scope denied$"):
        facade.research(authority.context("alice", "red", "r"), "shared")
    assert not generator.prompts
    assert "FOREIGN_SENTINEL" not in str(capsys.readouterr())


def test_foreign_view_rejected_before_index_and_error_sanitized():
    authority, backend, _ = setup()

    class Poisoned:
        scope_protocol = "jitmind.scoped.v1"

        def open_view(self, authority, scope, snapshots):
            return ScopedView(
                "red",
                "revision",
                (ScopedDocument("x", "red-sibling", "FOREIGN_SENTINEL"),),
            )

        def validate_view(self, *args):
            pass

    factory = ObservedFactory()
    facade = ScopedResearchFacade(authority, Poisoned(), index_factory=factory)
    with pytest.raises(ScopeDenied):
        facade.research(authority.context("alice", "red", "r"), "shared")
    assert not factory.inputs

    class ErrorFactory(LexicalScopedIndex):
        def build_scoped(self, *args):
            raise RuntimeError("FOREIGN_SENTINEL")

    facade = ScopedResearchFacade(authority, backend, index_factory=ErrorFactory())
    with pytest.raises(ScopedBackendError, match="^Scoped backend unavailable$"):
        facade.research(authority.context("alice", "red", "r2"), "shared")


def test_advanced_memory_selection_and_ttl_history():
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    memory = AdvancedMemoryStore(enable_auto_cleanup=False, ttl_seconds=0)
    memory.add_entry(MemoryEntry(content="shared expired", meta={"namespace_id": "n"}))
    pages = InMemoryPageStore()
    backend = LegacyScopedBackend((LegacyStoreBinding("n", pages, memory),))
    scope = authority.context("p", "n", "r")
    assert not backend.open_view(authority, scope).documents
    assert len(memory.get_entries(include_inactive=True)) == 1


def test_revocation_during_index_build_never_searches():
    authority, backend, _ = setup()

    class Revoking(LexicalScopedIndex):
        def build_scoped(self, view, authority, scope):
            authority.revoke(scope.principal_id, scope.namespace_id)

            class Index:
                def search(self, *args):
                    pytest.fail("search after revocation")

            return Index()

    facade = ScopedResearchFacade(authority, backend, index_factory=Revoking())
    with pytest.raises(ScopeDenied):
        facade.research(authority.context("alice", "red", "r"), "shared")


@pytest.mark.parametrize(
    "state", ["expired", "superseded", "deleted", "ttl", "foreign", "other-branch"]
)
def test_ineligible_fact_cannot_reenter_through_its_source_page(state):
    authority = ScopeAuthority()
    authority.grant("p", "n", ["repo"])
    memory = AdvancedMemoryStore(
        enable_auto_cleanup=False, ttl_seconds=0 if state == "ttl" else None
    )
    entry = MemoryEntry(
        id="m",
        content="shared abstract",
        source_page_id="0",
        status=state if state in ("expired", "superseded", "deleted") else "active",
        meta={"namespace_id": "foreign" if state == "foreign" else "n"},
    )
    if state == "other-branch":
        entry.meta.update(repo_id="repo", snapshot_id="other")
    memory.add_entry(entry)
    pages = InMemoryPageStore()
    pages.add(
        Page(header="source", content="shared SOURCE_SENTINEL", meta={"memory_id": "m"})
    )
    backend = LegacyScopedBackend((LegacyStoreBinding("n", pages, memory),))
    scope = authority.context("p", "n", "r")
    result = (
        ScopedResearchFacade(authority, backend)
        .research(scope, "shared", snapshots=(("repo", "main"),))
        .serialize()
    )
    assert result["raw_memory"]["hits"] == []


def test_real_store_roots_with_same_basename_are_distinct_and_alias_is_rejected(
    tmp_path,
):
    one = tmp_path / "one" / "project"
    two = tmp_path / "two" / "project"
    pages_one = InMemoryPageStore(str(one))
    pages_two = InMemoryPageStore(str(two))
    pages_one.add(Page(header="project", content="shared root one"))
    pages_two.add(Page(header="project", content="shared root two"))
    backend = LegacyScopedBackend(
        (
            LegacyStoreBinding("../opaque", pages_one),
            LegacyStoreBinding("other", pages_two),
        )
    )
    authority = ScopeAuthority()
    authority.grant("p", "../opaque", [])
    scope = authority.context("p", "../opaque", "r")
    result = (
        ScopedResearchFacade(authority, backend).research(scope, "shared").serialize()
    )
    assert "shared root one" in str(result) and "shared root two" not in str(result)
    alias = tmp_path / "alias"
    alias.symlink_to(one, target_is_directory=True)
    with pytest.raises(ScopedBackendError):
        LegacyScopedBackend(
            (
                LegacyStoreBinding("one", pages_one),
                LegacyStoreBinding("two", InMemoryPageStore(str(alias))),
            )
        )


def test_serialization_errors_never_echo_backend_content():
    authority, backend, _ = setup()
    scope = authority.context("alice", "red", "r")
    response = ScopedResearchFacade(authority, backend).research(scope, "shared")

    def broken(*args):
        raise RuntimeError("FOREIGN_SENTINEL")

    backend.validate_view = broken
    with pytest.raises(ScopedBackendError, match="^Scoped backend unavailable$"):
        response.serialize()


def test_real_writer_page_restrictions_protect_abstract_before_index():
    from jitmind.agents.memory_agent import MemoryAgent

    class LocalGenerator:
        def generate_single(self, *, prompt, schema=None):
            return (
                {"json": {"operation": "add"}}
                if schema
                else {"text": "shared RESTRICTED_ABSTRACT"}
            )

    class DisabledGraph:
        def __bool__(self):
            return False

    pages = InMemoryPageStore()
    memory = AdvancedMemoryStore(enable_auto_cleanup=False)
    MemoryAgent(
        memory_store=memory,
        page_store=pages,
        generator=LocalGenerator(),
        graph_store=DisabledGraph(),
    ).memorize(
        "shared RESTRICTED_SOURCE", meta={"repo_id": "repo", "snapshot_id": "main"}
    )
    assert memory.get_entries()[0].meta == {}
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    backend = LegacyScopedBackend((LegacyStoreBinding("n", pages, memory),))
    factory, generator = ObservedFactory(), Generator()
    facade = ScopedResearchFacade(
        authority, backend, index_factory=factory, generator=generator
    )
    scope = authority.context("p", "n", "r")
    assert facade.research(scope, "shared").serialize()["raw_memory"]["hits"] == []
    assert factory.inputs == [()]
    assert not generator.prompts
    authority.grant("p", "n", ["repo"])
    scope = authority.context("p", "n", "r2")
    assert not facade.research(scope, "shared").serialize()["raw_memory"]["hits"]
    hits = facade.research(scope, "shared", snapshots=(("repo", "main"),)).serialize()[
        "raw_memory"
    ]["hits"]
    assert {h["snippet"] for h in hits} == {
        "shared RESTRICTED_ABSTRACT",
        "shared RESTRICTED_SOURCE",
    }
    assert all(
        h["meta"]["repo_id"] == "repo" and h["meta"]["snapshot_id"] == "main"
        for h in hits
    )


@pytest.mark.parametrize("link", ["source", "reverse", "both"])
@pytest.mark.parametrize(
    "restriction",
    [
        {"repo_id": "repo", "snapshot_id": "main"},
        {"namespace_id": "foreign"},
    ],
)
def test_legacy_link_directions_restrict_both_documents(link, restriction):
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    memory = AdvancedMemoryStore(enable_auto_cleanup=False)
    memory.add_entry(
        MemoryEntry(
            id="m",
            content="shared abstract",
            source_page_id="0" if link != "reverse" else None,
        )
    )
    pages = InMemoryPageStore()
    pages.add(
        Page(
            header="h",
            content="shared page",
            meta={**restriction, **({"memory_id": "m"} if link != "source" else {})},
        )
    )
    facade = ScopedResearchFacade(
        authority, LegacyScopedBackend((LegacyStoreBinding("n", pages, memory),))
    )
    assert not facade.research(authority.context("p", "n", "r"), "shared").serialize()[
        "raw_memory"
    ]["hits"]


class LocalBackend:
    scope_protocol = "jitmind.scoped.v1"

    def __init__(self, documents):
        self.documents = documents
        self.down = False

    def open_view(self, authority, scope, snapshots):
        authority.require(scope)
        if self.down:
            raise ScopedBackendError()
        return ScopedView(
            scope.namespace_id,
            "revision-1",
            tuple(
                ScopedDocument(k, scope.namespace_id, text)
                for k, text in self.documents
            ),
            snapshots,
        )

    def validate_view(self, authority, scope, view):
        view.require(authority, scope)
        if view != self.open_view(authority, scope, view.repo_snapshots):
            raise ScopeDenied()


def test_optional_outage_retains_backend_cache_and_response_identity():
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    scope = authority.context("p", "n", "r")
    first = LocalBackend([("x", "needle first")])
    second = LocalBackend([("x", "irrelevant"), ("y", "needle second")])
    facade = ScopedResearchFacade(
        authority, LocalBackend([]), optional_backends=(first, second)
    )
    warm = facade.research(scope, "needle").serialize()["raw_memory"]["hits"]
    expected = next(hit for hit in warm if hit["snippet"] == "needle second")
    first.down = True
    assert facade.research(scope, "needle").serialize()["raw_memory"]["hits"] == [
        expected
    ]
    first.down = False
    assert facade.research(scope, "needle").serialize()["raw_memory"]["hits"] == warm


@pytest.mark.parametrize(
    "documents",
    [
        [("x", "needle " * 100_000)],
        [("x", "needle " + "😀" * 5000)],
        [(str(i), "needle " + "a" * 16000) for i in range(5)],
    ],
)
def test_default_evidence_budgets_precede_both_model_hooks(documents):
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    generator = Generator()

    class Reranker:
        def rerank(self, *args, **kwargs):
            pytest.fail("oversized evidence reached reranker")

    facade = ScopedResearchFacade(
        authority, LocalBackend(documents), generator=generator, reranker=Reranker()
    )
    for _ in range(2):  # cache reuse must apply the same budget
        with pytest.raises(ScopedBackendError, match="^Scoped backend unavailable$"):
            facade.research(authority.context("p", "n", "r"), "needle")
    assert not generator.prompts


def test_evidence_exact_utf8_limits_include_labels_and_newlines():
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    scope = authority.context("p", "n", "r")
    texts = ["needle 😀", "needle é"]
    evidence = "[0:x] " + texts[0] + "\n[0:y] " + texts[1]
    generator = Generator()
    observed = []

    class Reranker:
        def rerank(self, query, docs, top_k):
            observed.append(docs)
            return [(0, 2), (1, 1)]

    for budget, succeeds in [
        (len(evidence.encode("utf-8")), True),
        (len(evidence.encode("utf-8")) - 1, False),
    ]:
        facade = ScopedResearchFacade(
            authority,
            LocalBackend(list(zip(("x", "y"), texts))),
            generator=generator,
            reranker=Reranker(),
            max_document_bytes=11,
            max_evidence_bytes=budget,
        )
        if succeeds:
            result = facade.research(scope, "needle").serialize()
            assert [h["snippet"] for h in result["raw_memory"]["hits"]] == texts
            assert generator.prompts == ["Question: needle\nEvidence:\n" + evidence]
        else:
            with pytest.raises(ScopedBackendError):
                facade.research(scope, "needle")
    assert observed == [texts]
    assert len(generator.prompts) == 1


@pytest.mark.parametrize("limit", [True, 0, -1, 16_777_217, 1.5, "100"])
@pytest.mark.parametrize("name", ["max_document_bytes", "max_evidence_bytes"])
def test_invalid_evidence_configuration(limit, name):
    with pytest.raises(ValueError, match="Invalid evidence budget"):
        ScopedResearchFacade(ScopeAuthority(), LocalBackend([]), **{name: limit})


@pytest.mark.parametrize("content", ["needle " * 100_000, "needle \ud800"])
def test_generator_only_cannot_receive_oversized_or_malformed_evidence(content):
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    generator = Generator()
    facade = ScopedResearchFacade(
        authority, LocalBackend([("x", content)]), generator=generator
    )
    with pytest.raises(ScopedBackendError):
        facade.research(authority.context("p", "n", "r"), "needle", top_k=1)
    assert not generator.prompts


def test_bytes_content_is_not_implicitly_decoded():
    with pytest.raises(ScopeDenied):
        ScopedDocument("x", "n", b"needle secret")


def test_document_budget_exact_boundary_and_one_byte_overflow():
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    scope = authority.context("p", "n", "r")
    generator = Generator()
    for limit in (11, 10):
        facade = ScopedResearchFacade(
            authority,
            LocalBackend([("x", "needle 😀")]),
            generator=generator,
            max_document_bytes=limit,
        )
        if limit == 11:
            assert (
                facade.research(scope, "needle").serialize()["raw_memory"]["hits"][0][
                    "snippet"
                ]
                == "needle 😀"
            )
        else:
            with pytest.raises(ScopedBackendError):
                facade.research(scope, "needle")
    assert len(generator.prompts) == 1


def test_legacy_shared_page_reconciles_all_owners_and_preserves_untagged_data():
    authority = ScopeAuthority()
    authority.grant("p", "n", [])
    pages = InMemoryPageStore()
    pages.add(Page(header="h", content="shared source"))
    memory = AdvancedMemoryStore(enable_auto_cleanup=False)
    memory.add_entry(MemoryEntry(id="one", content="shared one", source_page_id="0"))
    memory.add_entry(MemoryEntry(id="two", content="shared two", source_page_id="0"))
    facade = ScopedResearchFacade(
        authority, LegacyScopedBackend((LegacyStoreBinding("n", pages, memory),))
    )
    scope = authority.context("p", "n", "r")
    assert len(facade.research(scope, "shared").serialize()["raw_memory"]["hits"]) == 3
    # A reverse link to a second page adds a restriction to the whole component.
    pages.add(
        Page(
            header="h",
            content="shared restricted",
            meta={"memory_id": "two", "repo_id": "repo", "snapshot_id": "main"},
        )
    )
    assert not facade.research(scope, "shared").serialize()["raw_memory"]["hits"]
    authority.grant("p", "n", ["repo"])
    scope = authority.context("p", "n", "r2")
    assert (
        len(
            facade.research(scope, "shared", snapshots=(("repo", "main"),)).serialize()[
                "raw_memory"
            ]["hits"]
        )
        == 4
    )
    pages.add(
        Page(
            header="h",
            content="shared conflict",
            meta={"memory_id": "one", "repo_id": "repo", "snapshot_id": "other"},
        )
    )
    with pytest.raises(ScopeDenied):
        facade.research(scope, "shared", snapshots=(("repo", "main"),))
