# Persistent code-memory bindings

`jitmind.code_memory.CodeMemoryService` attaches an existing authoritative fact
version to local source evidence. It never creates, edits, expires or deletes the
fact. A relocated declaration is source identity evidence, not semantic validation
of the claim. Fact `t_valid` / observation history and a binding's `validated_at`
answer different questions.

This is opt-in. Existing JSON memory and old-format data are unchanged. Without
this service or its optional source runtime, legacy memory still works; callers
must not label a legacy or unknown memory as a currently verified code fact.
There is no external product dependency, model call, network call, or new Python
package requirement in this layer.

## Public API

```python
CodeMemoryService(authority, facts, code_context, binding_path)
```

Supply the J03 `ScopeAuthority`, J02 `SQLiteDurableStore` (or a trusted host adapter
implementing its namespace-keyed `get_entry` contract), and J04 `CodeContext` using
the **same authority**. Construction does no binding database IO. The parent of
`binding_path` must already exist. Scope grant/registration are host operations,
not user-facing RPCs. A fact adapter must enforce exact namespace identity and
active/not-forgotten state; never substitute a cache for authoritative `get_entry`.

| Method | Arguments after `scope` | Result |
| --- | --- | --- |
| `create_binding` | keyword-only `logical_binding_id`, `fact_version_id`, `repo_id`, `snapshot_id`, `path`, `qualified_name` | Revision 1 of `CodeBinding`; requires an active fact and one adequately covered declaration |
| `revalidate` | `logical_binding_id`, keyword-only `snapshot_id`, `expected_revision` | Appended immutable revision, or typed conflict |
| `get_current` | `logical_binding_id` | Latest authorized binding metadata, downgraded to `unknown` if fact authority or source is no longer current; `None` for absent or unauthorized logical ID |
| `history` | `logical_binding_id`, optional keyword-only `snapshot_id`, `after_revision=0`, `limit=100` | Immutable stored metadata in revision order; explicit snapshot filter supports historical inspection |
| `project_fact` | keyword-only `fact_version_id`, `event_revision` | Whether high-water/tombstone metadata changed; safe for J02 outbox redelivery |

`logical_binding_id` is a stable caller-chosen identity within a namespace.
`binding_id` identifies one immutable revision; `supersedes_binding_id` connects
history. `fact_version_id` is the J02 `MemoryEntry.id`, not the source page ID.
J02 updates create another fact version; they do not silently retarget a binding.

Use a fresh host-issued `ScopeContext.request_id` for a new operation. Repeating
an identical create/revalidate request ID and arguments replays its stored version
without appending history. Conflicting request reuse raises `BindingConflict`.
Replay still consults current authority/freshness; an older replay cannot regain
`verified_current`. Concurrent identical first attempts may receive a conflict;
retry with the same request to retrieve the recorded result.

`history` returns recorded *at-check* states, not current verification. Metadata
retention is permitted by the current namespace/repository grant, including after
a fact is deleted or superseded. Neither current retrieval nor history returns
fact content or raw code. `get_current` reports `fact_unavailable` after forgetting
even when projection has not delivered the deletion event. Fact content retrieval
remains solely J02's responsibility.

Repository authorization constrains every payload SELECT, including request
replay and historical revisions. A valid scope for another repository receives
the same result for an unauthorized logical ID as for a nonexistent one:
`get_current` returns `None`, `history` returns `()`, and `revalidate` raises generic
`BindingUnavailable`. Invalid, foreign-issued or revoked contexts still raise
J03's generic `ScopeDenied` before access; authorization is checked again before
return. No ScopeAuthority API change is needed. Snapshot-filtered history selects
only matching rows in SQL **before** the revision ordering and limit. To paginate,
pass the last returned row's `revision` as `after_revision`, keeping the snapshot
filter unchanged; an empty page then means no later authorized matching revision.
The current head is not read as a prerequisite to historical selection.

## Runnable example

With the separately provisioned J04 Node adapter, run the following Python code.
Set `JITMIND_TEST_GRAFT_ADAPTER` to its absolute directory if it is not at
`adapters/graft`. This example uses a temporary unversioned source root; the tests
also exercise real temporary Git repositories.

```python
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from jitmind.scope import ScopeAuthority
from jitmind.storage import IngestRequest, Proposal, SQLiteDurableStore
from jitmind.code_context import CodeContext, NodeParser, RepoRegistry
from jitmind.code_memory import CodeMemoryService

with TemporaryDirectory() as directory:
    base = Path(directory).resolve()
    root = base / "source"
    root.mkdir()
    (root / "api.py").write_text("def original(x):\n    return x + 123\n")
    repo = str(uuid4())
    authority = ScopeAuthority()
    authority.grant("alice", "team", (repo,))
    scope = authority.context("alice", "team", "create-binding")
    registry = RepoRegistry()
    registry.register(repo, str(root), "team")
    node = str(Path(shutil.which("node")).resolve())
    adapter = str(Path(os.environ.get("JITMIND_TEST_GRAFT_ADAPTER", "adapters/graft")).resolve())
    context = CodeContext(authority, registry, NodeParser(node, adapter))
    facts = SQLiteDurableStore(base / "facts.db")
    receipt = facts.commit_proposal(
        IngestRequest.create("team", "fact-request", "Recorded source observation"),
        0,
        Proposal(abstract="Recorded source observation", header="Observation",
                 decorated="Observation", decision={"operation": "add"}),
    )
    service = CodeMemoryService(authority, facts, context, base / "bindings.db")
    snapshot = context.build(scope, repo, deadline_ms=10000)
    first = service.create_binding(
        scope, logical_binding_id="api-observation", fact_version_id=receipt.memory_id,
        repo_id=repo, snapshot_id=snapshot.snapshot_id, path="api.py",
        qualified_name="original",
    )
    (root / "api.py").write_text("def renamed(x):\n    return x + 123\n")
    scope = authority.context("alice", "team", "rename-check")
    snapshot = context.build(scope, repo, deadline_ms=10000)
    current = service.revalidate(scope, "api-observation",
                                 snapshot_id=snapshot.snapshot_id, expected_revision=1)
    assert current.reason_code == "same_file_rename"
    assert current.fact_version_id == first.fact_version_id
    assert len(service.history(scope, "api-observation")) == 2
    historical = service.history(scope, "api-observation", snapshot_id=first.snapshot_id)
    assert historical == (first,)
```

For J02 outbox delivery, iterate `pending_events(namespace)` and each event's
`memory_ids`, calling `project_fact` with its revision before acknowledging the
event in the host's delivery coordinator. This service does not acknowledge J02
events itself. Status comes from `get_entry(include_inactive=True)` at delivery,
never from event payloads. Older/equal upserts are discarded. A terminal fact
version cannot be revived even by a higher revision. An authoritative deletion
observed during stale delivery still establishes a terminal marker while retaining
the maximum observed revision. No fact or code payload is cached in this database.

## Revalidation policy

The versioned policy is `python-ast-name-only/v1`. SHA-256 records the original
source span and the normalized Python AST. Normalization erases only the outer
declaration's name and excludes AST position attributes. It preserves literals,
parameter names/defaults, annotations, function versus async function, nested
syntax and executable bodies. Comments and formatting are not semantic evidence.
The signature digest uses the same AST with its body removed. This is deterministic
for the supported Python AST runtime; a future normalization change must receive a
new policy version. There is no similarity matcher or accuracy claim.

1. Authorize before source/database access; load only the scoped logical binding;
   check authoritative fact state before source acquisition and again before publish.
2. Use J04's public `repo_map`, `file_api`, `find_all` and `check_freshness` queries.
   All code comes from captured snapshots. Never read current disk at old graph
   line numbers. Freshness checks do not rebuild the index; the host explicitly
   calls `CodeContext.build`.
3. Missing checkout, unsupported syntax, unreadability, partial coverage and
   result/context budgets yield `unknown` when the observation meets the source
   watermark below. Older generations and unavailable snapshots that cannot meet
   that watermark raise `BindingStaleSource` without publishing. No
   relocation or fact expiration follows from an unknown source.
4. A unique same-path/qualified identity with compatible kind/signature and equal
   normalized body is source verified. A changed body or signature becomes
   `changed`, with `review_required=True`; its original comparison baseline stays
   intact. Later repeated checks cannot clear that review requirement.
5. For relocation, examine **all** compatible declarations in the covered subset.
   Exactly one equal normalized fingerprint can produce `same_file_rename`,
   `file_move`, or `rename_and_move`. Multiple identical getters are ambiguous;
   enumeration order cannot break the tie. Nonexact compatible candidates or
   same-name candidates with changed signatures are `unknown` for review.
6. Confirm absence only after a complete current search finds no exact/plausible
   candidates. The binding becomes `orphaned_confirmed`; the fact remains active.

Evidence lists contain at most eight source references with total candidate count.
References contain path, qualified name, source ID and line range, never raw code.
Original provenance remains in revision 1 and the immutable history chain.

## Persistence and race boundaries

The separate local SQLite database uses schema version 2, DELETE journaling,
FULL synchronization, foreign keys and a 250 ms busy timeout. It rejects unknown
schemas and non-DELETE journal modes. IDs, revisions, paths, payload sizes,
history pages and evidence lists are bounded; booleans are rejected as revisions.
There is no legacy-memory migration or modification of the primary fact database.
Version 1 binding databases migrate locally in a transaction: repository and
snapshot index columns are extracted in SQL, without application payload hydration.
Stored binding JSON, receipts, fact tombstones and history remain unchanged. The
source watermark is recovered from the maximum recorded historical generation
to accommodate older databases that already contain a backward observation.
Migration does not rewrite or promote a legacy current head. Unknown versions
remain rejected.

Expensive source/fact acquisition happens outside the write transaction. Publishing
compares the original logical binding revision **and** a persistent per-repository
source-observation revision, then appends history and the idempotency receipt in
one short transaction. This source revision is independent of J04's process-local
generation counter. A concurrent successful observation of any binding in that
repository causes older prepared work to conflict. J04 generation/freshness is
checked again immediately before publication and current retrieval checks again
before returning. The adapter holds J04’s existing `_lock` across that final
check and the short CAS write, preventing concurrent in-process index publication.
This is the only private J04 integration point; source retrieval itself uses public
queries. Follow-up: J04 should export a public generation-lease context manager.

`source_generation` is a publication high-water mark per namespace/repository.
An append may keep the recorded generation only for the same snapshot, or advance
it. It may never lower it, including when revalidation would return `unknown`.
Backward publication raises exported `BindingStaleSource` (a `BindingConflict`
subclass) in the same SQLite transaction **before any writes**. This leaves the
current binding, immutable history, idempotency receipts and persistent
source-observation revision untouched. A generation sequence `1 -> 2 -> old 1`
therefore remains at 2; use `history(snapshot_id=old_snapshot)` to inspect the
original at-check observation. A freshly rebuilt copy of old content at generation
3 is a new observation and can be published subject to the usual fact, lineage
and freshness checks. Terminal fact tombstones remain permanent throughout.

J04's counter itself is process-local; J05 does not turn it into a distributed
clock or reset the durable watermark after restart. If a fresh J04 context is
behind that watermark, publication fails closed until its builds reach a higher
generation (or the same generation and snapshot). A scope with no retained source
snapshot also cannot lower it using the adapter's unknown-source fallback. Hosts
must coordinate context lifetime/builds; portable generation epochs remain a
dependency follow-up. The persistent observation revision counts successful
publications, independently of the generation, and never advances on rejection.

`BindingConflict`, `BindingStaleSource`, `BindingUnavailable`, and
`BindingStorageError` use generic messages; denied scope uses J03 `ScopeDenied`.

SQLite cannot atomically lock the checkout and a separate authoritative fact
store. The current return path rechecks authority/freshness to withhold a stale
verified result. Historical revisions record observations, not a distributed
transaction guarantee. Clients must not treat a saved response as indefinitely
current, and should re-read before code-dependent actions. External edits after
the final check are inherently outside the observation boundary.

## Current conservative limits and follow-ups

- The J04 public API is bounded at 50 files in `repo_map`, 50 declarations per
  `file_api`, and 16,000 source bytes per page. Any truncation yields unknown. A
  future paginated structured snapshot export could extend coverage efficiently;
  the adapter currently performs multiple bounded captures through J04 queries.
- J04 currently reports any unsupported file in repository coverage as incomplete.
  Mixed-language roots may therefore be unknown even for a Python binding. This
  layer does not invent a broader source-coverage claim.
- J04 declaration spans currently omit Python decorators. Until complete decorated
  spans are exposed, any `@` occurrence makes the source subset insufficient. This
  also conservatively rejects matrix operators and `@` inside strings/comments.
- J04 content snapshot IDs omit branch identity. The adapter additionally reads
  only the registered root's fixed `.git/HEAD` metadata with bounded no-follow file
  descriptors, before/after acquisition. It never reads Git config, credentials,
  refs or remote identity, and runs no Git command. A changed HEAD lineage cannot
  inherit verification even with identical source bytes. Git pointer-file layouts
  (linked worktrees/submodules), symlink metadata and unreadable metadata yield
  unknown. Unversioned registered roots are supported. Follow-up: move branch/
  checkout lineage into J04's structured snapshot API, including worktree support.
- Snapshots retained by J04 are process-local. After restart/eviction, binding
  history survives but current verification requires an explicit source build and
  revalidation. There is no external generator or automatic semantic review.
- Historical retention here is metadata only. Removing retained metadata or
  supplying historically valid fact content requires a separately authorized
  retention workflow in the authoritative store.

## Design provenance

Native Python design reference: Koragraph commit
`c9ce746fbd6db67e794f5f8714c8d9a8f4669e27`, specifically
`src/practice/revalidate.js`, `fingerprint.js`, `resolve.js`, `promote.js`, and
`repo-identity.js`. The reference informed the distinction between unknown source,
changed code and absence, and the separation of fact identity from volatile source
coordinates. This implementation deliberately requires all-match uniqueness,
retains immutable bindings, never expires a fact, and uses the target's UUID
registry and scope authority instead of remote/basename guesses. No donor source
code or donor database schema was copied. This note does not relicense donor code
or attest to inspection of an agreement.
