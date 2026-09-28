# Durable open work and bounded preflight

J-06 adds an opt-in library API. It needs no remote provider, vector index, graph,
model, hooks, CLI installation, or other product. Primary facts remain exclusively
owned by `SQLiteDurableStore` (J02); scope issuance remains owned by
`ScopeAuthority` (J03); code identity/revalidation remains owned by
`CodeMemoryService` (J05). Existing JSON fact/page formats are unchanged.

## Public composition API

Import from these modules directly (J05 owns `code_memory/__init__.py`):

- `code_memory.work_storage`: `WorkDatabase(path)`, `WorkError`, `Conflict`,
  `Deferred`, and the residual monotonic `Budget` passed to host adapters.
- `code_memory.open_work`: `OpenWorkService(database, authority,
  is_admin=None, verify_evidence=None)`, immutable `Obligation`, `WorkPage`.
  Methods: `create`, `transition`, `get`, `list`, `history`, `binding_locations`.
- `code_memory.lesson_models`: `LessonProjection(database, authority, facts,
  can_author=None, can_retire=None, binding_authority=None)`, `FactAuthority`,
  `DurableFactAuthority(store, authority)`, `DurableBindingAuthority(bindings, authority)`,
  `ProposedAction(repo_id, file_path, symbol='', action='edit')`.
  Methods: `import_observation`, `author_instruction`,
  `project_binding_observation`, `candidates`, `retire_instruction`, `forget`.
- `code_memory.preflight`: `PreflightService(projection, policy=None,
  approved_repo=None)`, `PreflightPolicy`, `PreflightResult`, `Session`.
  Methods: `start_session`, `deliver`, `end_session`, `cleanup`.

All scope arguments must be issued by the configured J03 authority. A manually
constructed context, even with identical claims/request ID, is rejected. Scope
checks run before disk/candidate work and before responses. Namespace and repo
IDs are exact opaque identifiers, never basenames or inferred remote identities.
Host initialization of `WorkDatabase` happens outside request handling.

`approved_repo(scope, repo_id, relative_path, budget)` optionally checks a
host-configured registry such as J04. It must return exactly `True` to approve and
must use bounded, nonwaiting operations. No path is opened by preflight. Absolute,
dot, dot-dot, backslash, NUL, drive/URI, empty component, and percent-escape paths
are rejected before this callback. Action is one of `edit`, `write`, `read`, `test`;
symbol is structured lexical metadata, never parsed from prose or evidence.
A lesson on `Outer` applies to `Outer.inner`, but not `Outerish`. A host-authored
lesson with file `*` and empty symbol applies throughout that exact authorized
repository; it cannot grant cross-repository scope. Ordinary file paths match
exactly. Instructions rank before observations, then by file/symbol specificity,
confidence and stable ID.

## Runnable local example

Run the following Python with the composed J02/J03/J06 modules on `PYTHONPATH`.
It uses temporary SQLite databases and deterministic proposals, with no provider.
Host authentication, grants, authoring, and verification adapters must be configured
by the application; do not expose these administrative objects as model tools.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from jitmind.scope import ScopeAuthority
from jitmind.storage import SQLiteDurableStore, IngestRequest, Proposal
from jitmind.code_memory.work_storage import WorkDatabase
from jitmind.code_memory.open_work import OpenWorkService
from jitmind.code_memory.lesson_models import (
    DurableFactAuthority, LessonProjection, ProposedAction,
)
from jitmind.code_memory.preflight import PreflightService

with TemporaryDirectory() as directory:
    root = Path(directory)
    authority = ScopeAuthority()
    authority.grant("alice", "team-exact-id", ["repo-exact-id"])
    scope = authority.context("alice", "team-exact-id", "request-1")
    facts = SQLiteDurableStore(root / "facts.sqlite")
    receipt = facts.ingest(
        IngestRequest.create("team-exact-id", "fact-1", "Null inputs need a guard."),
        lambda _: Proposal(
            abstract="Null inputs need a guard.", header="Null handling",
            decorated="Null inputs need a guard.", decision={"operation": "add"},
        ),
    )
    database = WorkDatabase(root / "work.sqlite")
    # A host verification registry, populated only after actual verification.
    verified_runs = {"run-42": {"passed": True}}
    def verify(scope, repo, refs, reason):
        return bool(reason) and all(
            ref in verified_runs and verified_runs[ref]["passed"] for ref in refs
        )
    work = OpenWorkService(database, authority, verify_evidence=verify)
    item = work.create(
        scope, "repo-exact-id", summary="Finish null-input regression review",
        expected_revision=0, idempotency_key="open-1",
        decision_refs=(receipt.memory_id,),
    )
    projection = LessonProjection(database, authority,
                                  DurableFactAuthority(facts, authority))
    target = ProposedAction("repo-exact-id", "src/service.py", "process")
    projection.import_observation(
        scope, target, lesson_id="null-lesson", fact_id=receipt.memory_id,
        version=1, source_revision=receipt.revision,
        body="Null inputs need a guard.", source_verified=True, confidence=.95,
    )
    preflight = PreflightService(projection)
    session = preflight.start_session(scope, target.repo_id)
    first = preflight.deliver(scope, session, target, delivery_key="edit-1")
    assert first.state == "delivered"
    assert preflight.deliver(scope, session, target).state == "nothing_relevant"
    assert preflight.deliver(scope, session, target,
                             delivery_key="edit-1").replayed
    done = work.transition(
        scope, target.repo_id, item.id, status="resolved", expected_revision=1,
        idempotency_key="resolve-1", reason="Verified against registered test run",
        evidence_refs=("run-42",),
    )
    assert done.status == "resolved"
    preflight.end_session(scope, session)
```

The example registry is illustrative configuration, not a claim that a test run
named `run-42` occurred. In a host application, populate verification references
from the actual authenticated verification system. Neither reasons nor evidence
strings are executed, interpreted as commands, or used to grant permissions.

## Work semantics and owner policy

An obligation has a persistent UUID, namespace, exact repo, creator, owner,
revision, status, independent authored summary, optional binding/decision/evidence
reference IDs, and UTC created/updated/optional due timestamps. Reference arrays
are limited to 32 IDs; descriptions/reasons to 2,000 characters. These references
are opaque IDs: no primary fact body is copied or hydrated. Do not use a work
summary/reason as an archival copy of a forgettable fact body.

Any repo-authorized principal can create self-owned work and read scoped work or
history. Assigning work to another owner and transitioning another owner's work
require the host-configured `is_admin(scope, repo)` predicate. Creator alone does
not confer owner privileges. The default denies admin actions. This is an explicit
application policy, not a role inferred from request metadata or text.

Create requires `expected_revision=0`; transitions require the exact current
revision. Every create/transition has a caller-supplied `idempotency_key`, scoped by
namespace/repo/authenticated actor. A SHA-256 digest covers the canonical bounded
operation payload. Identical retries return the original immutable result across
restart, including an old revision after later transitions. Changed payloads under
the same key conflict. `BEGIN IMMEDIATE` serializes the compare-and-swap and writes
state, immutable transition history, and receipt in the same transaction. Competing
resolutions cannot both commit; a contender gets `Conflict` or transient `Deferred`
and may retry. Revisions reject booleans and out-of-range integers.

`open`, `in_progress`, and `blocked` can transition to another nonterminal state
(except `open`) or to `resolved`/`cancelled`. Every transition requires a reason.
Resolution additionally requires nonempty evidence references accepted by the
configured verifier and records the actor. There is no default verifier. Text
saying “done”, elapsed due dates/TTL, missing code, and failed revalidation never
resolve work. Reopening `resolved` requires explicit `reopen=True`, status `open`,
a matching revision, authorization and a reason. `cancelled` is terminal; history
cannot silently disappear. Append-only history has SQL update/delete guards.

`list(limit=50, after=0)` returns `WorkPage` and an opaque-in-practice integer
`next_cursor`; pass it as `after`. Limits are 1–100. Ordering uses an immutable
insertion sequence, not updated time or OFFSET. Transitions do not skip items in
pagination. History uses `after_revision`/`limit`. These are bounded live reads,
not a multi-page snapshot of concurrent new inserts.

`binding_locations(scope, repo, work_id, bindings)` reads actual J05
`get_current`; moves refresh only returned location. Missing, unknown, or confirmed
orphaned bindings fall back to the same authorized repo, never a different repo or
global scope. It does not mutate the obligation or J05's binding authority. This
method is for explicit work queries and is **not** part of preflight's hotpath.

## Lessons, authority, and delivery

A fact is a belief/observation; a trusted authored instruction is a distinct
host-authorized record; an obligation is unfinished work. `tier='law'` supplied to
`import_observation` stays an untrusted observation. Imports cannot overwrite an
instruction or change a record's kind. Only the separate `author_instruction`
operation can create instructions, and only when a configured authenticated
`can_author(scope, repo)` returns exactly `True`. Default: denied. Hosts must gate
that operation on authenticated authoring intent and never route arbitrary model
output, event ingest or imported metadata into it. A principal check alone is
insufficient if that principal's generic tools expose the authoring endpoint.
Preflight has no instruction/policy mutation path. Rank never grants permission.

Authoring permission governs **future authoring**, including updates. Existing
repo-owner-approved policies are durable repository records: removing an author's
authoring permission or scope does not itself retire the repository's policy.
`can_author` must represent the repository owner's approval or delegated authority;
J06 cannot infer ownership from a principal name. Continued policy delivery to
another authorized reader is an explicit lifecycle choice, not a security claim.
Neither the author's departure nor policy retirement resolves unfinished work.

Repository owners can explicitly retire one policy with
`projection.retire_instruction(scope, repo, lesson_id, expected_version=version)`.
The separately configured `can_retire(scope, repo)` must return exactly `True` and
must represent authenticated repository-owner/admin authority; it defaults to deny.
Retirement checks the version, increments it, blanks the projection body, and keeps
a terminal retirement marker. The ID cannot be re-authored; an intentionally new
policy needs a new ID. Saved receipts stop resolving the retired version. Primary
J02 deletion/forget also suppresses instructions and saved receipts immediately,
without waiting for a J06 forget event. Authoring revocation is therefore distinct
from primary instruction revocation and explicit owner retirement.

Primary existence is **not** temporal applicability. `DurableFactAuthority` uses
actual J02 `get_entry` and `get_page` on every eligibility check. The fact, its
metadata and its backing page metadata must all be currently applicable. Recognized
timestamps are `t_created`, `created_at`, `t_observed`, `t_valid`, `t_invalid`,
`t_expired`, `expires_at`, `last_accessed`; timestamps must be full ISO timestamps
with seconds and explicit `Z` or `±HH:MM` offset (up to six fractional digits).
Naive/date-only strings, malformed offsets and reversed validity intervals fail
eligibility. Validity is `[t_valid, t_invalid)`; expiry/TTL end boundaries are
exclusive. `ttl_seconds` must be a finite nonnegative number (not a boolean or
string). TTL starts at metadata `t_created`/`created_at`, or the backing fact's
creation instant when metadata omits one. Absence of validity metadata adds no
extra limit; it never overrides a limit on another backing record. Authored
instructions inherit every backing limit and cannot grant unlimited validity.
Malformed temporal metadata yields `invalid_temporal_metadata`/`unavailable`;
corrupt records rejected within J02 yield sanitized authority/storage unavailability.
These checks run before candidates escape the projection and again before return
and receipt replay. A temporally expired but administratively active fact cannot
be passed to the accepted-forget cleanup API as if it had been deleted.

Projection ingestion is a **host/background validation boundary**: do not pass
model-supplied `source_verified`, confidence, binding IDs/revisions or authoring
decisions through as trusted fields. `project_binding_observation` obtains J05's
actual current binding and persists its logical ID and revision as well as its
repo/path/qualified name/fact version. Changed/unknown/orphaned states remain
unverified. `author_instruction` and `import_observation` also accept optional
`binding_id` and `binding_revision` for host-established code-bound records.

Bound records require `binding_authority.is_current(scope, repo, lesson, budget)`;
an absent adapter or non-boolean callback result returns explicit
`binding_authority_unavailable`, never a positive result. The included `DurableBindingAuthority(bindings, authority)` reads
J05 schema-v2 heads/history, terminal fact markers and recorded source snapshot/
generation with zero SQLite busy wait and a residual progress handler. It checks
the projected revision against the current J05 head and requires verified state,
no review flag, matching fact, and matching recorded source. This is a scoped,
read-only adapter, not a second binding writer or revalidation service. It requires
an existing schema-v2 J05 database (let J05 initialize/migrate it outside preflight)
and fails closed on an unavailable/unknown schema. A future J05 schema should
supply/update this bounded adapter. No graph rebuild,
source file read, parser or model is invoked by preflight.

Checks run before candidate emission, before rendering/receipt commit, and after
commit before response. Once J05 records changed/unknown/retracted binding state,
old bound text is suppressed even with intentionally delayed J06 projection.
Republish with a higher material lesson version after a newly verified binding
revision. A new binding revision is conservatively invalidating even if J05 found
the body unchanged. This does **not** claim atomicity with live filesystem changes
J05 has not observed, or a global transaction across the separate databases. Old
projection payloads without the additive binding fields still load as unbound
observations for compatibility; hosts must retire/reproject legacy code-bound rows
before exposing them under the new bound-record guarantee. J06 cannot reconstruct
missing binding identity from arbitrary lesson text.

The pipeline applies namespace/repo/file/lexical-symbol/action scope, kind,
confidence, projected validity, verification and retirement filters **before**
LIMIT. It orders authored policies first and then specificity/confidence/ID. It
reads at most 21 IDs (the last is a tail sentinel), hydrates/checks at most 20 bodies,
and renders at most two. SQLite scanning/sorting is interrupted by the residual
progress handler; this bounds execution by a best-effort deadline, not a constant
number of internal SQLite steps. Current primary/binding authority can still reject
all 20 while a tail exists; already-delivered candidates can also occupy a window.
Such empty selection returns `deferred/selection_incomplete`, with
`selection_complete=False`, rather than claiming nothing relevant. Delivered
results also disclose an unexamined tail. `candidate_count` reports examined
projection bodies on completed selection paths; failures before completing selection
report zero and unavailable/deferred results do not claim selection completeness.
No unbounded fallback or raw fallback occurs, even when ranking returns empty.
Default hypotheses, expired lessons, unvalidated sources and confidence below .8
are silent; explicit policies can admit hypotheses/unvalidated observations with
labels, but cannot bypass current bound-record authority.

Every body is quoted and labeled; imported observations never say “instruction”.
The renderer bounds input, strips controls, neutralizes delimiters and redacts
common credential assignments, URL credentials, authorization tokens and known
key-shaped literals. This is limited redaction, **not** a universal secret guarantee
or an injection-proofing claim. Evidence strings are never rendered or interpreted.
No parsed prose influences tool authorization, and exception text is not logged.

A session is a process-issued, immutable capability owned by principal/namespace/
repo/current authorization version, with at most 24-hour expiry. Ownership and
scope are checked **before any disk read** and again before return. Request IDs
are not authentication. Sessions are newly issued after restart. Delivery rows have
SQL uniqueness over session/code identity/lesson/material version/policy revision.
A new material version can resurface, and callers should increment policy revision
when changing policy. At most 256 receipts/512 lessons per session, 1024 live
sessions; quota exhaustion returns `deferred`. `end_session` cascades receipt and
dedup deletion; `cleanup(scope)` deletes expired sessions, and session creation
also runs cleanup. Schedule host cleanup if no new sessions arrive. If cleanup is
contended, retry; no hotpath growth exceeds the quotas.

Pass a stable `delivery_key` to deliberately replay a receipt after an uncertain
response. The digest binds target and policy; a changed request conflicts. Receipts
store IDs/versions, **not payload text**. Replay rechecks projection version,
current primary temporal authority, current binding authority, session and scope. A changed/forgotten lesson is not
replayed. Concurrent callers produce one logical delivery per uniqueness key;
losers return silence or `deferred` rather than waiting. A committed receipt can
outlive a response suppressed by a deadline/revocation race; replay the same
explicit delivery key after recovery. Without that key, the session may regard the
lesson as delivered even if the caller never received it.

Result states are `delivered`, `nothing_relevant` (normal silence), `deferred`
(contention/deadline/quota), and `unavailable` (safe generic authority/storage
failure). Invalid scopes/sessions raise generic J03 `ScopeDenied`; malformed
metadata raises `WorkError`; revision/idempotency conflicts raise `Conflict`.
Results expose safe reason codes, lesson IDs, receipt ID, replay flag, payload
bytes and `token_count_method='utf8_byte_upper_bound'`. The default 600-token target
is enforced conservatively as at most 600 UTF-8 bytes including all wrappers,
which upper-bounds UTF-8 byte-tokenizer counts without downloading a tokenizer.
The separate byte cap defaults to 2400 and can be stricter. The measured count is
bytes, not a claim of exact tokens for an arbitrary model's tokenizer.

## Deadlines and contention

The default deadline is 150 ms, finite and positive, measured with a monotonic
clock. SQLite connections have **zero busy wait**. The dedup write thus stays
within `min(residual, 100 ms)` rather than consuming a fresh timeout on each
statement. SQLite progress handlers interrupt longer projection queries; residual
checks surround selection, authority reads, rendering, transaction and return.
`DurableFactAuthority` calls true J02 `get_entry(namespace, id)` and `get_page` through a shallow,
operation-local configuration view with zero busy timeout; it never changes the
shared primary store's timeout or reinitializes its schema in preflight.

No graph/model/parser/source-file reads or full-graph ingestion occur here.
External authority/registry callbacks must implement bounded nonwaiting I/O. Python
cannot preempt an arbitrary callback or guarantee OS scheduling, filesystem-open,
fsync, or hardware latency; this is a best-effort synchronous deadline, not a hard
real-time guarantee. Tests hold independent-process RESERVED and EXCLUSIVE SQLite
locks. Work DB contention and primary EXCLUSIVE contention return visible
`deferred`; a primary RESERVED write lock permits a normal current read in DELETE
mode. The local probe timings are reported in the worker handoff, not generalized
to production latency.

## Forget, schema, migration and retention

An accepted primary forget is immediately authoritative. Every projection query
checks `FactAuthority.is_active(scope, fact_id, budget)`, and every selected payload
is checked again after rendering and just before final return. No body cache exists.
The real adapter uses J02's `get_entry` and `get_page`, not outbox/projection status. A namespace
forget does not resolve obligations. Binding history and work references remain
content-free IDs; independently authored work summaries remain work records.

After J02 accepts forget, call `LessonProjection.forget(scope, fact_id,
 tombstone_revision=event_revision)`. It refuses an active primary fact, removes
lesson bodies (already body-free retirement markers persist), retains the highest per-fact tombstone revision, and ignores upserts
at or below that revision. An inactive primary fact also rejects upserts of any
revision, so late/duplicate old events cannot resurrect it. Projection cleanup can
return `projection_purge_deferred`; primary checks still hide the body. The report
explicitly states `historical_ids_and_backups_may_remain`.

The metadata database has application ID `0x4A36574B`, user version 1, and tables:
`work`, `history`, `work_receipts`, `lessons`, `tombstones`, `sessions`, `deliveries`,
`delivery_receipts`. Startup accepts only an empty new DB or the exact known schema;
it rejects foreign/future/mismatched schemas. No legacy JSON data is rewritten.
There is no v0 adoption or silent schema migration. Future migrations must be
explicit, transactional, versioned, and preserve receipts, history and tombstone
watermarks. Do not point this API at the primary fact DB or a J05 binding DB.

Use a trusted local path and local single-host filesystem. DELETE journaling,
FULL synchronization, foreign keys, bounded requests and transactions are the
defaults. Back up the metadata DB with SQLite's backup facility during host
maintenance, and restore the DB as a unit; copying a live file is not the API.
Work histories/idempotency receipts and tombstone watermarks are retained for
correctness; delivery rows expire with sessions. No automatic obligation purge or
receipt TTL can silently invalidate historical idempotency. Current-row deletion
does not securely overwrite SQLite free pages, historical backups or external
copies. Physical backup purge requires the host's retention process and is not
promised by visibility hiding.

## Verification and integration notes for J08

Tests use actual temporary SQLite databases and J03-issued contexts. They cover
J16–J20 and J31: CAS/owner/admin/reopen/idempotency, independent-process contention,
material-version/session dedup, max-two rendering and full wrapper budgets,
malicious imported law, authored positive control, no-match control, empty ranking,
repo/session/namespace isolation, revocation, current-authority forget with pending
outbox, delayed purge, backup retention, and actual J05 binding movement/removal.

The binding integration requires J04's separately provisioned local Node adapter:
set `JITMIND_TEST_GRAFT_ADAPTER` (and optionally `JITMIND_TEST_NODE`). That dependency
is used by the integration test/background binding workflow, never by preflight.
No packages were installed and no dependency manifest was changed for J06.

Latest verification and remaining limits: [J56 revision report](j06-review-fixes.md).

### Original worker verification record (superseded by J56 revision report)

Final repository test command, run in the J06 worktree:

```sh
JITMIND_TEST_GRAFT_ADAPTER=$PWD/../J04/adapters/graft PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -q tests
```

Outcome: **81 passed, 3 expected optional-provider warnings, 20.19 seconds**.
This is the original 38-test suite plus 43 J06 cases, not the uncomposed peer
workers' entire new test suites. The J05 integration has both confirmed removal
and unknown/invalid-source cases, real source snapshots, SQLite facts and bindings.

```sh
PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m ruff check jitmind/code_memory/open_work.py jitmind/code_memory/preflight.py jitmind/code_memory/lesson_models.py jitmind/code_memory/work_storage.py tests/test_open_work.py tests/test_open_work_bindings.py tests/test_preflight.py
PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m compileall -q jitmind tests
git diff --check
```

All three exited 0. The Python example above was extracted and executed locally
against temporary SQLite databases and passed. `python -m mypy --version` reported
`No module named mypy`; a static type-checker run remains unavailable, and no
package was installed to change that. Ruff supplied lint/static checks; compileall
supplied syntax compilation, not type checking.

Explicit contention probe command:

```sh
PYTHONPATH=$PWD /Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python -m pytest -q -s tests/test_preflight.py -k independent_process_lock
```

That probe run passed 4 tests in 1.92 seconds (27 other preflight cases deselected
at that revision). Observed delivery call times: work/IMMEDIATE 0.001133 s deferred;
work/EXCLUSIVE 0.000548 s deferred; facts/IMMEDIATE 0.001859 s delivered;
facts/EXCLUSIVE 0.000867 s deferred. Release allowed subsequent useful delivery.
These are local observations, not a production deadline guarantee or benchmark win.

### Test-only dependency copies: not J06 contributions

The following peer files were copied into this fresh worktree solely to execute
the integrations. They remain owned by their respective workers; compose the
owners' reviewed versions, **do not stage these copies as J06 implementation**:

- J03: `jitmind/scope.py`.
- J02: `jitmind/storage/__init__.py`, `migration.py`, `models.py`, `sqlite.py`.
- J04: `jitmind/code_context/__init__.py`, `models.py`, `process.py`,
  `retriever.py`, `service.py`, `snapshot.py`.
- J05: `jitmind/code_memory/__init__.py`, `anchors.py`, `binding_models.py`,
  `binding_storage.py`, `revalidation.py`.

J04's `../J04/adapters/graft` was referenced read-only for the test runtime; it was
not copied or installed. J05's `code_memory/__init__.py` is now copied solely for composed tests.
Exact snapshot SHA-256 hashes are in `docs/j06-test-dependency-snapshot.json`.
These copies are not contributions; the supervisor must assemble current owner
files and rerun the combined suite, especially while J05 query fixes are concurrent. There were no commits,
GitHub writes, global/provider operations, or dependency manifest changes.

J08 followups: compose peer-owned modules/exports and rerun their combined suites;
wire the authenticated authoring/evidence adapters and bounded registry callback;
schedule source revalidation/projection and session cleanup outside preflight;
apply the host's physical backup retention process separately. A future primary
store deadline-aware read API could replace the isolated zero-timeout adapter view.

## Donor attribution and retained notice

Behavior was studied and adapted natively from **koragraph**, copyright (c) 2026
koragraph, audited donor commit `c9ce746fbd6db67e794f5f8714c8d9a8f4669e27`:
`src/practice/loop-anchors.js`, `open-loops.js`, `preflight.js`, `recall.js`,
`untrusted.js`, `repo-identity.js`; relevant dependency behavior in `resolve.js`,
`revalidate.js`, `db.js`, `paths.js`, `relevance.js`, `tier-lead.js` and the bounded
ranking/weight path was inspected. J05 owns the native binding/revalidation
adaptation; J06 does not port its graph lookup or source fingerprint machinery.

Retained ideas: unfinished work outlives anchor deletion; missing anchors broaden
only within the same repo; independent lesson/obligation records; bounded hotpath
candidate selection; authoritative empty ranking; quotation/redaction; session
dedup. Deliberate changes: no mechanical path-exists resolution, no trust inferred
from `tier='law'`, no basename repo fallback, no full-file read, no graph/model,
no positive delivery after a failed dedup write, and immediate primary forget gates.

The donor is **Business Source License 1.1**, not MIT. The user states private reuse
rights; this document does not reproduce or assert an unseen private contract.
The donor's complete supplied license notice follows. This notice does not
relabel the donor-derived adaptation as MIT or change other JITMIND components.

```text
Business Source License 1.1

License text copyright (c) 2017 MariaDB Corporation Ab, All Rights Reserved.
"Business Source License" is a trademark of MariaDB Corporation Ab.

-----------------------------------------------------------------------------

Parameters

Licensor:             koragraph

Licensed Work:        koragraph
                      The Licensed Work is (c) 2026 koragraph.

Additional Use Grant: You may make production use of the Licensed Work,
                      including for internal and commercial purposes within
                      your organization, provided that you do not offer the
                      Licensed Work to third parties as a commercial hosted
                      or managed service. A "commercial hosted or managed
                      service" means providing to third parties, for a fee or
                      other commercial consideration, the functionality of the
                      Licensed Work (or of a product or service that is
                      substantially similar to the Licensed Work) as a hosted,
                      managed, or software-as-a-service offering. Any use that
                      is not a commercial hosted or managed service is
                      permitted under this Additional Use Grant.

Change Date:          Four years from the date the Licensed Work is published.

Change License:       Apache License, Version 2.0

-----------------------------------------------------------------------------

Terms

The Licensor hereby grants you the right to copy, modify, create derivative
works, redistribute, and make non-production use of the Licensed Work. The
Licensor may make an Additional Use Grant, above, permitting limited
production use.

Effective on the Change Date, or the fourth anniversary of the first publicly
available distribution of a specific version of the Licensed Work under this
License, whichever comes first, the Licensor hereby grants you rights under
the terms of the Change License, and the rights granted in the paragraph
above terminate.

If your use of the Licensed Work does not comply with the requirements
currently in effect as described in this License, you must purchase a
commercial license from the Licensor, its affiliated entities, or authorized
resellers, or you must refrain from using the Licensed Work.

All copies of the original and modified Licensed Work, and derivative works
of the Licensed Work, are subject to this License. This License applies
separately for each version of the Licensed Work and the Change Date may vary
for each version of the Licensed Work released by Licensor.

You must conspicuously display this License on each original or modified copy
of the Licensed Work. If you receive the Licensed Work in original or
modified form from a third party, the terms and conditions set forth in this
License apply to your use of that work.

Any use of the Licensed Work in violation of this License will automatically
terminate your rights under this License for the current and all other
versions of the Licensed Work.

This License does not grant you any right in any trademark or logo of
Licensor or its affiliates (provided that you may use a trademark or logo of
Licensor as expressly required by this License).

TO THE EXTENT PERMITTED BY APPLICABLE LAW, THE LICENSED WORK IS PROVIDED ON
AN "AS IS" BASIS. LICENSOR HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS,
EXPRESS OR IMPLIED, INCLUDING (WITHOUT LIMITATION) WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND
TITLE.

MariaDB hereby grants you permission to use this License's text to license
your works, and to refer to it using the trademark "Business Source License",
as long as you comply with the Covenants of Licensor below.

-----------------------------------------------------------------------------

Covenants of Licensor

In consideration of the right to use this License's text and the "Business
Source License" name and trademark, Licensor covenants to MariaDB, and to all
other recipients of the licensed work to be provided by Licensor:

1. To specify as the Change License the GPL Version 2.0 or any later version,
   or a license that is compatible with GPL Version 2.0 or a later version,
   where "compatible" means that software provided under the Change License can
   be included in a program with software provided under GPL Version 2.0 or a
   later version. Licensor may specify additional Change Licenses without
   limitation.

2. To either: (a) specify an additional grant of rights to use that does not
   impose any additional restriction on the right granted in this License, as
   the Additional Use Grant; or (b) insert the text "None".

3. To specify a Change Date.

4. Not to modify this License in any other way.

-----------------------------------------------------------------------------

Notice

The Business Source License (this document, or the "License") is not an Open
Source license. However, the Licensed Work will eventually be made available
under an Open Source License, as stated in this License.
```
