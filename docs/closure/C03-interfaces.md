# C03 transactional history interfaces

Implemented interface contract; see transactional-history.md for verification evidence. All storage APIs are trusted-host
only. C01 authorizes namespace/repo before acquisition and before return.

* `outbox_events(namespace_id, *, after_revision=0, limit=100) -> tuple[ProjectionEvent,...]`
* `get_event(namespace_id, event_id) -> ProjectionEvent | None`
* Frozen `ProjectionEvent(namespace_id, event_id, revision, memory_ids: tuple[str,...])`.
  Includes ALL committed records, even delivered reference events. Independent
  consumers maintain their own receipts. `pending_events` is unchanged.
* `snapshot_at(namespace_id, *, transaction_at=None, revision=None, valid_at=None,
  limit=100, repo_id=None, snapshot_id=None, eligible_at=None) -> HistoricalSnapshot`.
  Strict aware timestamps, UTC normalization; transaction/revision mutually exclusive.
  Repo facts require explicit matching repo/snapshot selectors; omitted means
  namespace-only facts. Limit 1..1000 plus processing and byte bounds.
* `HistoricalSnapshot(namespace_id, revision, recorded_at, baseline_revision,
  baseline_recorded_at, coverage, complete, truncated, unknown_validity, facts,
  unknown_scope=False, unknown_eligibility=False)`.
  Coverage available/unavailable; `.entries` gives detached MemoryEntry tuple.
* `HistoricalFact(entry, revision, fact_key, repo_id, snapshot_id, single_valued, page_id)`.
* `commit_temporal(request, expected_revision, proposal, *, corrections=()) -> DurableReceipt`.
  `ValidityCorrection(memory_id, valid_from, valid_to)` changes validity of active or
  superseded facts in SAME transaction as normal proposal. Body bound to receipt
  digest. Prior versions unchanged. Metadata `fact_key`, `single_valued`, `repo_id`,
  `snapshot_id` declares temporal identity; exact case-sensitive identifiers.
* `enable_history(*, quiesced=True, recorded_at=...)`: explicit v1 -> v2 migration.
  New DBs default v2; old v1 remains usable without history. Baseline at existing
  namespace revision and supplied recorded time; earlier history unavailable.
  Stop all writers for migration; stale v1 instances are fenced on next transaction.

Same-time operations ordered by revision (time selects last). UPDATE retirement
retains effective intervals in latest history and old perspectives remain immutable.
DELETE hides tombstoned versions in latest history and ordinary reads. TTL eligibility
is separate from retention. Missing valid start is unknown, missing end is open.

* `historical_page(namespace_id, page_id) -> Page | None`: trusted-host immutable
  retained source lookup; current get_page visibility unchanged.
* `compact_history(destination, namespace_id, *, before_revision, quiesced=False,
  timeout_seconds=10) -> Path`: trusted administrative NEW-image copy, source unchanged.

C07 layout: immutable `history_coverage`, `history_revisions`, `history_versions`,
`history_pages`; versions FK history_pages by namespace/page ID. Compact-copy keeps
latest version per fact at cutoff plus later versions, advances coverage, and keeps
original version IDs, current facts, both source-page tables, receipts, aliases and
all event/revision identities. No live trigger dropping or implicit purge. This
prunes version records only; broader text/backups/log retention stays C07-owned.
Legacy split fact/page provenance is resolved only in archived metadata; ambiguous
scope is excluded with unknown_scope, never silently widened. Malformed TTL remains
accepted by ordinary APIs but historical eligible_at queries mark unknown_eligibility.
Backups retain independent retention policy.

C09 existing backup/restore uses one consistent SQLite image, exact v1/v2 schema
validation and new destination only. Old v1 code rejects v2; rollback needs compatible
code retaining v2 writes/history or explicit reconciled export, never stale restore.
J09 stays separate and supported; no two-file atomicity claim.

Earlier worker checkpoint evidence: 952 full-suite tests passed, zero failed/skipped (169.66s);
70 focused history tests passed. Real parser/process tests included. Ruff, formatting,
Python 3.10 grammar parsing and runnable example passed. No native Python 3.10 or
mypy run. See ../transactional-history.md for exact commands, failure corrections,
migration/maintenance limits and remaining composed CI/review gates.

## Verified integrity remediation contract (2026-09-29)

Public signatures and receipt shapes stay unchanged. The unpublished exact v2
schema adds immutable `history_effects` keyed by `(namespace_id,memory_id,revision)`.
It records every version's indexed identity/scope/status/interval and SHA-256 of
its serialized fact and original page. Missing/mismatched effects fail with generic
`StorageFailure` before indexed scope selection, under the existing VM ceiling;
selected payloads and source provenance are checked before return. These records
detect inconsistent data, not an attacker rewriting the entire database.
Compact-copy must copy effects for precisely the retained versions. Exact earlier
experimental v2 layouts are rejected, never silently certified or auto-upgraded;
retain their backups and use supervisor-reviewed reconciliation. Published v1
still opens unchanged and explicitly migrates atomically through `enable_history`.
Overlapping single-valued legacy baselines now reject migration/import atomically,
instead of committing a baseline that fails its first query.

C02 validity policy: current/corrected fact intervals govern effective validity;
original page t_valid/t_invalid are immutable evidence, never reapplied as current
intervals. Independent fact AND original-page expires_at/ttl_seconds constrain
historical eligible_at; TTL originates at fact t_created. Invalid expiry metadata
produces unknown_eligibility and excludes the result. C02 consumers remain owned
by C02. get_event and historical_page preflight serialized byte sizes inside the
same read transaction before acquisition (262144 and 1048576 bytes respectively).

Remediation verification: **986 full-suite tests passed, zero failed/skipped**,
including actual provisioned parser/process tests; 287 targeted compatibility
tests and 54 final integrity/process tests passed in their separate runs.
Five applicable independent probes passed unchanged; two older migration
observations intentionally mismatch the current documented contract. Exact
commands, development failures, owned files, fingerprints and residual gates:
[C03-remediation.md](C03-remediation.md). No mypy/native Python 3.10/3.11 claim.
