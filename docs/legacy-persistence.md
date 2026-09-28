# Legacy persistence repair (J-01)

## Audit and implementation plan

Before editing, `rg -n 'file_lock|cleanup_expired' .` identified every caller:

- `advanced_memory.py`: `_persistence_lock`, used by save, add_entry,
  update_entry, delete_entry, supersede_entry, touch, cleanup_expired;
  cleanup called by constructor and load. promote_demote and ranked reads also
  mutate tiers and need the same transaction boundary.
- `memory.py`, `page.py`: add locks then calls save; public save needs locking too.
- `ttl_memory.py`, `ttl_page.py`: add and cleanup lock across refresh/write;
  constructors and load call cleanup; public save needs locking too.
- `profile/profile_store.py`: update locks then calls save.
- `utils/checkpoint.py`: save_checkpoint locks; delete needs the same exclusion.
- `utils/__init__.py`: re-export only; `utils/file_lock.py`: definition.
- Cleanup callers outside stores: `tests/test_temporal_queries.py`,
  `tests/test_ttl_memory.py`, `tests/test_ttl_page.py`,
  `tests/test_ttl_before_after.py`, `tests/test_ttl_standalone.py`,
  `tests/run_ttl_tests.py`, `examples/quickstart/ttl_usage.py`.
  README files and `assets/readme/final_svgs/04_bitemporal_model.svg` mention it;
  `tests/test_ttl_before_after.py` also prints the method name.

Plan: add regressions for lost updates, acknowledged failures, and lock ownership;
replace age-based lock ownership with kernel ownership; retain complete mutation
transactions; fail closed on corrupt input; verify actual tests and compilation.
No schema redesign, provider work, dependency changes, or multi-store transaction.

## Behavior and compatibility

- POSIX `flock` owns the lock until the last owning descriptor closes or the
  process exits. The `.lock` file is persistent. Never unlink it during normal
  operation, never steal based on mtime, and do not run old O_EXCL-locking writers
  alongside new writers. Stop all legacy writers before upgrading. Leftover old
  lock-file contents do not prevent acquisition after all old writers stop.
- Same-thread nesting on the same resolved path is reentrant, including existing
  add -> save and profile update -> save calls. Other threads share a per-path
  mutex; independent processes open separate descriptors and contend in the
  kernel. Timeouts use monotonic time. `timeout_s` must be finite and >= 0;
  `poll_s` must be finite and > 0. Booleans are rejected for both waits. Sleep
  chunks are capped at one second so large finite waits do not overflow the
  platform sleep implementation. Zero timeout performs one immediate attempt.
  `stale_s` stays in the signature but is ignored (diagnostic-only).
- A dedicated descriptor registry and lifecycle guard cover open/registration
  and close/removal. POSIX fork preparation acquires that guard; the parent
  releases it, and the child closes every tracked inherited descriptor without
  `LOCK_UN`, then replaces the registry and synchronization locks. Per-path
  mutexes precede the lifecycle guard; no mutex or kernel-lock wait occurs while
  holding the guard. Fork does not wait for a store transaction to finish.
  Python 3.12 warns about multithreaded fork: these bounded Darwin regressions
  qualify descriptor handling only, not arbitrary post-fork application work.
  Prefer spawned processes. Fork from signal handlers or arbitrary user hooks
  within descriptor operations, resuming inherited lock contexts in the child,
  and third-party at-fork callbacks that acquire these locks are unsupported.
- This is cooperative, single-host filesystem locking, not distributed locking.
  All writers must use the same lock path. Only macOS execution has been tested
  here; network filesystem semantics and hardware power-loss guarantees are not
  qualified. Windows and other non-POSIX platforms raise `NotImplementedError`;
  a Windows msvcrt strategy remains unimplemented and unverified pending CI.
- Advanced cleanup holds the process lock through refresh, one captured expiry
  time, filtering, and save. `retain_history=True` preserves temporal histories;
  false purges inactive rows even when the returned newly-expired count is zero.
  Public saves and tier mutations also hold the lock. Ranked retrieval already
  changes tiers; those changes now receive a persistence acknowledgement.
  Whole-state `save` still deliberately replaces the snapshot supplied by the
  caller; it does not merge an arbitrarily stale caller snapshot.
- JSON formats, legacy abstracts migration, page envelopes/lists, TTL behavior,
  and public method signatures remain supported. Malformed JSON, wrong envelopes,
  and invalid model data raise `CorruptStoreError`; originals are not erased,
  silently reset, renamed, or quarantined. Recovery requires explicit repair by
  the owner. An empty valid store must use its actual envelope/list, not `{}`.
- Atomic writes sync temporary-file contents, replace, then sync the parent
  directory. `PersistenceError.outcome_uncertain=False` means failure before
  replacement: existing durable bytes are unchanged. True means replacement
  completed but the directory sync failed: new bytes are visible, but survival
  of a crash is uncertain. Do not treat it as rollback or retry appends blindly.
  Nested directory creation is not a transaction over ancestor directories.
- Public error text contains no paths or payloads. The original exception remains
  in `__cause__` for trusted diagnostics; full exception-chain tracebacks can
  contain private details and should not be exposed to untrusted clients.
- All affected memory/page stores reload their cache inside the transaction lock
  after a failed write (including uncertain outcomes). A failed reload clears
  unpublished cache data; disk reads/mutations continue to refuse unreadable or
  corrupt originals. Simple stores publish candidate caches only after success.
- No generation or provider call has been moved under a store lock. No multi-file
  transaction or idempotency API is introduced by this repair.

Example acknowledgement handling:

```python
from jitmind.utils import PersistenceError, CorruptStoreError

try:
    store.add("new memory")
except PersistenceError as error:
    if error.outcome_uncertain:
        # Inspect/reconcile the published state before retrying an append.
        raise
    # Existing durable bytes were preserved; caller decides whether to retry.
    raise
except CorruptStoreError:
    # Stop writes and arrange explicit recovery of the original file.
    raise
```

## Reviewed callers outside this ticket's write scope

`MemoryAgent.memorize` lets primary memory/page write failures propagate, so it
cannot return its ordinary result after those writes fail. Generation occurs
outside store transactions. However `_resolve_conflicts` catches failures around
`supersede_entry`, and the optional profile update is also caught. Those optional
agent integration catches need a separate ticket if their failures must become
part of the agent-level acknowledgement. Memory and page writes remain separate;
a page failure can follow a successful memory write. Unified idempotency and
multi-store acknowledgement belong to J-02.

## Checkpoint invariants for the later J07 integration

J07 has not been merged here. Carry these invariants and their regression tests
when composing its checkpoint changes:

- Save and delete use the same canonical checkpoint lock through validation,
  mutation, and durability acknowledgement.
- Keep the existing `{thread_id, timestamp, state}` envelope readable. Preserve
  returns: absent load → `None`, absent delete → `False`, successful save →
  `None`, acknowledged delete → `True`.
- A load whose file disappears between existence checking and opening returns
  `None`, just like an initially missing checkpoint. Malformed JSON or an invalid
  envelope still raises `CorruptStoreError`; save/delete preserve corrupt bytes.
- Keep atomic replacement and propagate storage errors. Pre-replacement failure
  preserves old bytes; post-replacement directory-sync failure reports
  `PersistenceError(outcome_uncertain=True)`.
- After unlink, a directory-sync failure also reports `outcome_uncertain=True`.
  The file is already absent; this is neither an acknowledged deletion nor a
  rollback. Tests cover failed unlink, failed late sync, and corrupt originals.
- Never remove persistent lock files, steal by age, or swallow persistence errors.
- Keep any new ID/path mapping consistent across save, load, delete, and legacy
  lookup. The existing lossy ID sanitization remains a separate followup.

## Verification evidence (macOS, Python 3.12.12)

Commands were run from the assigned J01 worktree with the installed isolated
Python; no packages or manifests were changed and no provider calls were made.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider tests/test_file_lock_regressions.py tests/test_atomic_io_regressions.py tests/test_legacy_persistence.py
# Remediation targeted suite: 107 passed, 3 expected optional-provider warnings (3.78s).

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider
# Remediation full suite: 145 passed, 3 expected optional-provider warnings (22.65s).
# Preserves the previous 133 cases and adds 12 regressions.

PYTHONPATH=$PWD python -m compileall -q jitmind tests
# Exit 0.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD python -m ruff check --no-cache --isolated --select E9,F jitmind/utils/atomic_io.py jitmind/utils/file_lock.py jitmind/utils/checkpoint.py jitmind/utils/__init__.py jitmind/schemas/advanced_memory.py jitmind/schemas/memory.py jitmind/schemas/page.py jitmind/schemas/ttl_memory.py jitmind/schemas/ttl_page.py jitmind/profile/profile_store.py tests/test_file_lock_regressions.py tests/test_legacy_persistence.py tests/test_atomic_io_regressions.py
# All checks passed.

git diff --check
# Exit 0.

python -m mypy --version
# Exit 1: No module named mypy. Type checking not executed.
```

Before implementation, `python -m pytest -q tests/test_legacy_persistence.py`
(with the same interpreter and PYTHONPATH above, when the file contained the
initial two tests) produced **2 failures**: previously inactive rows were not
persistently purged, and replacement failure did not raise. The process schedule
was verified on the repaired implementation, not rerun against an untouched
baseline. Its two orders use spawned children, an in-lock pause, a kernel
contention probe, release before waiting for the contender, and a fresh reader.

Before remediation, the unrestricted Ruff rule set was also run on the same 13 Python files:
`python -m ruff check --isolated --output-format concise <the 13 files above>`.
It reports **128 findings** (style/annotation modernization, import order,
exception-policy and simplification rules). This repair is not a claim of clean
unrestricted style lint; the explicit E9/F error checks pass. No mass style or
typing rewrite was applied to the legacy stores.

Coverage includes both cleanup orderings; live-owner age resistance; kill and
confirmed exit followed by exclusive successor ownership; nested/thread locking;
invalid waits; one cleanup clock; history retention/purge; corruption refusal;
old-format reads and writes; precommit/late errors and cache restoration; and
mkdir, temporary creation, fdopen, write, flush, file fsync, replace, directory
open and directory fsync failure cleanup. Tests use real stores and filesystem
operations with injected failures, not live services or power-loss simulation.
Remediation adds 12 cases: both real fork lifecycle windows (with idle children
reaped), four Boolean wait rejections, two oversized finite wait paths, one
barrier-controlled public checkpoint load/delete race, and three deletion
failure/corruption cases. These are correctness regressions, not external
benchmarks or platform-wide qualification.
