# C03 final-data-integrity remediation evidence

Work is uncommitted in C03 on base `48a2a72a4637298f46bce79d7bd045efcc30d58c`.
This pass changed `jitmind/storage/history.py`, `jitmind/storage/sqlite.py`,
`tests/test_history_integrity.py`, `tests/test_transactional_history.py`,
`tests/test_transactional_history_faults.py`, and these C03 docs. Existing earlier
C03 changes in storage models/exports and MemoryAgent were retained, not replaced.
No C02/peer production code, dependencies, manifests or donor code were added.
The existing parser runtime symlink is provisioning, excluded from source changes.

## Findings and resulting contracts

1. Page-only historical scope and split fact/page provenance were **already fixed**
   in current C03. Both original page-only probes passed before edits; the unchanged
   scoped compatibility tests remain passing. Shared source reconciliation is now
   also used during returned fact validation. Namespace and contradictory scope
   checks remain fail-closed; ambiguous imported scope produces unknown coverage.
2. Both independent corruption probes reproduced before edits. Immutable effect
   records now detect missing versions and mismatched indexed metadata before scope
   selection; selected serialized fact/page digests and decoded metadata are checked
   before return. The probes now pass. Generic errors do not disclose private content.
3. Overlap migration committed an unusable baseline before remediation. Baseline
   key validation now rejects migration/import atomically. The original local
   `test_migrated_conflicting_key_never_uses_insertion_order` still asserts
   `InvalidRequest`, now at migration, and additionally proves byte-for-byte v1
   preservation and unavailable history. New tests cover receipts across multiple
   namespaces, failed imports and adjacent non-overlapping imports.
4. `get_event` and historical-page lookup now preflight bytes before payload SELECTs
   under exact-schema validation and a read transaction. Tests spy on actual SQL
   and prohibit oversized decoding; numeric revision checks include the signed
   64-bit positive boundary. Snapshot bytes include both fact and source page.
5. Corrected fact validity remains primary. Original page intervals stay immutable;
   independent page/fact TTL and expiry restrict historical eligibility. Positive
   correction and negative page-expiry/invalid-TTL tests exercise actual imports,
   structured commits and original-page retrieval.

The main simplification is one shared fact/page provenance reconciliation helper
for history publication and selected-result verification. No competing C02 API.
Exact layout/migration conditions are in [C03-interfaces.md](C03-interfaces.md).

## Environment and commands

All commands run from this C03 worktree using actual candidate imports and disposable
fixtures. Python 3.12.12, SQLite 3.51.2, Darwin arm64. In commands below:

```sh
PY=/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/venv/bin/python
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
```

Targeted compatibility run (before the final six extra regression cases):

```sh
$PY -m pytest -p no:cacheprovider tests/test_history_integrity.py tests/test_transactional_history.py tests/test_transactional_history_faults.py tests/test_durable_storage.py tests/test_durable_remediation.py tests/test_durable_migration.py tests/test_durable_python310.py tests/test_scoped_durable.py tests/test_temporal_history_sqlite.py -q
```

**287 passed, 0 failed/skipped, 22.42s**, three optional-retriever warnings.
Log: `/private/tmp/C03-remediation-targeted.log`.

Final integrity and fault/process regression run:

```sh
$PY -m pytest -p no:cacheprovider tests/test_history_integrity.py tests/test_transactional_history_faults.py -q
```

**54 passed, 0 failed/skipped, 5.13s**, three optional-retriever warnings.
Log: `/private/tmp/C03-integrity-final.log`. This includes 34 new integrity tests
and all 20 existing fault/process tests; transaction row-count assertions now
also cover `history_effects` at every injected failure/process exit stage.

Independent review source was not edited:

```sh
$PY -m pytest --confcutdir=/tmp/jitmind-closure-c03-review-TKarL5 -p no:cacheprovider /tmp/jitmind-closure-c03-review-TKarL5/test_review_probes.py -q
$PY -m pytest --confcutdir=/tmp/jitmind-closure-c03-review-TKarL5 -p no:cacheprovider /tmp/jitmind-closure-c03-review-TKarL5/test_review_probes.py -k 'not overlap_migration_observation and not failed_migration_after_prior_namespace' -q
```

Before edits the whole unchanged probe file had **4 passed, 3 failed**: the two
corruption failures and an older oracle requiring malformed-scope migration to
abort. After edits: **5 passed, 2 failed, 0 skipped, 0.99s**. Those two failures
are intentionally obsolete behavior observations: overlap migration now rejects
before commit; ambiguous scope migration already used explicit unknown coverage
before this pass. They were not deleted or rewritten. The five still-applicable
cases cover both scope variants, both corruption variants, and foreign-schema byte
preservation. Their selected unchanged run passed **5 tests, 2 deselected, zero
failed/skipped, 1.00s**. Replacement current-contract tests cover both old observations with
stronger positive/negative and rollback checks. An initial invocation without
`--confcutdir` failed collection on an unrelated inaccessible `/tmp` directory;
it executed no tests and is not counted as test evidence.
Logs: `/private/tmp/C03-remediation-probes.log` and
`/private/tmp/C03-remediation-probes-meaningful.log`.

Full owner suite:

```sh
JITMIND_TEST_NODE=/opt/homebrew/bin/node $PY -m pytest -p no:cacheprovider -q --junitxml=/private/tmp/C03-remediation-full.xml
```

**986 passed, 0 failed, 0 skipped, 112.74s**, three optional-retriever warnings.
JUnit confirms 986 cases and no skips/errors/failures. Log
`/private/tmp/C03-remediation-full.log`.
The actual provisioned runtime is
`/Users/divyamtalwar/jitmind-implementation-20260928-yfk0b_jf/closure-20260929/runtime/parser/node_modules`.
No provider execution or skipped-parser integration claim.

Static verification:

```sh
$PY -m ruff check --no-cache jitmind/storage tests/test_history_integrity.py tests/test_transactional_history.py tests/test_transactional_history_faults.py
$PY -m ruff format --check jitmind/storage tests/test_history_integrity.py tests/test_transactional_history.py tests/test_transactional_history_faults.py
git diff --check
```

Passed; eight Python files formatted. Five `jitmind/storage/*.py` parsed using
`ast.parse(..., feature_version=(3,10))`. This is syntax/static checking, **not a
mypy typecheck or native Python 3.10/3.11 runtime run**.

Development failures: first history/scoped run 83 passed/1 failed because the
old local overlap test expected failure at query rather than migration; strengthened
as described above. First new integrity run 22 passed/6 failed because its source
page assertion used a legacy alias instead of the remapped immutable page ID;
corrected to use the actual imported fact's source_page_id. Initial lint found one
import-order issue in the new test file; fixed. No existing scoped failure assertion
was removed or relaxed. Independent probe mismatches are separately described above.

## Remaining limits

Checks detect inconsistent/missing recorded data, not hostile rewriting of every
SQLite table and digest. Unselected payloads are not exhaustively decoded; metadata
inventory comparisons are VM-bounded and can return explicit capacity failure.
Old experimental v2 images require reviewed reconciliation and are not silently
upgraded. Existing structural opening/backup checks remain. This is not physical
power-loss, native minimum-version/Linux CI, provider, production-data or whole-ZIP
qualification. Supervisor final C02/C07/C09 composition and release review remain;
this pass did not edit or run peer consumer code. Backups/logs are retained.

## Verified source fingerprints (SHA-256)

Base HEAD `48a2a72a4637298f46bce79d7bd045efcc30d58c` plus these uncommitted files:

| File | SHA-256 |
| --- | --- |
| `jitmind/storage/__init__.py` | `0c5823ee8e2ff29f4ba20ea7b9c7105aad68f9bd29873a4e4e2e108914c7a6d1` |
| `jitmind/storage/history.py` | `8ad10d6ca381b2fd199fdd3aedd0e8368d9569fdf16b682fd1381044615796d1` |
| `jitmind/storage/migration.py` | `26cf90e0911fe6bfae875ef31c50f0b8cbf94b05c2d5a694cd5408622533a15a` |
| `jitmind/storage/models.py` | `189c4179ec36f2ecefd7587004ade27f2c87da88142981e721b25c51be38635b` |
| `jitmind/storage/sqlite.py` | `ee29b1fa1d49f106c4c95e4b41e9a166e9ad433e6ac41c9e4b62c543e2f6424c` |
| `jitmind/agents/memory_agent.py` | `920b9e45a4da6f1770df325a2b8e1112f9a70826c530ab9130d86835e01c1aa0` |
| `tests/test_history_integrity.py` | `222aaff4a5495c12311abb023e1c1e1b6c9b2dd8b89d6a955b6cb2bce190e9f7` |
| `tests/test_transactional_history.py` | `4e1c55941dce80bb3668c2feeea1c8bd375471ac3c94bf404bb44fe1a2872e1d` |
| `tests/test_transactional_history_faults.py` | `44606adff2e33a3691824563e4a212e83692f88ee72db571144c4b51b7f6aa32` |

The JSON manifest is also retained at `/private/tmp/C03-remediation-fingerprints.json`.
No temporary dependency copies were used in this remediation. Existing parser
symlink provisioning remains excluded from the source diff. The donor reference
`adapters/graft/donor-audit.json` pins `80692e5ad0bc8e8f7e1edea648247d90b9f76820`
(`@nanonets/graft` 0.20.0); no donor code or private contracts were copied here.
