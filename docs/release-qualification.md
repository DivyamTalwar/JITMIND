# Release qualification and recovery

This is a reproducible **checks gate**, not a production certificate. The J08
worktree contains the baseline library plus qualification tooling and reviewed
J10 test contributions. Final owner sources were exercised in a frozen temporary
composite for mapping review; the supervisor must assemble the actual candidate
and rerun qualification before scenarios receive gate credit.
No peer-worktree result, mocked tool test, or generic pytest exit code establishes
acceptance of the combined candidate.

The original audit baseline is Git
`dc6877e85b4048f81ce801d8700dfb934277e5f3`: Python 3.12.12 on macOS arm64,
38 tests passing with three expected optional-provider warnings, as recorded by
the supervising baseline run. This document does not reinterpret that result as
validation of new components, other platforms, or production durability.

## Prepare an offline candidate check

Use an isolated Python with the reviewed core dependencies and pytest already
installed. Wheel building also needs the existing `pyproject.toml` build
requirements, **setuptools >=68 and wheel**, in this interpreter: build isolation
and network resolution are deliberately disabled. Python 3.12's default venv does
not necessarily include setuptools. Missing prerequisites produce a failed check,
not an automatic package installation or a retry.

The supervisor has supplied setuptools 84, wheel 0.48 and 31 hashed offline
wheels for the current isolated environment. The previous missing-setuptools
blocker is resolved; this remediation does not install anything or claim an
installed-wheel result. The supervisor runs that path after final assembly.
For another environment, separately supply a reviewed wheelhouse containing the complete
core dependency closure for the target Python/platform. Qualification installs
only into its own fresh temporary venv. It never installs a donor whole package,
Node, Cohere, Neo4j, FAISS, or another user product. Do not put provider credentials
into this environment. No extra Python dependency was added for this tool.

Portable shell recipes (replace the placeholder paths):

```sh
JITMIND_PYTHON=/path/to/isolated/bin/python
JITMIND_WHEELHOUSE=/path/to/reviewed-core-wheels

# Optional prerequisite preparation by the supervisor, offline:
"$JITMIND_PYTHON" -m pip install --no-index --only-binary=:all: \
  --find-links "$JITMIND_WHEELHOUSE" 'setuptools>=68' wheel

# Run from the candidate checkout. Output must be NEW and outside the checkout.
# No --worktree means dirty source/untracked files are refused.
"$JITMIND_PYTHON" scripts/qualify_jitmind.py run \
  --output /path/to/new-candidate-report \
  --wheelhouse "$JITMIND_WHEELHOUSE" \
  --overall-seconds 900 --command-seconds 180

# Developer checks explicitly allow and label an uncommitted worktree:
"$JITMIND_PYTHON" scripts/qualify_jitmind.py run --worktree \
  --output /path/to/new-developer-report \
  --wheelhouse "$JITMIND_WHEELHOUSE"
```

The output parent must already exist. The output directory is created exclusively
with mode 0700; reruns need new paths. Reports and logs are never overwritten.
Exit 0 means `checks_only_complete`, 1 means failed/incomplete checks, and 2 means
invalid/unavailable qualification input or setup. **No exit code means production
signoff.** `release_qualified` remains false in this version.

The fixed suite is:

1. `python -m pytest -q tests -p scripts.qualify_jitmind`, with explicit checkout
   `PYTHONPATH` and third-party pytest plugin autoload disabled.
2. `python -m compileall -q jitmind scripts/qualify_jitmind.py`.
3. `python -m pip wheel --no-index --no-deps --no-build-isolation`, building from a
   fresh copy of the pinned source snapshot. Ignored stale build outputs do not
   enter this copy. The wheel is retained and hashed in the report.
4. `python -m venv` in a private temporary directory outside the checkout.
5. That venv's `python -I -m pip install --no-index --only-binary=:all:` of the built
   wheel and reviewed wheelhouse dependencies. A missing dependency fails; no
   network fallback or install into the calling interpreter occurs.
6. That venv's `python -I` executes `tests/test_standalone_install.py` from a
   temporary working directory, without `PYTHONPATH`. PATH contains only that
   venv's bin directory, so Node cannot be accidentally discovered on host PATH.

All commands use trusted argv constructors with `shell=False`. Acceptance-map and
imported-report fields never become executable commands. The environment is a
small explicit allowlist; API keys, provider configuration, proxy variables,
PYTHONPATH, pip credentials/config, NODE_OPTIONS and pytest injection settings are
not inherited. The runner gives HOME/TMPDIR private scratch locations. It does
not read user credential files. Repository tests/build code are trusted executable
candidate code; process containment is not an OS sandbox for malicious tests.

## Standalone installed-package contract

The smoke can also be run against an actual already installed candidate venv:

```sh
# Set CHECKOUT before leaving it. The result path must not exist.
CHECKOUT=$PWD
cd /path/to/empty-temporary-directory
env -i LANG=C.UTF-8 PATH=/path/to/candidate-venv/bin \
  /path/to/candidate-venv/bin/python -I \
  "$CHECKOUT/tests/test_standalone_install.py" --installed-smoke \
  /path/to/candidate-venv /path/to/new-installed-results.json
```

Prefer the complete runner: it also scrubs provider keys and ties the install to
the freshly built wheel hash. Direct users must supply an equally clean environment.
The smoke asserts `sys.flags.isolated`, the expected venv prefix, a jitmind module
origin inside that venv, absence of checkout from cwd/sys.path, and absence of
Cohere, Neo4j, FAISS and discoverable Node. Socket connections are denied inside
the smoke process, including accidental provider connections.

It executes actual legacy `MemoryAgent.memorize` and `ResearchAgent.research`
using a deterministic generator, reopens the persisted memory/pages, and checks
retrieved evidence, answer and provenance. No language-model quality claim follows
from this fake generator. If the durable backend is installed, its `diagnostics()`
records actual application connection settings; the separate stdlib probe is
explicitly not a product-storage setting.

The optional adapter must import without Node. Configuring an absent Node/adapter
path must produce its meaningful typed `Unavailable` capability error; arbitrary
`FileNotFoundError`, `AttributeError`, a failed import, or silent remote fallback
is not acceptance. The corrected J04 constructor raises `CapabilityUnavailable`,
a subtype of `Unavailable`, with `node_executable_unavailable` for a missing or
nonexecutable Node, `adapter_directory_unavailable` for an absent adapter directory,
and `adapter_runner_unavailable` for a missing, unreadable or symlink runner.
Relative configuration paths still raise `ValueError`. Construction does not
start Node, import optional providers or fall back remotely; final owner tests
executed these boundaries successfully in the staged composite.

The unchanged installed-smoke helper currently accepts only `runtime_unavailable`
or `adapter_unavailable`. Executing that helper against final J04 reproduces an
assertion failure on `node_executable_unavailable`; this is a smoke/product
reason-code contract mismatch, not an unfixed constructor. The baseline-only J08
checkout also lacks `jitmind.code_context`. Neither checkout helper tests nor the
staged helper reproduction establish installed-wheel qualification. No install
was performed in the final mapping task, and J32 remains unrun. The supervisor
must resolve the reason-code contract explicitly and rerun the isolated smoke;
this task does not change its assertions.

Pytest's two checks in this file verify the guard and the fake-generator fixture
**from the checkout**. They cannot satisfy J32. J32's separate exact node identity
is `tests/test_standalone_install.py::installed_smoke`; only the runner's installed
artifact can supply its result.

## Acceptance and evidence schema

`docs/engineering/acceptance-map.json` retains the derived handoff's **32 cases and
41 required ticket stages**. Each case has one owner. Repeated stages remain
required, including J02 failure propagation, J03/J04 scope/path/cache checks,
J02/J04/J05 outbox chains, and J02/J04/J05/J06 forget/retention chains.

The map carries the supplied `source_handoff_sha256`, the digest of the derived
map itself, and the frozen temporal oracle's digest and seven expected answers:
3, 3, 4, 3, 4, 5, 3. Latest-perspective Jan20 is 4; equal instants with timezone
offsets agree; conflicting single-valued overlaps are errors. The legacy naive
API is unchanged. This derived map is not the original full-fixture handoff and
is not execution evidence or a signature.

Every case/stage stays `status: not_run` in source control. Nodes contain:

- `node_id`: exact pytest ID including every parameter suffix, or the distinct
  installed-smoke identity. Prefix matches and inferred test names do not count.
- `kind`: real storage, injected storage fault, real parser, migration, projection,
  public API, temporal oracle, or installed package. Tool/qualification tests and
  mocked-tool records cannot become product acceptance.
- `source`: repository-relative source file and SHA-256.
- `fixtures`: nonempty source-attested artifact references. Existing peer tests
  construct synthetic deterministic fixtures in their test source; they are
  labeled as such and do not claim original full-fixture execution.

Final mappings are based on actual owner-source definitions and staged execution,
not inherited peer pass reports. Changing a test source requires reviewing its
assertions and refreshing its hash. `nodes: []` or
`additional_mapping_required: true` still blocks a stage. All eight previous
mapping gaps now have the exact J10 supplement nodes, retaining parameters,
fixture intent, kinds and `not_run` status. The map has 100 stage-node associations
and 94 unique identities: 93 pytest nodes and the distinct installed smoke.
Every original mapped node remains, with 14 additional atomic durability checks
using J07's legitimate v2 checkpoint adaptation. The temporal oracle is unchanged.

Mapping coverage is separate from schema compatibility. The unchanged runner
rejects eight detailed J10 kinds with `unsupported_evidence_kind`; the two
compound nodes also have different kinds in J29 and J31, triggering the runner's
`conflicting_node_kind` rule. No kind was silently normalized and no runner or
schema was changed to accept them. These are genuine unresolved qualification
contract blockers. The supervisor must review a schema-compatible representation
that retains the fixture distinctions before rerunning the combined candidate.
Never delete a required stage to match a smaller denominator. This gate implements
no implicit feature exclusions:
any proposed exclusion needs a separate reviewed scope record and a reviewed gate
change that preserves the original denominator and states the exclusion.

The report directory contains:

| Artifact | Meaning |
| --- | --- |
| `report.json` | `jitmind.qualification/v2`: candidate Git SHA, dirty flag, complete source snapshot digest/roster, mode, host facts, fixed command records, package hash, findings and artifact manifest. |
| `acceptance-map.json` | Exact mapping used for this run, hashed by the report. |
| `pytest-results.json` | `jitmind.pytest/v2`: selected IDs/count, setup/call/teardown outcomes, per-node source digest/kind, pytest exit, phase counts, session counters and independent collection artifact hash. |
| `expected-collection.json` | `jitmind.collection/v1`: candidate identity, every item observed before deselection, final selected IDs, deselected IDs, collection failures/skips and filtering flag. Exclusively written at collection finish, before runtime results. |
| `installed-results.json` | Separate installed-package checks, origin relative to its venv, dependency versions and connection probes, when installation reached this step. |
| `sources/` | Every live entry in the candidate snapshot, including non-code fixtures and configuration. Deleted entries have null digests and no artifact. Peer files are never substituted. |
| `*.log`, `*.whl` | Bounded command logs and the built candidate wheel, if build succeeded. |
| `verdict.json` | Recomputed per-case/stage outcomes, counts, findings and release blockers. |

Missing, failed, skipped, timed-out, incomplete-phase, wrong-source, or unmapped
nodes cannot pass. Empty pytest collection cannot pass. An unavailable collection
has null counts, not fabricated zeros. Known findings survive a checker failing
to run. Truncated logs block success. Unknown performance measurements are null.
All enabled cases must have all required nodes/stages executed and passed.

The v2 schemas deliberately reject old bundles lacking independent collection
or snapshot evidence; regenerate evidence rather than upgrading a claim in place.
`snapshot_files` is a nonempty sorted list of unique `[relative_path, sha256_or_null]`
pairs. Paths must already be normalized POSIX relative paths. The verifier
recomputes `snapshot_sha256` from UTF-8 `json.dumps(roster, separators=(",", ":"))`
(with Python's default ASCII escaping), then binds every live entry to a verified
`sources/<path>` artifact. Extra sources outside the roster are rejected. Every
runtime record's source hash must agree with that roster; mapped source paths
must also agree with the node's file and the reviewed acceptance map.

The plugin observes `pytest_itemcollected`, `pytest_deselected` and
`pytest_collectreport`, preserving collection failures and module-level
`skip`/`importorskip`. `-k` (including `pytest.ini addopts`), `-m`, explicit
ignore/deselect, last-failed selection, collect-only and file/node-targeted
invocations cannot complete the fixed suite. Collection IDs must be unique;
selected/deselected sets must partition the original inventory. Runtime phase
counters and pytest's collected/failed counters are checked against the records.
Deleting an unmapped failing record and shrinking runtime collection cannot
change the independently recorded expected inventory. Missing expected records
remain unknown in the full denominator; `collection_count` reports that expected
count and `selected_count` separately reports executed selection. Unavailable
runtime measurements remain null. A collection artifact missing entirely blocks
verification. These checks detect internal contradictions, not a malicious
author rewriting all artifacts consistently or deliberately deceptive test code.

Reports are strict bounded JSON: duplicate keys, nonfinite numbers (including
exponent overflow), booleans as integers, duplicate cases/tests/artifacts, invalid
SHA formats, missing stages, path traversal/symlink artifacts and modified bytes
are rejected. Externally supplied JUnit/XML is **not imported**. This version
prefers its own pytest plugin records; adding an XML importer would require
separate hardening and tests. No imported artifact triggers network access or
command execution.

Validate an existing bundle against the **trusted map in the current checkout**
and independently selected expected candidate identities:

```sh
"$JITMIND_PYTHON" scripts/qualify_jitmind.py verify /path/to/report \
  --candidate-sha <reviewed-40-hex-git-sha> \
  --snapshot-sha <reviewed-64-hex-snapshot-digest>
```

Input JSON is capped at 8 MiB; each artifact at 128 MiB; source and evidence
rosters at 10,000 files and 512 MiB in aggregate. File hashing/copy loops also
check the overall deadline; blocking host filesystem calls are not hard real-time.
`bounded_copy(source, destination, expected_hash=None, deadline=None)` prechecks
regular-file size, opens without following a symlink, pins inode/device/size and
nanosecond modification/change times, and streams unbuffered reads of at most the
remaining limit plus one byte. Oversize/growth, replacement, metadata changes or
an expected-hash mismatch fail and remove partial output. Destinations are
exclusive. Snapshot/build/evidence copies and wheel acquisition all use this
helper; installation uses the copied, hashed wheel. This is a stable-file check,
not a filesystem snapshot or protection against a malicious privileged writer.

The API `evaluate(trusted_mapping, report, artifact_root)` likewise expects its
first argument from the reviewer, not an untrusted report. Hashes establish byte
consistency, not authorship. A malicious party can manufacture mutually consistent
JSON and hashes; independent execution and authenticated review remain necessary.
Imported claims of release approval are ignored. Keep raw logs private: they may
contain trusted test diagnostics or local paths. Public errors are fixed codes;
unknown imported findings are retained as a sanitized finding marker. Redacted
sharing copies are new artifacts, not silently edited originals.

## Runtime bounds and operational limits

Each command has both a relative timeout and the shared absolute suite deadline.
Stdout/stderr are drained concurrently before either can fill a pipe; logs are
capped during capture (2 MiB per command), not sliced after an unbounded capture.
Overflow fails the command. Cancellation/timeout kills the owned POSIX process
group and reaps the direct child, including when a child inherited pipes after
the leader exited. Both pipes are closed on every cleanup path. `run_command`
returns `status: cancelled` for SIGINT/KeyboardInterrupt and SIGTERM. During a
main-thread call, a scoped SIGTERM handler sets the cancellation event and the
previous handler is restored afterward. A caller in another thread never changes
process-global signal handlers: its owner must forward cancellation through the
explicit `threading.Event`. There is no process-wide handler installed on import.
An active command wrapper can therefore return normally with a cancelled result;
the CLI retains incomplete/cancelled evidence rather than claiming success.
SIGKILL cannot be trapped, and interpreter/host death cannot promise cleanup.
There is also no scoped signal handler between commands. A supervisor requiring
whole-lifetime containment must own that process lifecycle. The OS reaps orphan grandchildren. Commands with unknown
external effects are never retried. A process deliberately escaping its group
is outside this trusted-command containment model.

The process runner and new legacy locks are POSIX-only. Windows is explicitly
unsupported, not validated production Windows. Existing core Ubuntu Python
3.10/3.11/3.12 CI remains unchanged; no Windows skip matrix or unverified platform
job was introduced. macOS integration/platform matrix signoff remains separate.

Optional parser qualification requires Node >=20 and exactly tree-sitter 0.21.1 /
tree-sitter-python 0.21.0 in the reviewed extracted `adapters/graft` runtime. Use
`npm ci --ignore-scripts --no-audit --no-fund` only as a separate reviewed runtime
provisioning step; never install/run the whole donor package's lifecycle hooks.
Pass `--node-binary /absolute/path/to/node` to the runner on systems where it is
outside the allowlisted PATH. That runtime is supplied only to checkout parser
tests, never to the standalone smoke. A native parser probe must actually run;
installation success alone is insufficient. Python AST is the first-release
subset; other languages and whole-donor parity are not claimed.

SQLite 3.51.2 in the supplied environment has the noted WAL-reset issue. The new
local stores use DELETE/FULL; WAL requires a patched version accepted by the
storage component. Do not globally enable WAL or infer safe settings from the
qualification runner's unrelated stdlib connection. Review actual application
`diagnostics()` and failure/recovery tests for each candidate.

| Component disabled/unavailable | Required behavior and limit |
| --- | --- |
| Durable authority | Legacy format remains available by explicit configuration; arbitrary legacy memory/page pairs do not become transactional. |
| Structural adapter | Ordinary authorized memory continues independently; code evidence reports unavailable, with no other-product or remote fallback. |
| Code bindings | Preserve historical facts; unknown/stale source does not expire a fact or manufacture a current binding. |
| Preflight | Explicit no-match/deferred behavior; obligations remain open; imported law labels confer no trust. |
| Checkpoint ledger | Legacy state remains unvalidated; iteration lists do not become closed event ranges. |
| Providers/optional retrievers | No live-provider qualification in this gate; fake-generator success is only local public-API behavior. |

## Cutover and rollback drill

Perform this against disposable copies before any real cutover. The supervisor
records actual commands, candidate hashes, settings, data digests and outcomes in
a new evidence bundle; these instructions themselves are not a successful drill.

1. Inventory source formats, exact namespace/repository identities, outstanding
   operation keys and writer processes. Stop **all** old writers and confirm
   quiescence. J01's persistent flock strategy cannot mix with old O_EXCL writers;
   never unlink a live lock file or steal ownership based on age.
2. Preserve untouched legacy files and a manifest of their hashes. Corrupt or
   ambiguous input is a blocking finding, not an empty successful migration.
   Use J02 `stage_legacy(source, quiesced=True)` and explicitly import into an empty
   namespace of a new SQLite authority. Quiescence is an operator responsibility,
   not a property proven by that boolean.
3. Compare stable fact IDs, page aliases/provenance, temporal/version histories,
   namespace visibility, queries, receipts and outbox state. Record the explicit
   old-to-new ID mapping; never infer page aliases from basenames/list positions.
   Refuse unknown custom stores as transactional backends.
4. Exercise writes before COMMIT and after COMMIT/before acknowledgement with
   actual tests. Reconcile uncertain acknowledgements using the same operation
   key and payload. Do not allocate a new key or repeat a provider/tool effect
   merely because the acknowledgement was lost.
5. Cut over one quiesced configuration to the new authority; do not dual write.
   Exercise public ingest/research and scoped current-source filtering. Deliver
   duplicate/out-of-order projection events; verify primary durability, current
   authority filtering, and visible lag. External consumers need their own
   delivery ledger, not the built-in reference projection's delivered bit.
6. Commit a distinct **post-cutover write**. Stop candidate writers, take a fresh
   SQLite API backup including receipts/aliases/history/outbox, and restore it
   with `SQLiteDurableStore.restore(backup, new_destination)`. Never copy only a
   live main database file or overwrite the current authority with an old backup.
7. Verify that the restored candidate-era write and its receipt survive, replay
   the same key without regeneration, and compare queries/aliases/counts. Restart
   bindings/projections explicitly: code snapshots are process-local and require
   rebuilding; retained metadata is not proof of current code validation.
8. A rollback to **old application code** is not equivalent to restoring a new
   SQLite backup. Preserve the latest candidate authority, reconcile every
   candidate-era write, and require a reviewed export/mapping compatible with the
   old format before routing traffic. If none exists, keep writers stopped and
   choose a forward repair or compatible candidate restore. Never silently lose
   post-cutover writes by switching back to the pre-cutover files.
9. Checkpoint v2 hashed IDs also need explicit mapping for old consumers. Back up
   exact stored IDs, stop old/new writers, and do not guess sanitized aliases.
   Closed-range replay requires the original events/source digest; summaries do
   not replace missing events. Replay is not restoration of external side effects.
10. Record remaining limits: single-host local filesystem only; injected faults
    and process kills are not physical power-loss tests; historical data/backups
    may remain after logical forget; grants must be reissued after restart.

## Release decision and performance claims

`checks_only_complete` and `release_qualified` are separate. Even a fully passing
checks bundle retains independent signoff, platform matrix, provider evaluation,
power-loss drill and benchmark-comparison blockers. This tool has no flag that
silently converts these unrun dimensions into approval. A release owner must
review and scope them explicitly in a separate authenticated decision record.

No benchmark runner or donor comparison is claimed here. `benchmark_measurement`
is null. Future measurements must freeze datasets, source hashes, configs,
provider/token budgets and the complete unique baseline/candidate grid. Preserve
failed/inconclusive cells and raw bounded logs. Label synthetic workloads; never
convert unauthorized reads, lost writes, or poor anchors into a favorable speed
score. Retrieval correctness, durability and anchor quality stay separate from
performance, and unrun comparisons remain `not_run`.

## Final mapping review and staged execution

The earlier worker and remediation observations below are historical; they are
not final assembled-candidate results. The final mapping review closed the eight
source-mapping gaps without removing cases, required stages or existing nodes:

| Case/stage | Exact added J10 node associations |
| --- | ---: |
| J02 / J-01: real legacy publication ENOSPC | 5 |
| J02 / J-02: native SQLite capacity exhaustion | 1 |
| J04 / J-02: real process-exit boundaries | 5 |
| J16 / J-06: parser, bindings and open-work fallback | 2 |
| J22 / J-07: tool result/cancel/timeout boundaries | 3 |
| J25 / J-02: corrupt migration, faults and process exits | 8 |
| J29 / J-04: parser/binding/projection replay composition | 2 |
| J31 / J-04: forget, retained metadata and open obligation | 2 |

These are 28 associations for 26 unique nodes, including 10 J10-authored edge
cases and 16 reused owner cases. J08 adapted only the compound fixture's explicit
`DurableBindingAuthority` configuration; all forget, obligation and authorized
raw-source assertions remain. Memory forget hides derived bodies and leaves the
obligation open; authorized source files remain readable. Physical erasure and
backup purge are not asserted.

The frozen composite used J01 at `b3fe1d51a4cb98ea2ddce7e94fd3f732010b623d`,
corrected `delivery/J02` through `delivery/J05`, and final `worktrees/J06`,
`worktrees/J07`, `worktrees/J09`, relative to the orchestration root. Only owned
files were overlaid; stale dependency copies were excluded. The provisioned local
Node/tree-sitter adapter came from `delivery/J04/adapters/graft`.

Actual full staged execution: **861 collected; 860 passed, 1 failed, 0 skipped**.
The sole failure is the unchanged qualification test rejecting the supplement's
preserved evidence kinds. All **93 mapped pytest identities** passed all three
phases, including **26/26 supplement nodes**. Independent collection recorded no
filtering, deselection or collection errors/skips. These raw diagnostic records
are not an accepted qualification bundle; their kinds remain `unmapped` because
the preserved kind conflicts cannot be certified by this runner. No passing
case/stage status or release approval was synthesized from the raw outcomes.

Separate diagnostic reproductions demonstrate the compound kind conflict and
the unchanged smoke reason-code mismatch. At that diagnostic stage, J32 installed execution was
unrun and no full gate result was credited. See [the contract alignment record](engineering/qualification-review.md)
for exact commands, source ownership, copied-test inventory and remaining limits.

## Historical J08 remediation verification record

The initial regression run reproduced 11 failures and one already-passing SIGINT
case. Final verification: **96 focused helper tests passed** and **134 full checkout
tests passed**, each with three expected optional-provider warnings. Compile and
Ruff E9/F passed; mypy is unavailable. Exact commands and controlled probe results
are in the private historical remediation record. Prior historical counts were 59 helper tests and 97 full checkout
tests; their missing-setuptools build failure is historical and no longer an
environment blocker. No installation, live provider call, peer write, commit or
publication is part of this remediation. Release requirements remain unchanged.


## Final delivery contract corrections

The final delivery normalizes supplement labels to the existing fixed evidence
vocabulary and binds the missing-runtime smoke to the implemented typed reason.
No case, stage, assertion or release blocker was removed. See
[qualification-review.md](engineering/qualification-review.md) and the adjacent
normalization ledger. Re-run against the actual committed candidate; historical
worker counts are not the final candidate result.
