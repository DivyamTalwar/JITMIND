# J10 snapshot acceptance evidence — historical

J08 final integration note: the outcomes below describe the original J10 snapshot.
The copied fixture now configures J06's explicit `DurableBindingAuthority`; current
node/fixture/module hashes in the supplement describe final staged owner sources.
The supplement's `inputs` and `all_copied_dependencies` still preserve historical
J10 provenance, not the final composite roster. J08's final full run collected
861 tests: 860 passed, one unchanged gate-schema test failed, zero skipped.
All 26 supplement nodes and all 93 mapped pytest nodes passed every phase.
See [qualification-review.md](qualification-review.md) for current blockers and ownership.
No source specification status has been converted into an execution result.

This is internal integration evidence for the supervisor's J08 qualification
assembly, not a product feature, separate PR, or final qualification result.
All statuses in `acceptance-supplement.json` intentionally remain `not_run`.
The outcomes below describe the attested J10 snapshot only.

Only three files are J10 contributions: `tests/test_acceptance_edges.py`,
`docs/engineering/acceptance-supplement.json`, and this note. Product modules and
existing tests were read-only. No copies were refreshed, dependencies installed,
services/providers contacted, commits created, or peer files edited.

| Stage | Evidence | Final exact-node count |
| --- | --- | ---: |
| J02 / J-01 | New ENOSPC at actual legacy publication, five stores, positive recovery | 5 |
| J02 / J-02 | New native SQLITE_FULL, complete rollback, same-key positive recovery | 1 |
| J04 / J-02 | Reused real process exits before/after commit | 5 |
| J16 / J-06 | Reused actual parser → binding → move/disappearance → open-work fallback | 2 |
| J22 / J-07 | Reused end/start tool cuts and complete-range positive for three terminals | 3 |
| J25 / J-02 | Three reused corrupt sources, three reused import failures, two new process exits | 8 |
| J29 / J-04 | New full parser/binding/lesson/preflight/projection chain, two delivery schedules | 2 |
| J31 / J-04 | Same two compound nodes, primary forget with delayed projections and open task | 2 |

The table has 28 stage-node associations but **26 unique pytest nodes: 10 new,
16 reused**. Every reused definition was inspected and executed. Exact collected
parameter IDs, including escaped newlines in the existing J16 parameters, are
stored in `collection.exact_node_ids`. No static mapping is counted as a run.

Final verification: **26 passed, 3 expected optional-provider warnings, 3.56 s**.
Runtime: Python 3.12.12, SQLite 3.51.2, Node v25.2.1; the real adapter uses its
provisioned tree-sitter 0.21.1 / tree-sitter-python 0.21.0 dependencies.
The final own-test run separately passed **10/10**; the reused-node run passed
**16/16**. Ruff check and format check passed. Python compilation and Python 3.10
syntax parsing passed. JSON/source validation checked all eight stages, every
mapped source/module digest, and all 45 unchanged snapshot dependencies.
`git diff --check` passed. Neither mypy nor pyright is installed; no static type
check is claimed and no installation was attempted.

Final execution command from J10 (the variables expand to the prescribed paths):

```sh
ROOT=/path/to/orchestration-root
PYTHONPATH="$PWD" \
JITMIND_TEST_GRAFT_ADAPTER="$ROOT/worktrees/J04/adapters/graft" \
"$ROOT/venv/bin/python" - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
spec = json.loads(Path('docs/engineering/acceptance-supplement.json').read_text())
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q', *spec['collection']['exact_node_ids']],
    env=os.environ, timeout=120,
)
raise SystemExit(result.returncode)
PY
```

Collection used the same isolated Python with `-m pytest --collect-only -q` and
the exact selectors in `collection.selectors`: **26 collected**. Lint commands
were `"$ROOT/venv/bin/python" -m ruff check tests/test_acceptance_edges.py` and
`"$ROOT/venv/bin/python" -m ruff format --check tests/test_acceptance_edges.py`.
New spawned children have 20-second joins followed by bounded terminate/kill
and reap; the parser uses its actual bounded process wrapper. Only sanitized
ENOSPC is injected; SQLite capacity and process termination are real.

The supplement's `all_copied_dependencies` is the complete inventory of **45
pre-existing copied files**, with owner, initial/current SHA-256, and explicit
non-contribution markers. Every local digest still matches
`ROOT/reports/J10-dependency-snapshot.json`. The J04 adapter is read directly
from its owner and is not copied. Its source/lockfile digests are also recorded.
Tests use `JITMIND_TEST_GRAFT_ADAPTER`, falling back to repository
`adapters/graft`; no test depends on a home path or review-probe script.
Inventory counts: J02 8, J03 7, J04 8, J05 6, J06 7, J07 7, and J09 2.

At source review, nine owner files had advanced beyond this local snapshot:
J02 `storage/{migration,sqlite}.py` and `test_durable_remediation.py`;
J05 `code_memory/{anchors,binding_storage}.py` and `test_code_anchors.py`;
J06 `code_memory/{lesson_models,preflight}.py` and `test_open_work_bindings.py`.
Their observed owner digests are recorded without copying in-progress changes.
Final assembly must retest its actual versions. In particular, newer J06 work
introduces a binding-authority adapter; its final configuration/schema compatibility
is outside this snapshot run and must be composed explicitly, not bypassed.

The compound fixture first proves that the memory-only sentinel appears in the
authoritative fact and delivered lesson. It never appears in repository source.
One schedule starts with an existing primary projection; the other keeps the
original add pending until after the delete event is delivered. Both verify
metadata replay, lesson version ordering, source freshness/generation, primary
forget before deletion projection, empty current derived memory outputs,
content-free binding/work/delivery histories, and an obligation still open at
revision 1. **Raw authorized code remains readable after memory forget.** J04
indexes source bytes, not fact bodies; deleting a memory does not delete the
user's files. Retained primary administrative history and backups can still
contain bodies; no physical erasure or all-history purge is claimed.

No product defect blocked these 26 snapshot cases. Remaining limits are final
owner-version assembly, cross-platform/power-loss coverage, and the per-node
limits recorded in the supplement. The frozen temporal oracle was read but this
supplement adds no temporal acceptance claims. Reusing 16 exact existing nodes
avoids copying tests or introducing product abstractions.
