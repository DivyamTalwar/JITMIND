# Optional bounded local code context

JITMIND's `jitmind.code_context` API provides explicit local source snapshots and
structural evidence. It requires the host-issued `ScopeContext` and
`ScopeAuthority` from J-03. Ordinary memory use has no Node requirement, and the
base Python dependency manifest is unchanged. This package does not modify
ResearchAgent, PageStore, stored memory formats, or the later code-memory layer.

## Provision the optional adapter

Use a trusted Node executable (Node 20 or newer) and an application-controlled
copy of `adapters/graft`. Provision its exact dependencies independently:

```sh
cd adapters/graft
npm ci --ignore-scripts --no-audit --no-fund
```

This is an administrative installation command, never a code-query operation.
Compatible native prebuilds for tree-sitter must be present; if absent, the
adapter reports unavailable. Do not run the donor package's lifecycle hooks.
The runner needs no API key, network access, target build, LSP, or provider.
Keep this directory and its node_modules outside any untrusted target root and
writable only by the host administrator. The subprocess boundary is resource
containment, not an operating-system sandbox for a replaced trusted runner.

Importing `jitmind.code_context` does not discover or start Node, load the adapter,
or invoke a provider. `NodeParser(node_binary, adapter_dir)` accepts trusted
absolute paths and performs filesystem checks without launching a process.
Catch the exported `CapabilityUnavailable` exception when configuring optional
code context: its `.reason` is `node_executable_unavailable`,
`adapter_directory_unavailable`, or `adapter_runner_unavailable`. Missing paths,
non-executable Node files, and missing/unreadable/symlink runners produce these
fixed diagnostics, rather than leaking an incidental filesystem exception.
Relative paths remain configuration errors (`ValueError`). A successful
constructor does not prove Node version or native dependencies work; the bounded
`build` invocation checks actual startup. Later launch failures return
`runtime_unavailable`; a runner exiting unsuccessfully returns `adapter_failed`.
Neither error causes an install, provider call, or alternate runtime lookup.

Runtime extraction supports **Python `.py` and `.pyi` only**, using tree-sitter
0.21.1 and tree-sitter-python 0.21.0. `adapters/graft/PROVENANCE.md` records donor
SHA, source adaptation, exact dependency closure, exclusions, and MIT notices.
No TypeScript or other language support is claimed. Fixtures exercise the real
extracted parser, including the donor's chunked parsing for files over 32 KiB.

## Runnable fixture

From the repository root, after J-03 is present and the optional runtime is
provisioned, run this with your installed isolated Python and `PYTHONPATH=$PWD`:

```python
from pathlib import Path
from shutil import which
from uuid import uuid4
from jitmind.scope import ScopeAuthority
from jitmind.code_context import (
    CapabilityUnavailable, CodeContext, CodeQuery, NodeParser, RepoRegistry,
)

# Trusted host setup. Never expose grant/register/runtime configuration as query RPCs.
repo_id = str(uuid4())
authority = ScopeAuthority()
authority.grant('local-user', 'private-demo', (repo_id,))
scope = authority.context('local-user', 'private-demo', 'demo-1')
registry = RepoRegistry()
registry.register(repo_id, str(Path('adapters/graft/fixtures').resolve()), 'private-demo')
node_path = which('node')  # resolve trusted application PATH once
if node_path is None:
    raise CapabilityUnavailable('node_executable_unavailable')
node = str(Path(node_path).resolve())
parser = NodeParser(node, str(Path('adapters/graft').resolve()))
context = CodeContext(authority, registry, parser)
built = context.build(scope, repo_id, deadline_ms=10000)
assert built.status == 'ok', built.reason_codes
request = CodeQuery(operation='trace_calls', repo_id=repo_id,
                    snapshot_id=built.snapshot_id, query='alpha',
                    direction='out', depth='all', path_prefix='cycle.py')
page = context.query(scope, request)
print(page.model_dump_json(by_alias=True, indent=2))
```

Build explicitly publishes an immutable captured generation. Query requires its
snapshot ID and never auto-rebuilds. A source edit does not rewrite old evidence:
the response identifies its recorded snapshot and reports current freshness.
Call `build` again when a new generation is wanted. A failed build leaves existing
generations intact. This implementation keeps generations in memory, publishes
under a lock, and retains at most eight snapshots per service instance. Restart
requires rebuilding. No index files are written into the target repository.

## Request and operations

`CodeQuery` accepts the versioned JSON schema `jitmind.code-query/v1` through the
`schema` field. `CodeQuery.from_json(bytes)` also rejects duplicate keys and
nonfinite JSON numbers. Unknown fields are rejected. Required fields are
`operation`, canonical internal UUID `repo_id`, and 64-hex `snapshot_id`.
`query` defaults to empty (required for search/trace), `path_prefix` to empty,
`limit` to 10 (1–50), `max_context_tokens` to 2000 (1–16000), and `deadline_ms`
to 1500 (1–30000). Integer fields reject booleans, floats, and nonfinite values.
`direction` is `in` or `out`; `depth` is a positive integer up to 1024 or `all`.
`fixed` must be true; regex search is deliberately unsupported.

| Operation | Behavior |
|---|---|
| `find_code` | Deterministic term-overlap rank of symbol name/body, after path filtering. |
| `file_api` | Definitions/signatures for exact repo-relative `query` path; no basename fallback. |
| `trace_calls` | BFS over unique same-file bare-function candidate calls, incoming/outgoing; visited sets handle cycles. |
| `find_all` | Fixed-string matching by source line, including text of syntactically invalid UTF-8 source files. Count describes matching lines, not individual occurrences. |
| `repo_map` | Nonempty files ranked by number of parsed definitions, with a first-line source span; empty files have separate path/digest metadata (see below). |
| `check_freshness` | Hashes the current bounded source roster against the old snapshot; returns changed/added/removed/unknown without rebuilding. |

`workspace(scope, list[CodeQuery])` dispatches up to 16 individual repository
requests, preserving every option including prefix, direction, and `depth='all'`.
It prechecks every repository grant and caps the aggregate response at 256 KiB.
Its single absolute deadline is the largest requested deadline, measured from
dispatch setup. Each dispatch checks expiry first, and each query uses the earlier
of that shared deadline and its own requested duration. Sub-millisecond remaining
time is never rounded up into a fresh allowance. Expiry or cancellation stops
further dispatch and returns unavailable pages for the whole workspace; completed
partial pages are discarded. Each returned page retains its own repository/
snapshot identity. Results are not pooled into a global unscoped index.

## Evidence, limits, and incomplete results

`CodeEvidencePage` uses `jitmind.code-evidence/v1`. Results carry exact retained
source line spans (1-based inclusive), the SHA-256 of original UTF-8 file bytes,
namespace/repository/snapshot/source IDs, parser identity, and generation.
Source is sliced from captured bytes, not fetched from the live root or an
unrelated PageStore. CRLF and Unicode survive exactly. Signatures are normalized
parser metadata. Long spans may return only complete leading lines; the retained
span matches the returned text and `truncated` reports the reduction.
Signatures are capped at 2048 Unicode characters; cutting a signature sets
`truncated=true` and `signature_truncated` even when its source span is complete.
This metadata reduction never alters the original source text or file digest.

For `repo_map`, `coverage.empty_files` contains scoped, captured `{path, digest}`
records for zero-byte supported files, and `empty_files_without_source` explains
why they have no `CodeEvidence` result. No line span is invented for an empty
file. The metadata shares the page's snapshot/manifest identity and response-byte
ceiling; it does not consume source/context bytes. `coverage.files` includes
these files, while `repo_map.count` counts nonempty source-bearing candidates
before result/context limits. Empty files alone therefore give `count=0`, empty
results, and explicit metadata, with complete subset coverage and no truncation.
A file containing a newline is nonempty and has the real line-1 span.

The snapshot ID includes repository UUID, namespace, byte-manifest digest, and
parser/coverage digest. Commit SHA is explicitly unavailable: this adapter does
not invoke git or infer commit identity from repository-controlled hooks or
`.git` indirection. Dirty same-HEAD changes therefore still change snapshot IDs.
Identical basenames in separate roots never share repository identity.

Hard ceilings per build: 128 file records, 128 KiB/source, 2 MiB total supported
source bytes, 2048 directory entries, directory depth 32, 2048 total symbols,
1024 symbols/file, and 1024 calls/symbol. `.git`, `.hg`, `.svn`, `node_modules`,
`__pycache__`, `.venv`, and `venv` directories are excluded by the same config in
build and freshness. Other extensions and special files are recorded as
unsupported/excluded; they reduce coverage. Traversal allows at most 1024
visited symbols and 4096 examined call candidates, charging unresolved,
ambiguous, and duplicate calls before resolution. If candidates remain beyond
4096, `edge_budget` marks truncation and suppresses `count`; a complete scan of
exactly 4096 remains within budget. Fixed search caps at 2048
matching lines. Parser input JSON caps at 8 MiB and combined stdout/stderr at
256 KiB while streaming. Node's JS heap is capped at 128 MiB (native allocations
are not covered by this heap limit). The public serialized response, including
diagnostics, also caps at 256 KiB. No unbounded stderr is echoed to the caller.

`source_bytes` is measured UTF-8 source bytes returned. The context budget counts
returned source, names, and signatures using one UTF-8 byte per token as a
conservative upper bound (`context_tokens_upper_bound`), **not** a measured
model-token count. The protocol envelope/IDs are accounted separately by exact
`response_bytes`. Small budgets may yield no source and explicit truncation.
`count` is known only after a complete bounded match scan with complete subset
coverage; it is never a runtime completeness claim. `all` means bounded closure
of the extracted candidate graph, not all possible calls. No zero-result query
proves dead code. Imports, receivers, rebinding, dynamic dispatch, inheritance,
and module-level calls can produce omitted or incorrect candidate edges.

Unreadable files, invalid UTF-8/NUL data, malformed parses, directory enumeration
failures, unsupported languages, and budget exclusions are incomplete/unknown.
An incomplete directory scan never confirms deletion. Unknown results suppress
counts; cancellation, races, malformed/oversized/stuck adapters, or unavailable
runtime produce explicit `status='unavailable'` with fixed `reason_codes` and no
partial success. Coverage describes the stored generation; freshness describes
the current capture. Scope revocation raises generic `ScopeDenied`, even if it
happens while parsing or just before returning cached data.

All target reads use descriptor-relative no-follow opens, regular-file checks,
byte hashes, and before/after metadata checks. Registration anchors the real
root's device/inode; root ancestors are also opened without symlink following.
A second bounded capture before publication catches changes during parsing.
Captured strings are immutable; source/build hooks never execute. As with any
open-handle strategy, an adversarial change immediately after the final check
cannot be prevented; evidence remains tied to captured bytes and is never a
claim of a filesystem-wide transactional snapshot.

The POSIX process wrapper drains both output pipes concurrently with nonblocking
input, checks cancellation and an absolute deadline, kills the owned process
group, and reaps its direct child. The OS reaps orphan grandchildren. Environment
is rebuilt from a tiny allowlist with an empty private HOME/cwd; developer config,
provider keys, NODE_OPTIONS, NODE_PATH, and plugin injection variables are absent.
Windows process containment is not implemented; that runtime is unavailable.

## Local retrieval integration

`CodeContextRetriever(service, scope, repo_id, snapshot_id, path_prefix='')`
provides `search(list[str], top_k=10)`. It returns existing `Hit` objects with
`page_id=None`, `source='code_context'`, and full evidence in
`hit.meta['code_evidence']`. `last_pages` exposes freshness/coverage/unavailable
diagnostics. Consumers must use the evidence source directly. Code identities
are never flattened into numeric/string PageStore page IDs. This is a
standalone optional integration surface, not a ResearchAgent modification.

On optional adapter failure, a host can continue its independently authorized
ordinary memory search and display the diagnostic. It must not broaden scope,
search other roots, or silently replace unavailable code evidence with unscoped
fallback results.

## Verification

```sh
PYTHONPATH=$PWD /path/to/isolated/python -m pytest tests/test_code_context.py tests/test_code_context_process.py -q
PYTHONPATH=$PWD /path/to/isolated/python -m pytest tests -q
PYTHONPATH=$PWD /path/to/isolated/python -m compileall -q jitmind tests
PYTHONPATH=$PWD /path/to/isolated/python -m ruff check jitmind/code_context tests/test_code_context*.py
node --check adapters/graft/extract.mjs
node --check adapters/graft/runner.mjs
```

`test_code_context.py` runs actual filesystem, Python boundary, Node, and donor
parser paths. Forged responses are explicitly injected boundary probes within
that file. Regression fixtures include 6000 unresolved calls, bounded/exact-limit
controls, long Unicode signatures, empty files, and clock-controlled workspace
deadlines (no timing sleeps). `test_code_context_process.py` includes constructor
checks using filesystem fixtures and Python's executable, requiring no Node;
a fresh Python import probe blocks subprocess/network activity and has no PATH.
It separately runs hostile Python child
processes to exercise pipe pressure, scrubbed environments, deadlines,
cancellation, nonzero exit, and descendant cleanup; these are not labeled donor
parser tests. No live provider or target-source execution is involved.
