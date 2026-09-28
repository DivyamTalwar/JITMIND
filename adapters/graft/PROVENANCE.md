# Graft extraction provenance and boundary

Donor source: read-only clone pinned to `80692e5ad0bc8e8f7e1edea648247d90b9f76820`.
Its package manifest identifies `@nanonets/graft` **0.20.0**. The manifest's
repository URL is `NanoNets/context-graph-engine`; these names are not treated as
proof of an equivalent published npm release. No donor package was installed.

`extract.mjs` is a narrowed JavaScript adaptation of these Python branches in
`src/graph/extract.ts`: `PY_KINDS`, `PARSE_CHUNK`/`parseSource`, the named-child
`walk`, `describe` (definition name, method promotion, body/header boundary),
and the identifier branch of `calleeName`. It uses the donor's actual
`tree-sitter` 0.21.1 and `tree-sitter-python` 0.21.0 parser packages. It is not a
regex approximation or a native Python AST replacement. Definitions are scoped
by nested names and retain 1-based line spans; source text stays verbatim.
Duplicate definition identities use document ordinals in the host rather than
the donor's `mintId` suffixes. Malformed ASTs are explicitly incomplete.

The selected runtime closure is:

- `runner.mjs` → `extract.mjs`, `node:crypto`.
- `extract.mjs` → `tree-sitter@0.21.1`, `tree-sitter-python@0.21.0`.
- Both native parser packages → `node-gyp-build@4.8.4` → their packaged platform
  native prebuilds. No install/build command is called at query time.
- Installed header dependencies: `node-addon-api@8.9.2` and nested `@7.1.1`.
  Exact tarball integrity and dependency edges are in `package-lock.json`.

No other donor implementation module is imported. Removing non-Python branches,
binding inference, imported-symbol resolution, normalized body caches, and graph
node emission makes this extraction closed without `bindings.ts`, `types.ts`,
`util/id.ts`, or additional language grammars. The Python wrapper owns exact-byte
hashes, immutable snapshots, IDs, ranking, scope, and bounded candidate traversal.
It resolves only unique same-file bare function names; this remains a syntactic
candidate relation because local rebinding/shadowing/dynamic dispatch can differ.
Member calls, cross-file imports, inheritance, references, LSP resolution, and
module-level calls are not claimed.

Audit roots read: `src/context/check.ts`, `src/mcp/tools.ts`, and
`src/graph/extract.ts`. Engine, ask, graph type/resolve/traverse, source decoding,
ID helpers, and binding imports were inspected to establish the boundary.
`donor-audit.json` records the actual 67-file static local import/export closure
of those roots, including source hashes and import edges. Type-only edges are
included; dynamic imports are not asserted by that static audit. It explains
why importing the donor engine/MCP layer would exceed this adapter's scope.
The selected runtime closure above is much smaller than that audit closure.

Explicit deviations: no donor onboarding/CLI, telemetry, savings/marketing,
LLM enrichment, providers, LSP servers, source/build hooks, automatic graph
refresh, Git execution, or regex search. Strict UTF-8 replaces the donor's
permissive decoding. Unknown encodings/unreadable files are **unknown**, never
silently reported as removed. Every operation (including workspace dispatch)
retains scope/depth options. Freshness reports the old index without rebuilding.

The donor MIT notice is retained verbatim in `LICENSE`. Dependency notices are
copied verbatim into `THIRD_PARTY_NOTICES.txt`. Runtime dependencies are optional
and installed independently; the base Python dependency manifest is unchanged.
