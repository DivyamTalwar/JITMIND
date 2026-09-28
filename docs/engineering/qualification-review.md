# Final qualification contract alignment

This is a source-review record, not a passing execution report. The qualification
runner records actual results separately against a committed candidate and wheel.

## Preserved requirements

The map retains 32 cases, all 41 required stages, 100 stage-node associations and
94 unique identities (93 pytest nodes and one separately executed installed smoke).
Case text, node identities, parameter variants, fixture assertions and original
handoff/oracle hashes are unchanged. Source-controlled statuses remain `not_run`.

## Metadata correction

The J10 supplement used descriptive evidence labels outside the runner's existing
fixed vocabulary. `evidence-kind-normalization.json` records each translation.
Specific fixture descriptions, assertions and limitations remain with the nodes.
The two reused compound workflow nodes now have one consistent `projection` kind.
No runner allowlist, expected-test denominator or release blocker was weakened.

## Capability contract correction

The missing-node smoke now requires the actual typed `Unavailable` exception with
`node_executable_unavailable`. New negative controls reject ordinary filesystem
and attribute errors, unrelated typed failures and the obsolete generic code.
The real installed smoke still runs outside the checkout with Node and optional
provider packages absent; checkout helper tests cannot substitute for it.
