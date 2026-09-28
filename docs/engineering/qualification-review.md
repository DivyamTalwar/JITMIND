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


## Native SQLite capacity evidence on Python 3.10

The real max_page_count exhaustion fixture now records numeric SQLite error
metadata when exposed and otherwise requires the exact native OperationalError
message. Both paths still require a real SQLite capacity limit, zero committed
effects, a clean integrity check, and successful same-request retry after raising
the limit. No error injection substitutes for the real engine capacity failure.

## Reviewed CI fixture corrections

The quota fixture seeds 256 receipt rows in one bounded transaction instead of
256 autocommit fsyncs. The binding IMMEDIATE-lock compatibility control uses a
functional deadline but retains its delivered result and 300 ms assertion. The
EXCLUSIVE control retains the production deadline and deferred-result assertion.
No production source or existing assertion was changed. The acceptance map
refreshes only the two affected fixture file identities; all 32 cases, 41 stages,
node identities and not-run specification statuses remain unchanged. New
qualification evidence must be generated against this committed snapshot.
