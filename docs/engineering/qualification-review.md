# Final qualification contract alignment

This is a source-review record, not a passing execution report. The qualification
runner records actual results separately against a committed candidate and wheel.

## Preserved requirements

The map retains 32 cases, all 41 required stages, 103 stage-node associations and
97 unique identities (96 pytest nodes and one separately executed installed smoke).
Original case text, node identities, parameter variants and handoff/oracle hashes
remain. The timing-oracle corrections and three added boundary nodes are recorded
below; they are not described as unchanged assertions. Source-controlled statuses
remain `not_run`.

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

## Explicit primary-read deadline oracle correction

The facts/IMMEDIATE positive lock control now tests delivered content with a
bounded functional budget rather than claiming successful fsync delivery inside
150 ms on shared CI. The three incompatible-lock controls still use the original
production policy and 150 ms ceiling, with empty deferred payloads. Three new
real-storage cases verify before/at/after the default deadline using a controlled
monotonic clock. All original node identities and stages remain, and J20 now
requires the three additional boundary identities. The resulting inventory is
103 stage-node associations and 97 unique identities (96 pytest plus installed
smoke). This is a reviewed test-oracle correction, not unchanged assertions or a
production latency benchmark. Production code is unchanged.
