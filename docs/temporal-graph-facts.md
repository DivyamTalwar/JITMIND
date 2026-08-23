# Temporal graph fact search

JITMIND stores each extracted relation as an immutable observation keyed by
its source memory and triplet. Re-ingesting the same observation is idempotent;
observing the same triplet from a later memory creates another version instead
of overwriting history.

```python
facts = graph_store.query_facts(
    ["Alice", "Acme"],
    relation_types=["WORKS_AT"],
    valid_at="2026-02-03T00:00:00Z",
    observed_at="2026-02-04T00:00:00Z",
    namespace=("tenant", "red"),
)

hits = graph_retriever.search_temporal(
    ["Alice"],
    valid_at="2026-02-03T00:00:00Z",
    relation_types=["WORKS_AT"],
)
```

`valid_at` gates event time and uses a start-inclusive, end-exclusive
interval. `observed_at` gates ingestion time. Each result carries its source
memory, page, and full interval so an answer can cite why the fact was visible.

The interface is an original JITMIND implementation informed by Graphiti's
public model of episodes plus facts with `valid_at` / `invalid_at`, including
its fact search filters. No Graphiti code is incorporated.

Reference:

- <https://github.com/getzep/graphiti>
- <https://github.com/getzep/graphiti/blob/main/mcp_server/src/graphiti_mcp_server.py>

