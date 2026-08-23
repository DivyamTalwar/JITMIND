# Bi-temporal queries and version history

`AdvancedMemoryStore` retains inactive versions by default and supports two
distinct historical questions:

```python
# What is now known to have been valid on January 15?
facts = store.query_as_of("2026-01-15T00:00:00Z")

# What did the store believe on January 15 about that valid date?
facts_then = store.query_as_of(
    "2026-01-15T00:00:00Z",
    transaction_at="2026-01-15T00:00:00Z",
)

history = store.get_version_history(memory_id)
change = store.diff_as_of("2026-01-15T00:00:00Z", "2026-02-15T00:00:00Z")
```

Valid intervals are start-inclusive and end-exclusive. The valid start falls
back from `t_valid` to `t_observed` to `t_created`. Transaction-time filtering
uses `t_created` and `t_expired`.

Set `retain_history=False` only when storage minimization is more important
than audit and time-travel behavior.
