# Bounded memory mutation context

`MemoryAgent` no longer copies every stored abstract into each mutation prompt.
`MemoryContextSelector` keeps a small append-aware token index, considers exact
lexical matches plus a recent window, and returns at most 32 items by default.

```python
agent = MemoryAgent(generator=generator, context_limit=24)
```

The cap affects prompt construction only. The selected operation is still
applied by exact memory ID against the authoritative store. Selection stats are
available at `agent.context_selector.last_stats` for instrumentation.

Run the 100,000-memory benchmark from a source checkout:

```bash
python benchmarks/benchmark_memory_context.py
```
