# Memory namespaces

JITMIND treats a namespace as an ordered tuple of opaque components. The
default namespace is `("default",)`, while passing `user_id="alice"` derives
`("users", "alice")`. Components are never joined into a delimiter-bearing
string, so prefix checks cannot confuse `("team", "a.b")` with
`("team", "a", "b")`.

```python
memory_agent.memorize(
    "Alice prefers concise status updates",
    namespace=("workspace", "acme", "user", "alice"),
)

output = research_agent.research(
    "How should I format the update?",
    namespace=("workspace", "acme", "user", "alice"),
)
```

The namespace is persisted on both the `MemoryEntry` and its source `Page`.
Mutation prompts, version updates, valid-time queries, ranking context, and
retrieval hits are scoped before they can reach an LLM. A cross-namespace
update fails closed instead of creating a replacement in the wrong tenant.

This design adapts the isolation pattern visible in Mem0's
`user_id`/`agent_id`/`run_id` API and the component namespaces used by
LangGraph stores. The implementation is original and intentionally retains
component boundaries rather than copying either project's storage model.

References:

- <https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py>
- <https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint/langgraph/store>

