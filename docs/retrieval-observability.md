# Retrieval observability

`ResearchAgent` now produces one structured trace per successful research run.
The trace contains per-channel latency and candidate counts plus a fusion stage
with channel names, deduplication counts, temporal rejections, and score-only
metadata for the top evidence.

Query content is not stored by default. Instead JITMIND records a SHA-256 digest
and character count, which supports correlation without silently creating a
second sensitive-data store. Set `trace_capture_content=True` only after an
explicit privacy review.

```python
from jitmind import JsonlTraceSink, ResearchAgent

agent = ResearchAgent(
    # existing arguments...
    trace_sink=JsonlTraceSink("./traces/retrieval.jsonl"),
)
output = agent.research("What changed?", user_id="alice")
trace = agent.explain_last_retrieval()
```

The sink is optional and non-blocking: an export failure never turns a valid
research answer into an application failure. `InMemoryTraceSink` is available
for tests and custom adapters can implement the one-method `TraceSink` protocol.

The shape is an original, dependency-free JITMIND design informed by Arize
Phoenix's public focus on OpenTelemetry tracing and retrieval evaluation. It is
deliberately a small local contract rather than a vendored tracing SDK; teams
can translate it to OpenTelemetry or another backend at the boundary.

Reference: <https://github.com/Arize-ai/phoenix>

