"""Benchmark bounded context selection over a 100,000-memory collection."""

from __future__ import annotations

from statistics import median
from time import perf_counter

from jitmind.memory_context import MemoryContextSelector


def main() -> None:
    memories = [f"routine project note number {index}" for index in range(100_000)]
    memories[123] = "customer prefers cobalt blue dashboards"
    selector = MemoryContextSelector(limit=32)
    started = perf_counter()
    first = selector.select_abstracts("cobalt dashboard", memories)
    build_ms = (perf_counter() - started) * 1000
    timings = []
    for _ in range(100):
        started = perf_counter()
        selected = selector.select_abstracts("cobalt dashboard", memories)
        timings.append((perf_counter() - started) * 1000)
    assert memories[123] in first and len(selected) == 32
    print(
        {
            "memories": len(memories),
            "selected": len(selected),
            "index_build_ms": round(build_ms, 2),
            "warm_median_ms": round(median(timings), 2),
        }
    )


if __name__ == "__main__":
    main()
