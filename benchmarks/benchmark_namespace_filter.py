"""Microbenchmark the dependency-free namespace filter path."""

from __future__ import annotations

from time import perf_counter

from jitmind.schemas import AdvancedMemoryState, AdvancedMemoryStore, MemoryEntry


def main(total: int = 10_000, tenants: int = 100) -> None:
    store = AdvancedMemoryStore(enable_auto_cleanup=False)
    store.save(
        AdvancedMemoryState(
            entries=[
            MemoryEntry(
                content=f"memory {index}",
                namespace=("tenant", str(index % tenants)),
            )
                for index in range(total)
            ]
        )
    )
    started = perf_counter()
    selected = store.get_entries(namespace=("tenant", "42"))
    elapsed_ms = (perf_counter() - started) * 1_000
    print(
        f"entries={total} tenants={tenants} selected={len(selected)} "
        f"elapsed_ms={elapsed_ms:.3f}"
    )


if __name__ == "__main__":
    main()
