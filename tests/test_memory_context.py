from jitmind.memory_context import MemoryContextSelector
from jitmind.schemas import MemoryEntry


def test_selection_is_bounded_and_keeps_old_relevant_memory() -> None:
    abstracts = [f"routine note {index}" for index in range(100)]
    abstracts[2] = "customer prefers cobalt blue dashboards"
    selector = MemoryContextSelector(limit=8, recent_candidates=12)

    selected = selector.select_abstracts("cobalt dashboard preference", abstracts)

    assert len(selected) == 8
    assert abstracts[2] in selected
    assert abstracts[-1] in selected
    assert selector.last_stats.candidates < len(abstracts)


def test_index_extends_without_rebuilding_for_append_only_memory() -> None:
    selector = MemoryContextSelector(limit=2)
    selector.select_abstracts("alpha", ["alpha", "beta"])
    selector.select_abstracts("gamma", ["alpha", "beta", "gamma"])

    assert selector.last_stats.rebuilt is False


def test_entry_selection_preserves_lifecycle_metadata() -> None:
    entries = [
        MemoryEntry(id="old", content="account timezone is UTC"),
        MemoryEntry(id="new", content="unrelated recent note"),
    ]
    selected = MemoryContextSelector(limit=1).select_entries("timezone", entries)
    assert [entry.id for entry in selected] == ["old"]
