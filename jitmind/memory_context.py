"""Bounded lexical candidate selection for memory mutation prompts."""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence, TypeVar

from jitmind.schemas.advanced_memory import MemoryEntry

T = TypeVar("T")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class ContextSelectionStats:
    total: int
    candidates: int
    selected: int
    rebuilt: bool


class _IncrementalTokenIndex:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.tokens: list[frozenset[str]] = []
        self.postings: dict[str, set[int]] = defaultdict(set)

    def sync(self, keys: Sequence[str], texts: Sequence[str]) -> bool:
        sample = min(4, len(self.keys))
        prefix_matches = len(keys) >= len(self.keys)
        if prefix_matches and sample:
            prefix_matches = (
                list(keys[:sample]) == self.keys[:sample]
                and list(keys[len(self.keys) - sample : len(self.keys)])
                == self.keys[-sample:]
            )
        rebuilt = not prefix_matches
        if rebuilt:
            self.keys.clear()
            self.tokens.clear()
            self.postings.clear()
        start = len(self.keys)
        for index in range(start, len(keys)):
            item_tokens = _tokens(texts[index])
            self.keys.append(keys[index])
            self.tokens.append(item_tokens)
            for token in item_tokens:
                self.postings[token].add(index)
        return rebuilt


class MemoryContextSelector:
    """Select relevant plus recent memories while enforcing a hard item limit."""

    def __init__(self, limit: int = 32, recent_candidates: int = 64) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.recent_candidates = max(limit, recent_candidates)
        self._abstract_index = _IncrementalTokenIndex()
        self._entry_index = _IncrementalTokenIndex()
        self.last_stats = ContextSelectionStats(0, 0, 0, False)

    def _select(
        self,
        query: str,
        values: Sequence[T],
        *,
        keys: Sequence[str],
        texts: Sequence[str],
        index: _IncrementalTokenIndex,
        limit: int | None,
    ) -> list[T]:
        cap = min(limit or self.limit, len(values))
        if cap <= 0:
            self.last_stats = ContextSelectionStats(len(values), 0, 0, False)
            return []
        rebuilt = index.sync(keys, texts)
        query_tokens = _tokens(query)
        candidates: set[int] = set(range(max(0, len(values) - self.recent_candidates), len(values)))
        for token in query_tokens:
            candidates.update(index.postings.get(token, ()))

        denominator = max(1, len(values) - 1)

        def score(position: int) -> tuple[float, int]:
            overlap = len(query_tokens.intersection(index.tokens[position]))
            lexical = overlap / max(1, len(query_tokens))
            recency = position / denominator
            return lexical * 10.0 + recency, position

        selected_indices = heapq.nlargest(cap, candidates, key=score)
        selected_indices.sort()
        self.last_stats = ContextSelectionStats(
            total=len(values),
            candidates=len(candidates),
            selected=len(selected_indices),
            rebuilt=rebuilt,
        )
        return [values[index] for index in selected_indices]

    def select_abstracts(
        self, query: str, abstracts: Sequence[str], *, limit: int | None = None
    ) -> list[str]:
        return self._select(
            query,
            abstracts,
            keys=abstracts,
            texts=abstracts,
            index=self._abstract_index,
            limit=limit,
        )

    def select_entries(
        self,
        query: str,
        entries: Sequence[MemoryEntry],
        *,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        return self._select(
            query,
            entries,
            keys=[entry.id for entry in entries],
            texts=[entry.content for entry in entries],
            index=self._entry_index,
            limit=limit,
        )


__all__ = ["ContextSelectionStats", "MemoryContextSelector"]
