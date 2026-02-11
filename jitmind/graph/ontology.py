from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set


@dataclass
class GraphOntology:
    entity_types: Set[str] = field(default_factory=set)
    relation_types: Set[str] = field(default_factory=set)
    allow_unknown: bool = True
    default_entity_type: str = "Entity"
    default_relation_type: str = "RELATED_TO"

    def normalize_entity_type(self, etype: str | None) -> str:
        t = (etype or self.default_entity_type).strip()
        if not t:
            t = self.default_entity_type
        if self.entity_types and t not in self.entity_types:
            return t if self.allow_unknown else self.default_entity_type
        return t

    def normalize_relation_type(self, rel: str | None) -> str:
        r = (rel or self.default_relation_type).strip().replace(" ", "_").upper()
        if not r:
            r = self.default_relation_type
        if self.relation_types and r not in self.relation_types:
            return r if self.allow_unknown else self.default_relation_type
        return r

    @classmethod
    def from_lists(
        cls,
        entity_types: Iterable[str] | None = None,
        relation_types: Iterable[str] | None = None,
        allow_unknown: bool = True,
    ) -> "GraphOntology":
        return cls(
            entity_types=set(e for e in (entity_types or []) if e),
            relation_types=set(r for r in (relation_types or []) if r),
            allow_unknown=allow_unknown,
        )
