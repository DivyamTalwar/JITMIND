from __future__ import annotations

import os
from typing import Optional

from .ontology import GraphOntology


def load_ontology_from_env() -> Optional[GraphOntology]:
    entities = os.getenv("GRAPH_ENTITY_TYPES")
    relations = os.getenv("GRAPH_RELATION_TYPES")
    allow_unknown = os.getenv("GRAPH_ALLOW_UNKNOWN", "true").lower() in ("1", "true", "yes")
    if not entities and not relations:
        return None
    entity_types = [e.strip() for e in (entities or "").split(",") if e.strip()]
    relation_types = [r.strip().upper().replace(" ", "_") for r in (relations or "").split(",") if r.strip()]
    return GraphOntology.from_lists(entity_types=entity_types, relation_types=relation_types, allow_unknown=allow_unknown)
