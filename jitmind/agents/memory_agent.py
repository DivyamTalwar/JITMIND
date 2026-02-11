# memory_agent.py
# -*- coding: utf-8 -*-
"""
MemoryAgent Module

This module defines the MemoryAgent for the JITMind (JITMind) framework.

- Memory is represented as a list[str] of abstracts (no events/tags included).
- MemoryAgent exposes only: memorize(message) -> MemoryUpdate, allowing the agent to store new information.
- Prompts within the module are used as placeholders for future prompt templates or instructions.
"""


from __future__ import annotations

from typing import Dict, Optional, Tuple, Any
import os

from jitmind.prompts import MemoryAgent_PROMPT, MemoryOperation_PROMPT, ConflictCheck_PROMPT
from jitmind.schemas import (
    MemoryState, Page, MemoryUpdate, MemoryStore, PageStore,
    InMemoryMemoryStore, InMemoryPageStore, Retriever,
    AdvancedMemoryStore, MemoryEntry, MEMORY_OP_SCHEMA
)
from jitmind.generator import AbsGenerator
from jitmind.profile import UserProfileAgent
try:
    from jitmind.graph import GraphMemoryStore, load_ontology_from_env
except ImportError:
    GraphMemoryStore = None  # type: ignore
    load_ontology_from_env = None  # type: ignore

class MemoryAgent:
    """
    Public API:
      - memorize(message) -> MemoryUpdate
    Internal only:
      - _decorate(message, memory_state) -> (abstract, header, decorated_new_page)
    Note: memory_state contains ONLY abstracts (list[str]).
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        page_store: Optional[PageStore] = None,
        generator: Optional[AbsGenerator] = None,  # Generator instance is required
        dir_path: Optional[str] = None,  # Filesystem storage path
        system_prompts: Optional[Dict[str, str]] = None,  # system prompts dictionary
        graph_store: Optional[GraphMemoryStore] = None,
        profile_agent: Optional[UserProfileAgent] = None,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for MemoryAgent")
        self.memory_store = memory_store or AdvancedMemoryStore(dir_path=dir_path)
        self.page_store = page_store or InMemoryPageStore(dir_path=dir_path)
        self.generator = generator
        if graph_store is None:
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USERNAME")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
            if neo4j_uri and neo4j_user and neo4j_password:
                try:
                    ontology = load_ontology_from_env() if load_ontology_from_env else None
                    self.graph_store = GraphMemoryStore(
                        uri=neo4j_uri,
                        username=neo4j_user,
                        password=neo4j_password,
                        database=neo4j_db,
                        ontology=ontology,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to init GraphMemoryStore: {e}")
                    self.graph_store = None
            else:
                self.graph_store = None
        else:
            self.graph_store = graph_store
        
        # Initialize system_prompts (default empty strings)
        default_system_prompts = {
            "memory": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            # Merge user prompts with defaults
            self.system_prompts = {**default_system_prompts, **system_prompts}
        self.profile_agent = profile_agent


    # ---- Public ----
    def memorize(self, message: str, meta: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> MemoryUpdate:
        """
        Update long-term memory with a new message and persist a decorated page.
        Steps:
          1) _decorate(...) => abstract, header, decorated_new_page
          2) Merge into MemoryState (append unique abstract)
          3) Write Page into page_store  (page_id left None by default)
        """
        message = message.strip()
        state = self.memory_store.load()

        # (1) Decorate - this generates the abstract and decorated page
        abstract, header, decorated_new_page = self._decorate(message, state)

        # (2) Decide memory operation (self-edit)
        decision = self._decide_operation(abstract, message)

        # (3) Apply memory operation first (uses page_id for provenance)
        page_id = len(self.page_store.load())
        memory_id = self._apply_memory_operation(decision, abstract, page_id)

        # (4) Persist page
        page = Page(
            header=header,
            content=message,
            meta={
                "decorated": decorated_new_page,
                "page_id": page_id,
                "memory_id": memory_id,
                "t_observed": decision.get("t_observed"),
                "t_valid": decision.get("t_valid"),
                "t_invalid": decision.get("t_invalid"),
            },
        )
        if meta:
            # Avoid overwriting reserved keys if provided
            for k, v in meta.items():
                if k not in page.meta:
                    page.meta[k] = v
        if user_id:
            page.meta["user_id"] = user_id
        self.page_store.add(page)

        # (5) Optional user profile update
        if self.profile_agent and user_id:
            try:
                self.profile_agent.update_profile(user_id=user_id, message=message)
            except Exception:
                pass
        
        # (6) Get updated state after adding abstract
        updated_state = self.memory_store.load()

        return MemoryUpdate(new_state=updated_state, new_page=page, debug={"decorated_page": decorated_new_page})


    # ---- Internal----

    def _decorate(self, message: str, memory_state: MemoryState) -> Tuple[str, str, str]:
        """
        Private. Generate abstract for the message and compose: "abstract; header; new_page".
        Returns: (abstract, header, decorated_new_page)
        """
        # Build memory context from all abstracts (concatenate all memories)
        if memory_state.abstracts:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        else:
            memory_context = "No memory currently."
        
        # Generate abstract for the current message using LLM with memory context
        system_prompt = self.system_prompts.get("memory")
        template_prompt = MemoryAgent_PROMPT.format(
            input_message=message,
            memory_context=memory_context
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        
        try:
            response = self.generator.generate_single(prompt=prompt)
            abstract = response.get("text", "").strip()
        except Exception as e:
            print(f"Error generating abstract: {e}")
            abstract = message[:200]
        
        # Create header with the new abstract
        header = f"[ABSTRACT] {abstract}".strip()
        decorated_new_page = f"{header}; {message}"
        return abstract, header, decorated_new_page

    def _build_memory_context(self) -> str:
        if hasattr(self.memory_store, "get_entries"):
            entries = self.memory_store.get_entries(include_inactive=True)
            lines = [f"{e.id} [{e.tier}/{e.status}]: {e.content}" for e in entries]
            return "\n".join(lines) if lines else "No memory currently."
        state = self.memory_store.load()
        if not state.abstracts:
            return "No memory currently."
        return "\n".join([f"Page {i}: {a}" for i, a in enumerate(state.abstracts)])

    def _decide_operation(self, abstract: str, message: str):
        memory_context = self._build_memory_context()
        prompt = MemoryOperation_PROMPT.format(
            memory_context=memory_context,
            new_abstract=abstract,
            new_message=message
        )
        try:
            response = self.generator.generate_single(prompt=prompt, schema=MEMORY_OP_SCHEMA)
            data = response.get("json")
            if not data:
                text = response.get("text", "")
                try:
                    import json
                    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
                except Exception:
                    data = {}
        except Exception as e:
            print(f"Error in memory operation decision: {e}")
            data = {}
        return data

    def _apply_memory_operation(self, decision: Dict[str, Any], abstract: str, page_id: int) -> Optional[str]:
        op = (decision.get("operation") or "add").lower()
        importance = decision.get("importance") or "short"
        t_observed = decision.get("t_observed")
        t_valid = decision.get("t_valid")
        t_invalid = decision.get("t_invalid")
        entities = decision.get("entities") or []
        relations = decision.get("relations") or []
        extends_id = decision.get("extends_id")

        content = decision.get("updated_content") or abstract
        target_id = decision.get("target_id")

        # If store supports advanced ops, use them
        if isinstance(self.memory_store, AdvancedMemoryStore):
            if op == "noop":
                return None
            if op == "delete" and target_id:
                self.memory_store.delete_entry(target_id)
                if self.graph_store:
                    self.graph_store.mark_memory_status(target_id, "deleted")
                    self.graph_store.mark_memory_latest(target_id, False)
                return None

            entry = MemoryEntry(
                content=content,
                tier=importance if importance in ("short", "mid", "long") else "short",
                t_observed=t_observed or MemoryEntry.model_fields["t_observed"].default_factory(),
                t_valid=t_valid,
                t_invalid=t_invalid,
                source_page_id=str(page_id),
                version_of=target_id if op == "update" else None,
            )

            if op == "update" and target_id:
                self.memory_store.update_entry(target_id, entry)
                if self.graph_store:
                    self.graph_store.mark_memory_status(target_id, "superseded")
                    self.graph_store.link_memory_relation(entry.id, target_id, "UPDATES")
                    self.graph_store.mark_memory_latest(target_id, False)
            else:
                self.memory_store.add_entry(entry)

            if self.graph_store:
                self.graph_store.upsert_memory(entry.id, {
                    "content": entry.content,
                    "tier": entry.tier,
                    "status": entry.status,
                    "t_created": entry.t_created,
                    "t_observed": entry.t_observed,
                    "t_valid": entry.t_valid,
                    "t_invalid": entry.t_invalid,
                    "source_page_id": entry.source_page_id,
                })
                self.graph_store.add_entities_relations(
                    entry.id,
                    entities,
                    relations,
                    t_observed=entry.t_observed,
                    t_valid=entry.t_valid,
                    t_invalid=entry.t_invalid,
                )
                self.graph_store.mark_memory_latest(entry.id, True)
                if extends_id:
                    self.graph_store.link_memory_relation(entry.id, extends_id, "EXTENDS")
                # Episode tier: timestamped event node
                try:
                    self.graph_store.add_episode(
                        memory_id=entry.id,
                        content=entry.content,
                        timestamp=entry.t_observed,
                        entities=entities,
                    )
                except Exception:
                    pass
                # Semantic tier: fact nodes derived from relations
                name_to_type = {}
                for ent in entities or []:
                    n = ent.get("name")
                    if not isinstance(n, str):
                        continue
                    n = n.strip()
                    if not n:
                        continue
                    name_to_type[n] = ent.get("type") or "Entity"
                for r in relations:
                    head = r.get("head")
                    tail = r.get("tail")
                    rel_type = r.get("relation")
                    if not head or not tail:
                        continue
                    try:
                        h_name = head.strip() if isinstance(head, str) else str(head)
                        t_name = tail.strip() if isinstance(tail, str) else str(tail)
                        self.graph_store.add_semantic_fact(
                            head=h_name,
                            relation=rel_type,
                            tail=t_name,
                            memory_id=entry.id,
                            timestamp=entry.t_observed,
                            t_valid=r.get("t_valid") or entry.t_valid,
                            t_invalid=r.get("t_invalid") or entry.t_invalid,
                            head_type=name_to_type.get(h_name, "Entity"),
                            tail_type=name_to_type.get(t_name, "Entity"),
                        )
                    except Exception:
                        continue
                # If no explicit relations, still capture semantic statement linked to entities
                if not relations and entities:
                    try:
                        self.graph_store.add_semantic_statement(
                            content=entry.content,
                            entities=entities,
                            memory_id=entry.id,
                            timestamp=entry.t_observed,
                        )
                    except Exception:
                        pass
                # Conflict detection and resolution (for ADD operations)
                if op == "add" and entities:
                    self._resolve_conflicts(entry, entities)
            return entry.id
        else:
            # Fallback to simple add
            if op in ("add", "update"):
                self.memory_store.add(content)
                return None
        return None

    def _resolve_conflicts(self, new_entry: MemoryEntry, entities: List[Dict[str, Any]]) -> None:
        if not self.graph_store or not isinstance(self.memory_store, AdvancedMemoryStore):
            return
        names = [e.get("name") for e in entities if e.get("name")]
        if not names:
            return

        # Fetch related memories from graph
        try:
            related = self.graph_store.query_memories(names, depth=1, limit=10)
        except Exception as e:
            print(f"[WARN] Conflict query failed: {e}")
            return

        for row in related:
            mid = row.get("id")
            if not mid or mid == new_entry.id:
                continue
            existing = row.get("content", "")
            if not existing:
                continue
            if self._is_contradictory(existing, new_entry.content):
                try:
                    self.memory_store.supersede_entry(mid, t_invalid=new_entry.t_observed)
                    self.graph_store.mark_memory_status(mid, "superseded")
                except Exception as e:
                    print(f"[WARN] Failed to supersede conflict {mid}: {e}")

    def _is_contradictory(self, existing_fact: str, new_fact: str) -> bool:
        prompt = ConflictCheck_PROMPT.format(existing_fact=existing_fact, new_fact=new_fact)
        try:
            response = self.generator.generate_single(prompt=prompt)
            text = response.get("text", "")
            import json
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return bool(data.get("contradict", False))
        except Exception:
            return False
