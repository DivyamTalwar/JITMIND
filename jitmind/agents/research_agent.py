# research_agent.py
# -*- coding: utf-8 -*-
"""
ResearchAgent Module

This module defines the ResearchAgent for the JITMind (JITMind) framework.

- ResearchAgent is responsible for research tasks, reasoning, and advanced information retrieval.
- It interacts with the MemoryAgent to store and access past knowledge as abstracts (memory is represented as a list[str], without events/tags).
- ResearchAgent uses explicit research functions to process queries and generate insights.
- Prompts within the module are placeholders for future extensions, such as customizable instructions or templates.

The module focuses on providing clear abstraction and extensible interfaces for research-related agent functionalities.
"""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import asyncio
from contextvars import ContextVar

from jitmind.prompts import (
    Planning_PROMPT,
    Integrate_PROMPT,
    InfoCheck_PROMPT,
    GenerateRequests_PROMPT,
    REFLECTION_PROMPT,
    HYDE_PROMPT,
    SELF_RAG_PROMPT,
)
from jitmind.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA, SELF_RAG_SCHEMA
)
from jitmind.generator import AbsGenerator
from jitmind.schemas import AdvancedMemoryStore, MemoryEntry
from jitmind.utils.checkpoint import CheckpointManager
from jitmind.learning.replay_buffer import ExperienceReplayBuffer
from jitmind.profile import UserProfileStore
from jitmind.observability import RetrievalTrace, TraceRecorder, TraceSink
try:
    from jitmind.graph import GraphMemoryStore, load_ontology_from_env
except ImportError:
    GraphMemoryStore = None  # type: ignore
    load_ontology_from_env = None  # type: ignore
try:
    from jitmind.retriever.graph_retriever import GraphRetriever
except ImportError:
    GraphRetriever = None  # type: ignore

class ResearchAgent:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: Optional[MemoryStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: Optional[AbsGenerator] = None,  # Generator instance is required
        max_iters: int = 3,
        dir_path: Optional[str] = None,  # Filesystem storage path
        system_prompts: Optional[Dict[str, str]] = None,  # system prompts dictionary
        reranker: Optional[Any] = None,
        rrf_k: int = 60,
        rerank_top_n: int = 50,
        rerank_weight: float = 1.0,
        enable_reflection_learning: bool = False,
        enable_hyde: bool = True,
        enable_self_rag: bool = True,
        enable_dynamic_alpha: bool = True,
        max_context_tokens: int = 8000,
        context_warning_threshold: float = 0.8,
        checkpoint_manager: Optional[CheckpointManager] = None,
        checkpoint_every: int = 1,
        replay_buffer: Optional[ExperienceReplayBuffer] = None,
        profile_store: Optional[UserProfileStore] = None,
        trace_sink: Optional[TraceSink] = None,
        trace_capture_content: bool = False,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or AdvancedMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.rerank_weight = rerank_weight
        self.enable_reflection_learning = enable_reflection_learning
        self.enable_hyde = enable_hyde
        self.enable_self_rag = enable_self_rag
        self.enable_dynamic_alpha = enable_dynamic_alpha
        self.max_context_tokens = max_context_tokens
        self.context_warning_threshold = context_warning_threshold
        self._last_hits: List[Hit] = []
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_every = max(1, checkpoint_every)
        self.replay_buffer = replay_buffer
        self.profile_store = profile_store
        self._current_user_id: Optional[str] = None
        self.trace_sink = trace_sink
        self.trace_capture_content = trace_capture_content
        self._trace_context: ContextVar[Optional[TraceRecorder]] = ContextVar(
            f"jitmind_trace_{id(self)}", default=None
        )
        self._last_trace: Optional[RetrievalTrace] = None
        
        # Initialize system_prompts (default empty strings)
        default_system_prompts = {
            "planning": "",
            "integration": "",
            "reflection": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            # Merge user prompts with defaults
            self.system_prompts = {**default_system_prompts, **system_prompts}

        # Auto-add graph retriever if Neo4j env is configured and packages available
        if "graph" not in self.retrievers and GraphMemoryStore is not None and GraphRetriever is not None:
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USERNAME")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
            if neo4j_uri and neo4j_user and neo4j_password:
                try:
                    ontology = load_ontology_from_env() if load_ontology_from_env else None
                    graph_store = GraphMemoryStore(
                        uri=neo4j_uri,
                        username=neo4j_user,
                        password=neo4j_password,
                        database=neo4j_db,
                        ontology=ontology,
                    )
                    self.retrievers["graph"] = GraphRetriever({"graph_store": graph_store, "use_ppr": True})
                except Exception as e:
                    print(f"[WARN] Failed to init GraphRetriever: {e}")

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                # Call retriever.build with page_store
                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")
                pass

    # ---- Public ----
    def research(
        self,
        request: str,
        checkpoint_id: Optional[str] = None,
        resume: bool = True,
        feedback: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> ResearchOutput:
        recorder = self._begin_trace(request)
        # Ensure retriever indexes are up to date before research
        self._update_retrievers()
        self._current_user_id = user_id
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request
        start_step = 0

        if checkpoint_id and resume:
            loaded = self._load_checkpoint_state(checkpoint_id)
            if loaded:
                temp = loaded.get("temp", temp)
                iterations = loaded.get("iterations", iterations)
                next_request = loaded.get("next_request", next_request)
                start_step = loaded.get("step", 0)

        for step in range(start_step, self.max_iters):
            # Load current memory state dynamically
            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)

            temp = self._search(plan, temp, request)

            # Self-RAG critic
            self_rag = None
            if self.enable_self_rag:
                retrieved = [h.snippet for h in self._last_hits[:5]]
                self_rag = self._self_rag_reflect(request, retrieved, temp.content)

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
                "self_rag": self_rag,
            })

            if checkpoint_id and self.checkpoint_manager and (step % self.checkpoint_every == 0):
                self._save_checkpoint_state(
                    checkpoint_id=checkpoint_id,
                    iterations=iterations,
                    temp=temp,
                    next_request=next_request,
                    step=step + 1,
                )

            if self_rag and (not self_rag.get("ISREL", True) or not self_rag.get("ISSUP", True) or not self_rag.get("ISUSE", True)):
                decision.enough = False
                if not decision.new_request:
                    decision.new_request = request

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request


        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }

        # Invoke reflection learning if enabled
        if self.enable_reflection_learning and temp.content:
            reflection = self.post_reflection(question=request, answer=temp.content)
            if reflection:
                raw["reflection"] = reflection

        if self.replay_buffer:
            try:
                self.replay_buffer.add_experience(
                    query=request,
                    retrieved=[h.snippet for h in self._last_hits[:20]],
                    response=temp.content,
                    feedback=feedback,
                    meta={"iterations": len(iterations)},
                )
            except Exception:
                pass

        trace = self._finish_trace(recorder)
        raw["retrieval_trace"] = trace.to_dict()
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    async def research_async(
        self,
        request: str,
        checkpoint_id: Optional[str] = None,
        resume: bool = True,
        feedback: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> ResearchOutput:
        recorder = self._begin_trace(request)
        self._update_retrievers()
        self._current_user_id = user_id

        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request
        start_step = 0

        if checkpoint_id and resume:
            loaded = self._load_checkpoint_state(checkpoint_id)
            if loaded:
                temp = loaded.get("temp", temp)
                iterations = loaded.get("iterations", iterations)
                next_request = loaded.get("next_request", next_request)
                start_step = loaded.get("step", 0)

        for step in range(start_step, self.max_iters):
            memory_state = self.memory_store.load()
            plan = await asyncio.to_thread(self._planning, next_request, memory_state)

            temp = await self._search_async(plan, temp, request)

            self_rag = None
            if self.enable_self_rag:
                retrieved = [h.snippet for h in self._last_hits[:5]]
                self_rag = await asyncio.to_thread(self._self_rag_reflect, request, retrieved, temp.content)

            decision = await asyncio.to_thread(self._reflection, request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
                "self_rag": self_rag,
            })

            if checkpoint_id and self.checkpoint_manager and (step % self.checkpoint_every == 0):
                self._save_checkpoint_state(
                    checkpoint_id=checkpoint_id,
                    iterations=iterations,
                    temp=temp,
                    next_request=next_request,
                    step=step + 1,
                )

            if self_rag and (not self_rag.get("ISREL", True) or not self_rag.get("ISSUP", True) or not self_rag.get("ISUSE", True)):
                decision.enough = False
                if not decision.new_request:
                    decision.new_request = request

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request

        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }

        if self.enable_reflection_learning and temp.content:
            reflection = self.post_reflection(question=request, answer=temp.content)
            if reflection:
                raw["reflection"] = reflection

        if self.replay_buffer:
            try:
                self.replay_buffer.add_experience(
                    query=request,
                    retrieved=[h.snippet for h in self._last_hits[:20]],
                    response=temp.content,
                    feedback=feedback,
                    meta={"iterations": len(iterations)},
                )
            except Exception:
                pass

        trace = self._finish_trace(recorder)
        raw["retrieval_trace"] = trace.to_dict()
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    def explain_last_retrieval(self) -> Optional[Dict[str, Any]]:
        """Return the last completed trace without exposing query text by default."""

        return self._last_trace.to_dict() if self._last_trace else None

    def _begin_trace(self, query: str) -> TraceRecorder:
        recorder = TraceRecorder(query, capture_content=self.trace_capture_content)
        self._trace_context.set(recorder)
        return recorder

    def _finish_trace(self, recorder: TraceRecorder) -> RetrievalTrace:
        trace = recorder.finish()
        self._last_trace = trace
        if self.trace_sink:
            try:
                self.trace_sink.emit(trace)
            except Exception as exc:
                recorder.record(
                    "trace.export",
                    attributes={"error_type": type(exc).__name__},
                )
        self._trace_context.set(None)
        return trace

    def _timed_retrieval(self, name: str, function, *args):
        recorder = self._trace_context.get()
        if recorder is None:
            return function(*args)
        with recorder.stage(name, input_count=len(args[0]) if args else None) as stage:
            result = function(*args)
            stage.output_count = sum(
                len(group) if isinstance(group, list) else 1
                for group in (result or [])
            )
            return result

    def _record_fusion_trace(
        self,
        tool_hits: Dict[str, List[Hit]],
        before_temporal: int,
        final_hits: List[Hit],
    ) -> None:
        recorder = self._trace_context.get()
        if recorder is None:
            return
        top = []
        for hit in final_hits[:5]:
            top.append(
                {
                    "page_id": hit.page_id,
                    "source": hit.source,
                    "rrf_score": hit.meta.get("rrf_score"),
                    "hybrid_score": hit.meta.get("hybrid_score"),
                    "rerank_score": hit.meta.get("rerank_score"),
                }
            )
        recorder.record(
            "retrieval.fusion",
            input_count=sum(len(hits) for hits in tool_hits.values()),
            output_count=len(final_hits),
            attributes={
                "channels": sorted(tool_hits),
                "deduplicated_before_temporal": before_temporal,
                "temporal_rejections": before_temporal - len(final_hits),
                "top_evidence": top,
            },
        )

    def _update_retrievers(self):
        """Ensure retriever indexes are up to date."""
        # Check if new pages require index updates
        current_page_count = len(self.page_store.load())
        
        # If page count changes, update all retriever indexes
        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"Page count changed ({self._last_page_count} -> {current_page_count}); updating retriever indexes...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        
        # Update page count
        self._last_page_count = current_page_count

    def _load_checkpoint_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        if not self.checkpoint_manager:
            return None
        state = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if not state:
            return None
        try:
            temp = Result(**state.get("temp_memory", {}))
        except Exception:
            temp = Result()
        return {
            "temp": temp,
            "iterations": state.get("iterations", []),
            "next_request": state.get("next_request"),
            "step": state.get("step", 0),
        }

    def _save_checkpoint_state(
        self,
        checkpoint_id: str,
        iterations: List[Dict[str, Any]],
        temp: Result,
        next_request: str,
        step: int,
    ) -> None:
        if not self.checkpoint_manager:
            return
        state = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
            "next_request": next_request,
            "step": step,
        }
        self.checkpoint_manager.save_checkpoint(checkpoint_id, state)

    # ---- Internal ----
    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """

        memory_context = self._build_memory_context(memory_state)
        
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt
        
        # Debug: print prompt length
        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4  # Rough estimate: 1 token ≈ 4 chars
        print(f"[DEBUG] Planning prompt length: {prompt_chars} chars (~{estimated_tokens} tokens)")

        try:
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                # keyword_collection=[request],
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", []),
                graph_queries=data.get("graph_queries", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[],
                graph_queries=[]
            )
    

    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        searching_prompt: Optional[str] = None
    ) -> Result:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        tool_hits: Dict[str, List[Hit]] = {}

        # Execute each planned tool and collect all hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._timed_retrieval(
                        "retriever.keyword",
                        self._search_by_keyword,
                        [combined_keywords],
                        5,
                    )
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)

            elif tool == "vector":
                if plan.vector_queries:
                    vector_queries = list(plan.vector_queries)
                    if self.enable_hyde:
                        hyde = self._hyde_expand(question)
                        if hyde:
                            vector_queries = vector_queries + [hyde]
                    vector_results = self._timed_retrieval(
                        "retriever.vector", self._search_by_vector, vector_queries, 5
                    )
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)

            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._timed_retrieval(
                        "retriever.page_index",
                        self._search_by_page_index,
                        plan.page_index,
                    )
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)

            elif tool == "graph":
                if plan.graph_queries:
                    graph_results = self._timed_retrieval(
                        "retriever.graph", self._search_by_graph, plan.graph_queries, 5
                    )
                    if graph_results and isinstance(graph_results[0], list):
                        for result_list in graph_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(graph_results)

            if hits:
                tool_hits[tool] = hits

        if not tool_hits:
            return result

        # Dynamic hybrid alpha scores (dense + sparse)
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}
        if self.enable_dynamic_alpha:
            for tool_name, hits in tool_hits.items():
                for h in hits:
                    if not h.page_id:
                        continue
                    score = h.meta.get("score", 0.0) if h.meta else 0.0
                    if tool_name == "vector":
                        dense_scores[h.page_id] = dense_scores.get(h.page_id, 0.0) + score
                    if tool_name == "keyword":
                        sparse_scores[h.page_id] = sparse_scores.get(h.page_id, 0.0) + score

        # RRF fusion across tools
        rrf_scores: Dict[str, float] = {}
        hit_by_key: Dict[str, Hit] = {}

        for _, hits in tool_hits.items():
            for rank, hit in enumerate(hits):
                key = hit.page_id or f"hit-{id(hit)}"
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if key not in hit_by_key:
                    hit_by_key[key] = hit
                else:
                    existing = hit_by_key[key]
                    existing_score = existing.meta.get("score", 0) if existing.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        hit_by_key[key] = hit

        unique_hits = list(hit_by_key.values())
        for h in unique_hits:
            key = h.page_id or f"hit-{id(h)}"
            h.meta["rrf_score"] = rrf_scores.get(key, 0.0)
            if self.enable_dynamic_alpha and h.page_id:
                dense = dense_scores.get(h.page_id, 0.0)
                sparse = sparse_scores.get(h.page_id, 0.0)
                if dense or sparse:
                    alpha = self._dynamic_alpha(question)
                    h.meta["hybrid_score"] = alpha * dense + (1 - alpha) * sparse

        # Apply tier-based score boosting if using AdvancedMemoryStore
        unique_hits = self._apply_tier_boost(unique_hits)

        sorted_hits = sorted(
            unique_hits,
            key=lambda h: (h.meta.get("tier_boosted_score", h.meta.get("rrf_score", 0.0)), h.meta.get("score", 0.0)),
            reverse=True
        )

        before_temporal = len(sorted_hits)
        sorted_hits = self._filter_temporal_hits(sorted_hits)

        if self.reranker:
            sorted_hits = self._rerank_hits(sorted_hits, question)

        self._record_fusion_trace(tool_hits, before_temporal, sorted_hits)

        # Keep last hits for Self-RAG critique
        self._last_hits = sorted_hits

        return self._integrate(sorted_hits, result, question)

    async def _search_async(self, plan: SearchPlan, result: Result, question: str) -> Result:
        tool_hits: Dict[str, List[Hit]] = {}
        tasks: List[tuple[str, Any]] = []

        for tool in plan.tools:
            if tool == "keyword" and plan.keyword_collection:
                combined_keywords = " ".join(plan.keyword_collection)
                tasks.append(("keyword", asyncio.to_thread(
                    self._timed_retrieval,
                    "retriever.keyword",
                    self._search_by_keyword,
                    [combined_keywords],
                    5,
                )))
            elif tool == "vector" and plan.vector_queries:
                vector_queries = list(plan.vector_queries)
                if self.enable_hyde:
                    hyde = await asyncio.to_thread(self._hyde_expand, question)
                    if hyde:
                        vector_queries = vector_queries + [hyde]
                tasks.append(("vector", asyncio.to_thread(
                    self._timed_retrieval,
                    "retriever.vector",
                    self._search_by_vector,
                    vector_queries,
                    5,
                )))
            elif tool == "page_index" and plan.page_index:
                tasks.append(("page_index", asyncio.to_thread(
                    self._timed_retrieval,
                    "retriever.page_index",
                    self._search_by_page_index,
                    plan.page_index,
                )))
            elif tool == "graph" and plan.graph_queries:
                tasks.append(("graph", asyncio.to_thread(
                    self._timed_retrieval,
                    "retriever.graph",
                    self._search_by_graph,
                    plan.graph_queries,
                    5,
                )))

        if not tasks:
            return result

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
        for (tool, _), res in zip(tasks, results):
            if isinstance(res, Exception):
                continue
            hits: List[Hit] = []
            if res and isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                for result_list in res:
                    hits.extend(result_list)
            else:
                hits.extend(res or [])
            if hits:
                tool_hits[tool] = hits

        if not tool_hits:
            return result

        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}
        if self.enable_dynamic_alpha:
            for tool_name, hits in tool_hits.items():
                for h in hits:
                    if not h.page_id:
                        continue
                    score = h.meta.get("score", 0.0) if h.meta else 0.0
                    if tool_name == "vector":
                        dense_scores[h.page_id] = dense_scores.get(h.page_id, 0.0) + score
                    if tool_name == "keyword":
                        sparse_scores[h.page_id] = sparse_scores.get(h.page_id, 0.0) + score

        rrf_scores: Dict[str, float] = {}
        hit_by_key: Dict[str, Hit] = {}

        for _, hits in tool_hits.items():
            for rank, hit in enumerate(hits):
                key = hit.page_id or f"hit-{id(hit)}"
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if key not in hit_by_key:
                    hit_by_key[key] = hit
                else:
                    existing = hit_by_key[key]
                    existing_score = existing.meta.get("score", 0) if existing.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        hit_by_key[key] = hit

        unique_hits = list(hit_by_key.values())
        for h in unique_hits:
            key = h.page_id or f"hit-{id(h)}"
            h.meta["rrf_score"] = rrf_scores.get(key, 0.0)
            if self.enable_dynamic_alpha and h.page_id:
                dense = dense_scores.get(h.page_id, 0.0)
                sparse = sparse_scores.get(h.page_id, 0.0)
                if dense or sparse:
                    alpha = self._dynamic_alpha(question)
                    h.meta["hybrid_score"] = alpha * dense + (1 - alpha) * sparse

        unique_hits = self._apply_tier_boost(unique_hits)

        sorted_hits = sorted(
            unique_hits,
            key=lambda h: (h.meta.get("tier_boosted_score", h.meta.get("rrf_score", 0.0)), h.meta.get("score", 0.0)),
            reverse=True
        )

        before_temporal = len(sorted_hits)
        sorted_hits = self._filter_temporal_hits(sorted_hits)

        if self.reranker:
            sorted_hits = await asyncio.to_thread(self._rerank_hits, sorted_hits, question)

        self._record_fusion_trace(tool_hits, before_temporal, sorted_hits)

        self._last_hits = sorted_hits
        return await asyncio.to_thread(self._integrate, sorted_hits, result, question)

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # Combine multiple keywords into a single query string
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    # Search each vector query independently and aggregate scores
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)
                    
            elif tool == "graph":
                if plan.graph_queries:
                    graph_results = self._search_by_graph(plan.graph_queries, top_k=5)
                    if graph_results and isinstance(graph_results[0], list):
                        for result_list in graph_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(graph_results)
                    all_hits.extend(hits)

        # Format all hits as text content without integration
        if not all_hits:
            return result
        
        # Deduplicate by page_id to avoid duplicate hits across tools
        unique_hits: Dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: List[Hit] = []  # hits without page_id
        for hit in all_hits:
            if hit.page_id:
                # If page_id is new or current hit has higher score, update
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    # If page_id exists, keep the higher-score hit
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                # Keep hits without page_id
                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        
        # Sort by score (if available), then format
        # Merge hits with and without page_id
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        for i, hit in enumerate(sorted_hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            # Collect unique sources
            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        # Update memory access strength (decay reinforcement)
        entry_ids: List[str] = []
        if hasattr(self.memory_store, "touch"):
            get_page = getattr(self.page_store, "get", None)
            if get_page:
                for h in hits:
                    if h.page_id:
                        try:
                            page = get_page(int(h.page_id))
                        except Exception:
                            page = None
                        if page and page.meta:
                            mid = page.meta.get("memory_id")
                            if mid:
                                entry_ids.append(str(mid))
                if entry_ids:
                    try:
                        self.memory_store.touch(entry_ids)
                    except Exception as e:
                        print(f"[WARN] Failed to update memory strength: {e}")

        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "No search results"
        
        system_prompt = self.system_prompts.get("integration")
        template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            
            # Process sources: ensure list of strings (convert ints if needed)
            llm_sources = data.get("sources", sources)
            if llm_sources:
                # Convert ints or mixed types to string list
                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources
            
            return Result(
                content=data.get("content", ""),
                sources=sources
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result

    # ---- search channels ----
    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                # BM25Retriever returns List[List[Hit]]
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                # IndexRetriever expects List[str]; join page_index as comma-separated string
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        
        # Fallback: read pages directly from page_store
        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]  # Wrap as List[List[Hit]]

    def _search_by_graph(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("graph")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in graph search: {e}")
                return []
        return []

    def _rerank_hits(self, hits: List[Hit], query: str) -> List[Hit]:
        if not hits or not self.reranker:
            return hits
        top_n = min(self.rerank_top_n, len(hits))
        candidates = hits[:top_n]
        docs = [h.snippet for h in candidates]
        try:
            ranked = self.reranker.rerank(query, docs, top_k=top_n)
        except Exception as e:
            print(f"Error in reranking: {e}")
            return hits
        # Build score map
        score_map = {idx: score for idx, score in ranked}
        for i, h in enumerate(candidates):
            h.meta["rerank_score"] = score_map.get(i, 0.0)
        candidates = sorted(
            candidates,
            key=lambda h: h.meta.get("tier_boosted_score", h.meta.get("hybrid_score", h.meta.get("rrf_score", 0.0)))
            + self.rerank_weight * h.meta.get("rerank_score", 0.0),
            reverse=True
        )
        return candidates + hits[top_n:]

    def _apply_tier_boost(self, hits: List[Hit]) -> List[Hit]:
        """Apply tier-based score boosting from AdvancedMemoryStore."""
        if not isinstance(self.memory_store, AdvancedMemoryStore):
            return hits
        # Ensure tier promotion/demotion is up-to-date at retrieval time
        try:
            self.memory_store.promote_demote()
        except Exception:
            pass

        get_page = getattr(self.page_store, "get", None)
        if not get_page:
            return hits

        for h in hits:
            if not h.page_id:
                continue
            try:
                page = get_page(int(h.page_id))
            except Exception:
                continue
            if not page or not page.meta:
                continue
            memory_id = page.meta.get("memory_id")
            if memory_id:
                tier_score = self.memory_store.get_tier_score(str(memory_id))
                base_score = h.meta.get("hybrid_score", h.meta.get("rrf_score", 0.0))
                h.meta["tier_score"] = tier_score
                h.meta["tier_boosted_score"] = base_score * tier_score
            else:
                h.meta["tier_boosted_score"] = h.meta.get("hybrid_score", h.meta.get("rrf_score", 0.0))
        return hits

    def _build_memory_context(self, memory_state: MemoryState) -> str:
        profile_ctx = self._profile_context()
        if not memory_state.abstracts:
            return profile_ctx or "No memory currently."

        # Prefer ranked entries if available and context pressure is high
        if isinstance(self.memory_store, AdvancedMemoryStore):
            entries = self.memory_store.get_ranked_entries(limit=len(self.memory_store.get_entries()))
            lines = []
            for e in entries:
                if e.source_page_id is not None:
                    lines.append(f"Page {e.source_page_id}: {e.content}")
                else:
                    lines.append(f"Memory {e.id[:8]}: {e.content}")
            full_context = "\n".join(lines)
            pressure, needs_trim = self._check_context_pressure(full_context)
            if not needs_trim:
                return f"{profile_ctx}\n\n{full_context}" if profile_ctx else full_context
            # Trim to fit budget
            budget = int(self.max_context_tokens * self.context_warning_threshold)
            trimmed_lines = []
            current = 0
            for line in lines:
                current += self._estimate_tokens(line)
                if current > budget:
                    break
                trimmed_lines.append(line)
            context = "\n".join(trimmed_lines) if trimmed_lines else "No memory currently."
            return f"{profile_ctx}\n\n{context}" if profile_ctx else context

        # Fallback: use recent abstracts if pressure is high
        lines = [f"Page {i}: {a}" for i, a in enumerate(memory_state.abstracts)]
        full_context = "\n".join(lines)
        pressure, needs_trim = self._check_context_pressure(full_context)
        if not needs_trim:
            return f"{profile_ctx}\n\n{full_context}" if profile_ctx else full_context
        # Keep most recent abstracts
        budget = int(self.max_context_tokens * self.context_warning_threshold)
        trimmed_lines = []
        current = 0
        for line in reversed(lines):
            current += self._estimate_tokens(line)
            if current > budget:
                break
            trimmed_lines.append(line)
        trimmed_lines.reverse()
        context = "\n".join(trimmed_lines) if trimmed_lines else "No memory currently."
        return f"{profile_ctx}\n\n{context}" if profile_ctx else context

    def _profile_context(self) -> str:
        if not self.profile_store or not self._current_user_id:
            return ""
        try:
            profile = self.profile_store.load(self._current_user_id)
        except Exception:
            return ""
        if not (profile.static or profile.dynamic or profile.traits):
            return ""
        return (
            "USER_PROFILE\n"
            f"static: {profile.static}\n"
            f"dynamic: {profile.dynamic}\n"
            f"traits: {profile.traits}"
        )

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _check_context_pressure(self, context: str) -> tuple[float, bool]:
        tokens = self._estimate_tokens(context)
        pressure = tokens / float(self.max_context_tokens)
        return pressure, pressure > self.context_warning_threshold

    def _dynamic_alpha(self, query: str) -> float:
        q = query.strip()
        tokens = q.split()
        keyword_score = 0
        if any(ch.isdigit() for ch in q):
            keyword_score += 1
        if any(sym in q for sym in ["_", "::", ".", "/", "\\"]):
            keyword_score += 1
        if len(tokens) <= 3:
            keyword_score += 1
        if q.isupper():
            keyword_score += 1
        if keyword_score >= 2:
            return 0.3  # favor sparse
        if len(tokens) >= 8:
            return 0.8  # favor dense
        return 0.5

    def _hyde_expand(self, question: str) -> str:
        prompt = HYDE_PROMPT.format(question=question)
        try:
            response = self.generator.generate_single(prompt=prompt)
            return (response.get("text") or "").strip()
        except Exception as e:
            print(f"[WARN] HyDE generation failed: {e}")
            return ""

    def _self_rag_reflect(self, question: str, retrieved: List[str], response: str) -> Optional[Dict[str, bool]]:
        if not retrieved:
            retrieved_text = ""
        else:
            retrieved_text = "\n".join(retrieved)
        prompt = SELF_RAG_PROMPT.format(
            question=question,
            retrieved=retrieved_text,
            response=response
        )
        try:
            resp = self.generator.generate_single(prompt=prompt, schema=SELF_RAG_SCHEMA)
            data = resp.get("json")
            if not data:
                import json
                text = resp.get("text", "")
                data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return {
                "ISREL": bool(data.get("ISREL", False)),
                "ISSUP": bool(data.get("ISSUP", False)),
                "ISUSE": bool(data.get("ISUSE", False)),
            }
        except Exception as e:
            print(f"[WARN] Self-RAG reflection failed: {e}")
            return None

    def _filter_temporal_hits(self, hits: List[Hit]) -> List[Hit]:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        filtered: List[Hit] = []
        get_page = getattr(self.page_store, "get", None)
        if get_page is None:
            return hits
        for h in hits:
            if not h.page_id:
                filtered.append(h)
                continue
            try:
                page = get_page(int(h.page_id))
            except Exception:
                page = None
            if not page or not page.meta:
                filtered.append(h)
                continue
            t_valid = page.meta.get("t_valid")
            t_invalid = page.meta.get("t_invalid")
            try:
                if t_valid:
                    tv = datetime.fromisoformat(str(t_valid).replace("Z", "+00:00"))
                    if tv > now:
                        continue
                if t_invalid:
                    ti = datetime.fromisoformat(str(t_invalid).replace("Z", "+00:00"))
                    if ti <= now:
                        continue
            except Exception:
                filtered.append(h)
                continue
            filtered.append(h)
        return filtered
        
        

    # ---- reflection & summarization ----
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_prompt: Optional[str] = None
    ) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        
        try:
            system_prompt = self.system_prompts.get("reflection")
            
            # Debug: print reflection prompt length
            result_content_chars = len(result.content)
            estimated_result_tokens = result_content_chars // 4
            print(f"[DEBUG] Reflection result.content length: {result_content_chars} chars (~{estimated_result_tokens} tokens)")
            
            # Step 1: Check for completeness of information
            template_check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_data = check_response.get("json") or json.loads(check_response["text"])
            
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            
            # Step 2: Generate a list of new requests
            template_generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                result=result.content
            )
            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = generate_response.get("json") or json.loads(generate_response["text"])
            
            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return ReflectionDecision(enough=False, new_request=None)

    def post_reflection(self, question: str, answer: str, feedback: Optional[str] = None) -> Optional[str]:
        if not self.enable_reflection_learning:
            return None
        prompt = REFLECTION_PROMPT.format(
            question=question,
            answer=answer,
            feedback=feedback or ""
        )
        try:
            response = self.generator.generate_single(prompt=prompt)
            reflection = response.get("text", "").strip()
        except Exception as e:
            print(f"Error in post reflection: {e}")
            return None

        if not reflection:
            return None

        if isinstance(self.memory_store, AdvancedMemoryStore):
            entry = MemoryEntry(
                content=reflection,
                tier="long",
                meta={"type": "reflection", "question": question}
            )
            self.memory_store.add_entry(entry)
        else:
            self.memory_store.add(reflection)
        return reflection
