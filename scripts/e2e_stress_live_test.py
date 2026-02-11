#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live stress test for JITMind.

Goals:
- Exercise concurrency (multiple async research calls in parallel)
- Exercise idempotent ingestion (dedup by content hash)
- Exercise maintenance (summarization + consolidation)
- Exercise graph resilience (optional brief Neo4j outage)

This hits real providers:
- OpenRouter (LLM)
- Cohere (embeddings + rerank)
- Neo4j (localhost recommended)

Required env vars:
  OPENROUTER_API_KEY
  COHERE_API_KEY

Optional (defaults to localhost):
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=jitmind_local_password
  NEO4J_DATABASE=neo4j
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _opt(name: str, default: str) -> str:
    return os.getenv(name) or default


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


async def main() -> int:
    # Import after sys.path is stable (we run from repo root).
    import subprocess
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from neo4j import GraphDatabase  # type: ignore

    from jitmind import (
        OpenAIGenerator,
        OpenAIGeneratorConfig,
        MemoryAgent,
        AdvancedMemoryStore,
        InMemoryPageStore,
        CohereDenseRetriever,
        CohereReranker,
        GraphMemoryStore,
        GraphRetriever,
        IndexRetriever,
        CheckpointManager,
        ExperienceReplayBuffer,
        IngestionPipeline,
        Document,
    )
    from jitmind.profile import UserProfileStore, UserProfileAgent

    run_id = str(uuid.uuid4())[:8]
    user_id = f"stress_user_{run_id}"

    base_dir = Path(".tmp_e2e_stress") / f"jitmind_{run_id}"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    data_dir = base_dir / "data"
    ckpt_dir = base_dir / "checkpoints"
    prof_dir = base_dir / "profiles"
    index_dir = base_dir / "index"
    for d in (data_dir, ckpt_dir, prof_dir, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    openrouter_key = _req("OPENROUTER_API_KEY")
    openrouter_base = _opt("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model = _opt("OPENROUTER_MODEL", "google/gemini-3-flash-preview")

    cohere_key = _req("COHERE_API_KEY")
    cohere_base = _opt("COHERE_BASE_URL", "https://api.cohere.com")
    cohere_embed = _opt("COHERE_EMBED_MODEL", "embed-v4.0")
    cohere_rerank = _opt("COHERE_RERANK_MODEL", "rerank-v4.0-pro")

    neo4j_uri = _opt("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = _opt("NEO4J_USERNAME", "neo4j")
    neo4j_pass = _opt("NEO4J_PASSWORD", "jitmind_local_password")
    neo4j_db = _opt("NEO4J_DATABASE", "neo4j")

    # Ensure local Neo4j is up (compose is optional; if user already runs it, this is a no-op).
    log("Starting/ensuring local Neo4j (docker compose)...")
    subprocess.run(["docker", "compose", "-f", "docker-compose.neo4j.yml", "up", "-d"], check=True)

    # Wait for bolt readiness.
    deadline = time.time() + 90
    last = None
    while time.time() < deadline:
        try:
            drv = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            with drv.session(database=neo4j_db) as s:
                s.run("RETURN 1 AS ok").single()
            drv.close()
            last = None
            break
        except Exception as e:
            last = e
            time.sleep(1)
    if last is not None:
        raise RuntimeError(f"Neo4j did not become ready: {last}")

    gen_cfg = OpenAIGeneratorConfig(
        model_name=openrouter_model,
        api_key=openrouter_key,
        base_url=openrouter_base,
        temperature=0.0,
        max_tokens=1024,
        timeout=60.0,
        use_schema=False,
    )
    generator = OpenAIGenerator.from_config(gen_cfg)

    graph_store = GraphMemoryStore(
        uri=neo4j_uri,
        username=neo4j_user,
        password=neo4j_pass,
        database=neo4j_db,
        ontology=None,
    )

    memory_store = AdvancedMemoryStore(dir_path=str(data_dir))
    page_store = InMemoryPageStore(dir_path=str(data_dir))

    profile_store = UserProfileStore(dir_path=str(prof_dir))
    profile_agent = UserProfileAgent(generator=generator, profile_store=profile_store)

    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=generator,
        dir_path=str(data_dir),
        graph_store=graph_store,
        profile_agent=profile_agent,
    )

    # Ingest a slightly larger synthetic corpus.
    docs = [
        Document(content="Divyam prefers concise answers and loves coffee.", source="stress:profile", doc_id=f"p-{run_id}"),
        Document(content="Alice works at Acme Corp in New York.", source="stress:graph", doc_id=f"g1-{run_id}"),
        Document(content="Acme Corp has products: AlphaWidget and BetaWidget.", source="stress:graph", doc_id=f"g2-{run_id}"),
        Document(content="On 2020-01-01, CEO of Acme Corp became Alice.", source="stress:temp", doc_id=f"t1-{run_id}"),
        Document(content="On 2022-01-01, CEO of Acme Corp became Bob.", source="stress:temp", doc_id=f"t2-{run_id}"),
        Document(content="Bob lives in San Francisco and works remotely.", source="stress:misc", doc_id=f"m1-{run_id}"),
    ]

    pipeline = IngestionPipeline(deduplicate=True)
    log("Ingesting corpus (pass 1)...")
    updates1 = pipeline.ingest_documents(docs, memory_agent=memory_agent, user_id=user_id, extra_meta={"run_id": run_id})
    assert updates1, "ingestion pass 1 produced no updates"
    pages1 = page_store.load()
    assert len(pages1) >= 6

    log("Ingesting same corpus (pass 2, should dedup to zero updates)...")
    updates2 = pipeline.ingest_documents(docs, memory_agent=memory_agent, user_id=user_id, extra_meta={"run_id": run_id})
    assert updates2 == [], "expected dedup to prevent duplicate ingestion"
    pages2 = page_store.load()
    assert len(pages2) == len(pages1)

    # Build retrievers once.
    log("Building retrievers...")
    vector = CohereDenseRetriever({
        "api_key": cohere_key,
        "base_url": cohere_base,
        "model_name": cohere_embed,
        "input_type_doc": "search_document",
        "input_type_query": "search_query",
        "index_dir": str(index_dir / "cohere_dense"),
    })
    vector.build(page_store)

    graph_ret = GraphRetriever({"graph_store": graph_store, "use_ppr": True, "depth": 2, "top_k": 10})
    index_ret = IndexRetriever({"index_dir": str(index_dir / "page_index")})
    index_ret.build(page_store)

    reranker = CohereReranker({
        "api_key": cohere_key,
        "base_url": cohere_base,
        "model_name": cohere_rerank,
        "top_k": 10,
    })

    ckpt = CheckpointManager(dir_path=str(ckpt_dir))
    replay = ExperienceReplayBuffer(max_size=2000)

    def make_agent():
        # Separate store instances to exercise file-backed concurrency safety.
        local_pages = InMemoryPageStore(dir_path=str(data_dir))
        local_memory = AdvancedMemoryStore(dir_path=str(data_dir))
        from jitmind.agents.research_agent import ResearchAgent

        return ResearchAgent(
            page_store=local_pages,
            memory_store=local_memory,
            retrievers={"vector": vector, "graph": graph_ret, "page_index": index_ret},
            generator=generator,
            reranker=reranker,
            max_iters=2,
            enable_reflection_learning=True,
            enable_hyde=True,
            enable_self_rag=True,
            checkpoint_manager=ckpt,
            replay_buffer=replay,
            profile_store=profile_store,
        )

    queries = [
        "Where does Alice work and where is Acme Corp located?",
        "Who is the CEO of Acme Corp now?",
        "List Acme Corp products.",
        "Where does Bob live?",
        "Summarize what we know about Acme Corp.",
        "What does Divyam prefer?",
    ]

    log("Running async research calls concurrently...")
    t0 = time.perf_counter()

    async def run_query(i: int, q: str):
        agent = make_agent()
        out = await agent.research_async(
            q,
            checkpoint_id=f"stress-{run_id}-{i}",
            resume=False,
            feedback=1.0,
            user_id=user_id,
        )
        assert out.integrated_memory
        return out

    outs = await asyncio.gather(*[run_query(i, q) for i, q in enumerate(queries)], return_exceptions=True)
    for o in outs:
        if isinstance(o, Exception):
            raise o

    dt = time.perf_counter() - t0
    log(f"Concurrent research done in {dt:.2f}s for {len(queries)} queries")

    # Maintenance runner (cron path) should work against the same data directory.
    log("Running maintenance runner (summarize + consolidate)...")
    env = dict(os.environ)
    env.update({
        "OPENROUTER_API_KEY": openrouter_key,
        "OPENROUTER_BASE_URL": openrouter_base,
        "OPENROUTER_MODEL": openrouter_model,
        "COHERE_API_KEY": cohere_key,
        "COHERE_BASE_URL": cohere_base,
        "COHERE_EMBED_MODEL": cohere_embed,
        "COHERE_RERANK_MODEL": cohere_rerank,
        "NEO4J_URI": neo4j_uri,
        "NEO4J_USERNAME": neo4j_user,
        "NEO4J_PASSWORD": neo4j_pass,
        "NEO4J_DATABASE": neo4j_db,
    })
    subprocess.run(
        [
            os.environ.get("PYTHON", "python3"),
            "scripts/run_maintenance.py",
            "--data-dir",
            str(data_dir),
            "--limit",
            "200",
            "--summarize",
            "--consolidate",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    # Optional resilience smoke: briefly stop neo4j and ensure research still returns (vector/index only).
    log("Graph outage smoke (stop neo4j briefly)...")
    subprocess.run(["docker", "compose", "-f", "docker-compose.neo4j.yml", "stop"], check=True, capture_output=True, text=True)
    try:
        agent = make_agent()
        out = await agent.research_async("Where does Alice work?", resume=False, feedback=1.0, user_id=user_id)
        assert out.integrated_memory
    finally:
        subprocess.run(["docker", "compose", "-f", "docker-compose.neo4j.yml", "start"], check=True, capture_output=True, text=True)

    log("Stress test completed successfully")

    graph_store.close()

    # Tear down local Neo4j to keep environment clean.
    log("Tearing down local Neo4j (docker compose down -v)...")
    subprocess.run(["docker", "compose", "-f", "docker-compose.neo4j.yml", "down", "-v"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

