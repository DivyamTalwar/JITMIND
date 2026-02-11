#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live end-to-end test for JITMind.

This script is intended to hit real providers:
- OpenRouter (LLM)
- Cohere (embeddings + rerank)
- Neo4j Aura (graph)

It uses only synthetic test data.

Required env vars:
  OPENROUTER_API_KEY
  OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
  OPENROUTER_MODEL (default: google/gemini-3-flash-preview)

  COHERE_API_KEY
  COHERE_BASE_URL (default: https://api.cohere.com)
  COHERE_EMBED_MODEL (default: embed-v4.0)
  COHERE_RERANK_MODEL (default: rerank-v4.0-pro)

  NEO4J_URI
  NEO4J_USERNAME
  NEO4J_PASSWORD
  NEO4J_DATABASE (default: neo4j)

Optional:
  JITMIND_E2E_RUN_ID
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import socket
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jitmind import (
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    MemoryAgent,
    ResearchAgent,
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


def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _opt(name: str, default: str) -> str:
    return os.getenv(name) or default


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}")


async def main() -> int:
    run_id = os.getenv("JITMIND_E2E_RUN_ID") or str(uuid.uuid4())[:8]
    user_id = f"e2e_user_{run_id}"

    base_dir = Path(".tmp_e2e") / f"jitmind_{run_id}"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    data_dir = base_dir / "data"
    ckpt_dir = base_dir / "checkpoints"
    prof_dir = base_dir / "profiles"
    index_dir = base_dir / "index"

    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    log(f"E2E run_id={run_id} base_dir={base_dir}")

    # --- Providers / clients ---
    openrouter_key = _req("OPENROUTER_API_KEY")
    openrouter_base = _opt("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model = _opt("OPENROUTER_MODEL", "google/gemini-3-flash-preview")

    cohere_key = _req("COHERE_API_KEY")
    cohere_base = _opt("COHERE_BASE_URL", "https://api.cohere.com")
    cohere_embed = _opt("COHERE_EMBED_MODEL", "embed-v4.0")
    cohere_rerank = _opt("COHERE_RERANK_MODEL", "rerank-v4.0-pro")

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")
    neo4j_db = _opt("NEO4J_DATABASE", "neo4j")
    docker_neo4j_name = None

    def _free_port() -> int:
        s = socket.socket()
        s.bind(("", 0))
        port = int(s.getsockname()[1])
        s.close()
        return port

    def _can_resolve(uri: str) -> bool:
        try:
            from urllib.parse import urlparse
            host = urlparse(uri).hostname
            if not host:
                return False
            socket.getaddrinfo(host, 7687)
            return True
        except Exception:
            return False

    def _wait_tcp(host: str, port: int, timeout_s: int = 60) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return True
            except Exception:
                time.sleep(1)
        return False

    def _wait_neo4j_bolt(uri: str, username: str, password: str, database: str, timeout_s: int = 90) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as e:
            raise RuntimeError("neo4j python driver is required for local Neo4j readiness checks") from e

        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                driver = GraphDatabase.driver(uri, auth=(username, password))
                with driver.session(database=database) as session:
                    session.run("RETURN 1 AS ok").single()
                driver.close()
                return
            except Exception as e:
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"Neo4j bolt did not become ready: {last_err}")

    def _can_connect_neo4j(uri: str, username: str, password: str, database: str) -> bool:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception:
            return False
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session(database=database) as session:
                session.run("RETURN 1 AS ok").single()
            driver.close()
            return True
        except Exception:
            return False

    def _start_local_neo4j() -> tuple[str, str, str, str]:
        nonlocal docker_neo4j_name
        bolt_port = _free_port()
        http_port = _free_port()
        password = f"jitmind_e2e_{run_id}_pw"
        docker_neo4j_name = f"jitmind-neo4j-e2e-{run_id}"
        import subprocess
        log(f"Starting local Neo4j via Docker (name={docker_neo4j_name}, bolt={bolt_port}, http={http_port})")
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                docker_neo4j_name,
                "-p",
                f"{bolt_port}:7687",
                "-p",
                f"{http_port}:7474",
                "-e",
                f"NEO4J_AUTH=neo4j/{password}",
                "neo4j:5",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not _wait_tcp("127.0.0.1", bolt_port, timeout_s=90):
            raise RuntimeError("Local Neo4j did not become ready in time")
        uri = f"bolt://127.0.0.1:{bolt_port}"
        _wait_neo4j_bolt(uri, "neo4j", password, "neo4j", timeout_s=120)
        return uri, "neo4j", password, "neo4j"

    # Prefer provided Aura; fallback to local docker if it doesn't resolve.
    if neo4j_uri and neo4j_user and neo4j_pass:
        if not _can_resolve(neo4j_uri):
            log(f"[WARN] NEO4J_URI does not resolve; falling back to local docker Neo4j")
            neo4j_uri, neo4j_user, neo4j_pass, neo4j_db = _start_local_neo4j()
        elif not _can_connect_neo4j(neo4j_uri, neo4j_user, neo4j_pass, neo4j_db):
            log(f"[WARN] NEO4J_URI is set but not reachable/auth failed; falling back to local docker Neo4j")
            neo4j_uri, neo4j_user, neo4j_pass, neo4j_db = _start_local_neo4j()
    else:
        log("[WARN] Neo4j env vars not fully set; starting local docker Neo4j for graph tests")
        neo4j_uri, neo4j_user, neo4j_pass, neo4j_db = _start_local_neo4j()

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

    # Provider sanity
    log("Testing OpenRouter LLM call...")
    pong = generator.generate_single(prompt="Reply with exactly: pong")
    assert "pong" in (pong.get("text") or "").lower(), "LLM sanity check failed"

    log("Testing Cohere embed + rerank...")
    import cohere  # type: ignore

    co = cohere.Client(cohere_key, base_url=cohere_base)
    emb = co.embed(texts=["hello world"], model=cohere_embed, input_type="search_query")
    vecs = getattr(emb, "embeddings", None) or emb["embeddings"]
    assert vecs and len(vecs[0]) > 10, "Cohere embedding sanity check failed"

    rr = co.rerank(
        model=cohere_rerank,
        query="what is a cat",
        documents=["A cat is an animal.", "The sky is blue."],
        top_n=2,
    )
    results = getattr(rr, "results", None) or rr["results"]
    assert results and (getattr(results[0], "index", None) if hasattr(results[0], "index") else results[0].get("index")) is not None

    log("Testing Neo4j connectivity...")
    graph_store.upsert_memory(f"e2e-memory-{run_id}", {"content": "e2e", "run_id": run_id})
    rows = graph_store.run_cypher("MATCH (m:Memory {run_id: $rid}) RETURN count(m) AS n", {"rid": run_id})
    assert rows and int(rows[0].get("n", 0)) >= 1, "Neo4j sanity check failed"

    # --- Local stores ---
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

    # --- Ingestion (synthetic corpus) ---
    log("Ingesting synthetic documents...")
    docs = [
        Document(
            content=(
                "User profile facts: My name is Divyam. I prefer concise, direct answers. "
                "I like coffee."
            ),
            source="e2e:profile",
            doc_id=f"profile-{run_id}",
            metadata={"run_id": run_id},
        ),
        Document(
            content=(
                "Alice works at Acme Corp. Acme Corp is located in New York. "
                "Use relation labels: WORKS_AT and LOCATED_IN."
            ),
            source="e2e:graph",
            doc_id=f"graph-{run_id}",
            metadata={"run_id": run_id},
        ),
        Document(
            content=(
                "On 2020-01-01, the CEO of Acme Corp became Alice. "
                "On 2022-01-01, the CEO of Acme Corp became Bob."
            ),
            source="e2e:temporal",
            doc_id=f"temporal-{run_id}",
            metadata={"run_id": run_id},
        ),
    ]

    pipeline = IngestionPipeline()
    updates = pipeline.ingest_documents(docs, memory_agent=memory_agent, user_id=user_id, extra_meta={"run_id": run_id})
    assert updates, "Ingestion produced no updates"

    pages = page_store.load()
    assert len(pages) >= 3, "Expected at least 3 pages"

    # Tag Memory nodes for cleanup (MemoryAgent doesn't forward run_id into graph props)
    for p in pages:
        mid = (p.meta or {}).get("memory_id")
        if mid:
            graph_store.upsert_memory(str(mid), {"run_id": run_id})

    # Verify 3-tier graph architecture is actually being populated
    log("Verifying graph tiers (Episode + Semantic)...")
    ep = graph_store.run_cypher(
        "MATCH (:Memory {run_id: $rid})-[:HAS_EPISODE]->(e:Episode) RETURN count(e) AS n",
        {"rid": run_id},
    )
    assert ep and int(ep[0].get("n", 0)) >= 1, "Expected at least 1 Episode node"

    sem = graph_store.run_cypher(
        "MATCH (:Memory {run_id: $rid})-[:HAS_SEMANTIC]->(s:Semantic) RETURN count(s) AS n",
        {"rid": run_id},
    )
    # Some runs may only create Semantic statements (still Semantic nodes); this should be non-zero in practice.
    assert sem and int(sem[0].get("n", 0)) >= 1, "Expected at least 1 Semantic node"

    # Verify profile store wrote something
    prof = profile_store.load(user_id)
    assert prof.user_id == user_id

    # --- Build retrievers ---
    log("Building retrievers (CohereDense + Graph + Index)...")
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
    replay = ExperienceReplayBuffer(max_size=1000)

    agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers={
            "vector": vector,
            "graph": graph_ret,
            "page_index": index_ret,
            # keyword retriever intentionally omitted to exercise built-in fallback
        },
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

    # --- Research queries ---
    log("Running ResearchAgent (sync) with checkpoints...")
    out1 = agent.research(
        "Where does Alice work and where is Acme Corp located?",
        checkpoint_id=f"e2e-{run_id}",
        resume=True,
        feedback=1.0,
        user_id=user_id,
    )
    assert out1.integrated_memory, "Research output is empty"

    log("Running ResearchAgent (resume from checkpoint)...")
    out2 = agent.research(
        "Where does Alice work?",
        checkpoint_id=f"e2e-{run_id}",
        resume=True,
        feedback=1.0,
        user_id=user_id,
    )
    assert out2.integrated_memory

    log("Running ResearchAgent (async)...")
    out3 = await agent.research_async(
        "Summarize what we know about Acme Corp.",
        checkpoint_id=f"e2e-{run_id}-async",
        resume=False,
        feedback=1.0,
        user_id=user_id,
    )
    assert out3.integrated_memory

    # --- Graph checks ---
    log("Graph retrieval sanity (query_memories + PPR)...")
    rows = graph_store.query_memories(["Alice", "Acme Corp"], depth=2, limit=10)
    assert isinstance(rows, list)

    ppr = graph_store.personalized_pagerank(["Alice", "Acme Corp"], depth=2, limit=10)
    assert isinstance(ppr, list)

    # --- Graph CRUD smoke ---
    log("Graph CRUD smoke...")
    graph_store.upsert_entity("E2EEntity", "Concept", {"run_id": run_id})
    graph_store.upsert_relation("E2EEntity", "RELATED_TO", "Acme Corp", {"run_id": run_id})

    # --- Maintenance (consolidation + summaries) ---
    log("Running maintenance runner (consolidation + summarization)...")
    # Run the existing script to simulate cron/production usage.
    import subprocess
    cmd = [
        os.environ.get("PYTHON", "python3"),
        "scripts/run_maintenance.py",
        "--data-dir",
        str(data_dir),
        "--limit",
        "50",
        "--summarize",
        "--consolidate",
    ]
    env = dict(os.environ)
    # Ensure maintenance has the same provider config.
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
    subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)

    log("Verifying community tier (Community nodes)...")
    comm = graph_store.run_cypher("MATCH (c:Community) RETURN count(c) AS n")
    assert comm and int(comm[0].get("n", 0)) >= 1, "Expected at least 1 Community node after maintenance"
    derives = graph_store.run_cypher("MATCH (:Memory)-[:DERIVES]->(:Memory) RETURN count(*) AS n")
    assert derives and int(derives[0].get("n", 0)) >= 1, "Expected at least 1 DERIVES edge after maintenance"

    log("E2E completed successfully")

    # Optional cleanup: remove local artifacts
    # shutil.rmtree(base_dir)

    # Optional cleanup: remove tagged memories
    # graph_store.run_cypher("MATCH (m:Memory {run_id: $rid}) DETACH DELETE m", {"rid": run_id})

    graph_store.close()
    if docker_neo4j_name:
        import subprocess
        log(f"Stopping local Neo4j container: {docker_neo4j_name}")
        subprocess.run(["docker", "stop", docker_neo4j_name], check=False, capture_output=True, text=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
