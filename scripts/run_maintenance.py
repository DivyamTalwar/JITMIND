#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cron-safe maintenance runner:
- optional consolidation
- optional hierarchical summarization
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running from `scripts/` without requiring `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jitmind import (
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    AdvancedMemoryStore,
    HierarchicalSummarizer,
    MemoryConsolidator,
    GraphMemoryStore,
)
from jitmind.graph import load_ontology_from_env


def main() -> int:
    parser = argparse.ArgumentParser(description="JITMind maintenance runner")
    parser.add_argument("--data-dir", default="./data", help="Memory store directory")
    parser.add_argument("--limit", type=int, default=200, help="Max entries to process")
    parser.add_argument("--summarize", action="store_true", help="Run hierarchical summarization")
    parser.add_argument("--consolidate", action="store_true", help="Run sleep-time consolidation")
    parser.add_argument("--summary-level1", type=int, default=10, help="Max L1 clusters")
    parser.add_argument("--summary-level2", type=int, default=5, help="Max L2 clusters")
    parser.add_argument("--consolidate-threshold", type=float, default=0.85, help="Similarity threshold")
    parser.add_argument("--consolidate-max-cluster", type=int, default=10, help="Max cluster size")
    args = parser.parse_args()

    do_summarize = args.summarize
    do_consolidate = args.consolidate
    if not do_summarize and not do_consolidate:
        do_summarize = True
        do_consolidate = True

    gen_config = OpenAIGeneratorConfig(
        model_name=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.2,
        max_tokens=512,
    )
    generator = OpenAIGenerator.from_config(gen_config)

    memory_store = AdvancedMemoryStore(dir_path=args.data_dir)
    graph_store = None
    if GraphMemoryStore:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                ontology = load_ontology_from_env()
                graph_store = GraphMemoryStore(
                    uri=neo4j_uri,
                    username=neo4j_user,
                    password=neo4j_password,
                    database=neo4j_db,
                    ontology=ontology,
                )
            except Exception:
                graph_store = None

    print(f"[{datetime.utcnow().isoformat()}] Maintenance start")
    if do_consolidate:
        consolidator = MemoryConsolidator(
            generator=generator,
            memory_store=memory_store,
            similarity_threshold=args.consolidate_threshold,
            max_cluster_size=args.consolidate_max_cluster,
        )
        new_entries = consolidator.consolidate(limit=args.limit)
        print(f"[OK] Consolidated: {len(new_entries)} new entries")
        if graph_store:
            for entry in new_entries:
                graph_store.upsert_memory(entry.id, {
                    "content": entry.content,
                    "tier": entry.tier,
                    "status": entry.status,
                    "t_created": entry.t_created,
                    "t_observed": entry.t_observed,
                    "t_valid": entry.t_valid,
                    "t_invalid": entry.t_invalid,
                    "source_page_id": entry.source_page_id,
                })
                # Community tier: consolidated cluster summary
                try:
                    members = entry.meta.get("consolidated_from") if entry.meta else None
                    graph_store.add_community_summary(
                        community_id=entry.id,
                        summary=entry.content,
                        member_ids=members or [],
                        level=0,
                    )
                except Exception:
                    pass

    if do_summarize:
        summarizer = HierarchicalSummarizer(
            generator=generator,
            memory_store=memory_store,
            max_level1_clusters=args.summary_level1,
            max_level2_clusters=args.summary_level2,
        )
        summaries = summarizer.build_summaries(limit=args.limit, include_existing_summaries=False)
        for entry in summaries:
            memory_store.add_entry(entry)
            if graph_store:
                graph_store.upsert_memory(entry.id, {
                    "content": entry.content,
                    "tier": entry.tier,
                    "status": entry.status,
                    "t_created": entry.t_created,
                    "t_observed": entry.t_observed,
                    "t_valid": entry.t_valid,
                    "t_invalid": entry.t_invalid,
                    "source_page_id": entry.source_page_id,
                })
                # Community tier: hierarchical summaries
                try:
                    level = entry.meta.get("summary_level") if entry.meta else None
                    members = entry.meta.get("children") if entry.meta else None
                    graph_store.add_community_summary(
                        community_id=entry.id,
                        summary=entry.content,
                        member_ids=members or [],
                        level=level,
                    )
                except Exception:
                    pass
        print(f"[OK] Summaries added: {len(summaries)}")

    print(f"[{datetime.utcnow().isoformat()}] Maintenance done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
