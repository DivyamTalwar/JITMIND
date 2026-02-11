#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JITMind + LoCoMo evaluation script.

Combines locomoqa_v3.py data processing with the JITMind framework to test multi-turn dialogue memory.
"""

import sys
import os
import re
import json
import math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm


from jitmind import (
    MemoryAgent,
    ResearchAgent,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGenerator,
    VLLMGeneratorConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

# ========== Data Loading (from locomoqa_v3.py) ==========

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON and return the list of samples."""
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")

def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    Extract sessions as (idx, timestamp, turns, optional_session_summary).
    """
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r'^session_(\d+)$', k)
        if not (m and isinstance(v, list)):
            continue
        original_idx = int(m.group(1))
        idx = original_idx - 1
        ts = conv_obj.get(f"session_{original_idx}_date_time", "")
        ssum = conv_obj.get(f"session_{original_idx}_summary", None)
        sessions.append((idx, ts, v, ssum if isinstance(ssum, str) and ssum.strip() else None))
    sessions.sort(key=lambda x: x[0])
    return sessions

def session_to_text(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    # Put time info first using a clearer format
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {ts} ==="]
    lines.append("")  # blank line separator
    
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id  = turn.get("dia_id", "")
        text    = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
    
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    
    return "\n".join(lines).strip()

def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    """Build session chunks from a sample."""
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text(idx, ts, turns, ssum))
    return chunks

def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect QA items from a sample."""
    qas: List[Dict[str, Any]] = []
    sid = sample.get("sample_id", None)
    for q in sample.get("qa", []):
        qas.append({
            "sample_id": sid,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "category": q.get("category"),
            "evidence": q.get("evidence"),
        })
    return qas

# ========== Prompt Design (from locomoqa_v3.py) ==========

def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """Parse model output (string/dict) into dict; return None on failure."""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None

def make_summary_prompt(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.
    
QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def answer_with_summary(category: Optional[int], summary: str, question: str, generator) -> str:
    """Select prompt by category."""
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()

# ========== Metrics (from eval_metric_locomo.py) ==========

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)  # drop english articles
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision

def compute_metrics_by_category(items, pred_key: str = "summary_answer", pred_field: str = "answer"):
    agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        pred = ""
        val = ex.get(pred_key, "")
        if isinstance(val, dict):
            pred = val.get(pred_field, "")
        else:
            pred = val
        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({
            "q_idx": idx,
            "category": cat,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "BLEU1": b1
        })
    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        if scores:
            f1_avg = sum(s[0] for s in scores)/len(scores)
            b1_avg = sum(s[1] for s in scores)/len(scores)
            summary.append({"category": cat, "count": len(scores), "F1_avg": f1_avg, "BLEU1_avg": b1_avg})
    return summary, rows

# ========== Core Processing Logic ==========

def process_sample(
    sample: Dict[str, Any], 
    sample_index: int, 
    outdir: str,
    memory_api_key: str,
    memory_base_url: str,
    memory_model: str,
    research_api_key: str,
    research_base_url: str,
    research_model: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    use_schema: bool = False,
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai"
):
    """
    Process a single sample with JITMind.

    Flow:
    1. Build memory with MemoryAgent
    2. Run deep research with ResearchAgent
    3. Answer based on research result
    """
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"Processing sample #{sample_index}: {sample_id}")
    print(f"{'='*60}")
    
    try:
        # 1. Build session chunks
        session_chunks = build_session_chunks_for_sample(sample)
        print(f"Sessions: {len(session_chunks)}")
        if session_chunks:
            print(f"First session preview:\n{session_chunks[0][:400]}...")
        
        # Create output directory
        sample_results_dir = os.path.join(outdir, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"Output directory: {sample_results_dir}")
        
        # 2. Create shared stores
        memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
        page_store = InMemoryPageStore(dir_path=sample_results_dir)
        
        # 3. Create Memory Generator
        print("\nStep 1: Create Memory Generator")
        if memory_api_type == "openai":
            memory_generator_config = OpenAIGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=0.3,
                max_tokens=256
            )
            memory_generator = OpenAIGenerator(memory_generator_config.__dict__)
        elif memory_api_type == "vllm":
            memory_generator_config = VLLMGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=0.3,
                max_tokens=256
            )
            memory_generator = VLLMGenerator(memory_generator_config.__dict__)
        print("[OK] Memory Generator created")
        
        # 4. Build memory with MemoryAgent (each session as one message)
        print("\nStep 2: Build memory with MemoryAgent")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator
        )
        
        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            for i, session_chunk in enumerate(session_chunks, 1):
                print(f"  Processing session {i}/{len(session_chunks)}...")
                memory_update = memory_agent.memorize(session_chunk)
        
        # Inspect built memory
        final_state = memory_store.load()
        print(f"[OK] Memory built. Total abstracts: {len(final_state.abstracts)}")
        
        # Show memory abstracts
        print("\nMemory abstracts:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        
        # Save memory state
        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] Memory state saved: {memory_state_file}")
        
        # 5. Create retrievers
        print("\nStep 3: Create retrievers")
        retrievers = {}
        
        # Index retriever
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
            # If index dir exists, remove it to avoid \"Directory not empty\"
            if os.path.exists(page_index_dir):
                import shutil
                shutil.rmtree(page_index_dir)
                print(f"[INFO] Cleaning existing page index dir: {page_index_dir}")
            
            index_config = IndexRetrieverConfig(
                index_dir=page_index_dir
            )
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            print("[OK] Index retriever created")
        except Exception as e:
            print(f"[WARN] Index retriever creation failed: {e}")
        
        # BM25 retriever
        try:
            bm25_index_dir = os.path.join(sample_results_dir, "bm25_index")
            # If index dir exists, remove it to avoid \"Directory not empty\"
            if os.path.exists(bm25_index_dir):
                import shutil
                shutil.rmtree(bm25_index_dir)
                print(f"[INFO] Cleaning existing BM25 index dir: {bm25_index_dir}")
            
            bm25_config = BM25RetrieverConfig(
                index_dir=bm25_index_dir,
                threads=1
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print("[OK] BM25 retriever created")
        except Exception as e:
            print(f"[WARN] BM25 retriever creation failed: {e}")
        
        # Dense retriever
        try:
            dense_index_dir = os.path.join(sample_results_dir, "dense_index")
            # If index dir exists, remove it to avoid \"Directory not empty\"
            if os.path.exists(dense_index_dir):
                import shutil
                shutil.rmtree(dense_index_dir)
                print(f"[INFO] Cleaning existing dense index dir: {dense_index_dir}")

            dense_config = DenseRetrieverConfig(
                index_dir=dense_index_dir,
                model_name="BAAI/bge-m3"
            )

            # dense_config = DenseRetrieverConfig(
            #     index_dir=dense_index_dir,
            #     api_url="http://localhost:8001"  # API mode: shared service
            # )

            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print("[OK] Dense retriever created")
        except Exception as e:
            print(f"[WARN] Dense retriever creation failed: {e}")
        
        print(f"[INFO] Created {len(retrievers)} retrievers")
        
        print("\nStep 4: Create Research Generator and Working Generator")
        if research_api_type == "openai":
            research_generator_config = OpenAIGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=2048,
                use_schema=use_schema
            )
            research_generator = OpenAIGenerator(research_generator_config.__dict__)
        elif research_api_type == "vllm":
            research_generator_config = VLLMGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=2048,
                use_schema=use_schema
            )
            research_generator = VLLMGenerator(research_generator_config.__dict__)

        if working_api_type == "openai":
            working_generator_config = OpenAIGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=256
            )
            working_generator = OpenAIGenerator(working_generator_config.__dict__)
        elif working_api_type == "vllm":
            working_generator_config = VLLMGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=256
            )
            working_generator = VLLMGenerator(working_generator_config.__dict__)
        print("[OK] Research and Working Generators created")


        # 6. Create ResearchAgent
        print("\nStep 5: Create ResearchAgent")
        research_agent = ResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=research_generator,
            max_iters=3
        )
        print("[OK] ResearchAgent created")
        
        # 7. Run QA
        print("\nStep 6: Run QA")
        qas = collect_qa_items_for_sample(sample)
        print(f"Total questions: {len(qas)}")
        
        # Worker to process a single question
        def process_question(qi_with_index):
            """Process one question."""
            i, qi = qi_with_index
            q = qi.get("question") or ""
            gold = qi.get("answer")
            cat = qi.get("category")
            
            print(f"\n--- Question {i}/{len(qas)} ---")
            print(f"Question: {q}")
            print(f"Gold: {gold}")
            print(f"Category: {cat}")
            
            if cat == 5:
                return None

            try:
                # Run research via ResearchAgent
                print(f"[Q{i}] Running deep research...")
                result = research_agent.research(q)
                research_summary = result.integrated_memory
                print(f"[Q{i}] [OK] Research complete. Iterations: {len(result.raw_memory.get('iterations', []))}")
                print(f"[Q{i}] Research summary: {research_summary[:200]}...")
                
                # Save research trace
                research_trace = {
                    "question": q,
                    "raw_memory": result.raw_memory,
                    "integrated_memory": result.integrated_memory,
                    "iterations": result.raw_memory.get("iterations", []),
                    "search_plans": result.raw_memory.get("search_plans", []),
                    "reflections": result.raw_memory.get("reflections", [])
                }
                
                # Save per-question research trace
                trace_file = os.path.join(sample_results_dir, f"research_trace_q{i}.json")
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(research_trace, f, ensure_ascii=False, indent=2)
                print(f"[Q{i}] [INFO] Research trace saved: {trace_file}")
                
                # Generate answer based on category-specific prompt
                print(f"[Q{i}] Generating answer...")
                summary_answer = answer_with_summary(cat, research_summary, q, working_generator)
                
                print(f"[Q{i}] Predicted answer: {summary_answer}")
                
                qa_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "research_summary": research_summary,
                    "summary_answer": summary_answer,
                    "iterations": len(result.raw_memory.get("iterations", [])),
                    "research_trace_file": trace_file
                }
                return qa_result
            
            except Exception as e:
                print(f"[Q{i}] [ERROR] Failed to process question: {e}")
                import traceback
                traceback.print_exc()
                qa_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "error": str(e)
                }
                return qa_result
        
        # Process all questions
        qa_items_with_index = [(i, qi) for i, qi in enumerate(qas, 1)]
        
        print(f"Starting sequential processing for {len(qa_items_with_index)} questions...")
        
        qa_results = []
        for qa_item in tqdm(qa_items_with_index, desc="Processing questions"):
            result = process_question(qa_item)
            # Filter out None results (category == 5)
            if result is not None:
                qa_results.append(result)
        
        # Save results
        results_file = os.path.join(sample_results_dir, "qa_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Results saved to: {results_file}")
        
        # Save research trace summary
        all_research_traces = []
        for i, qa_result in enumerate(qa_results, 1):
            if "research_trace_file" in qa_result:
                trace_file = qa_result["research_trace_file"]
                if os.path.exists(trace_file):
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)
                        all_research_traces.append({
                            "question_index": i,
                            "question": qa_result["question"],
                            "category": qa_result["category"],
                            "research_trace": trace_data
                        })
        
        if all_research_traces:
            traces_summary_file = os.path.join(sample_results_dir, "all_research_traces.json")
            with open(traces_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
            print(f"[OK] Research trace summary saved to: {traces_summary_file}")
        
        # Summary
        print(f"\n{'='*60}")
        print("Processing summary")
        print(f"{'='*60}")
        print(f"Sample ID: {sample_id}")
        print(f"Sessions: {len(session_chunks)}")
        print(f"Memory abstracts: {len(final_state.abstracts)}")
        print(f"Questions processed: {len(qa_results)}")
        print(f"Research trace files: {len(all_research_traces)}")
        print(f"Results saved to: {sample_results_dir}")
        print("  - QA results: qa_results.json")
        print("  - Memory state: memory_state.json")
        print("  - Research trace summary: all_research_traces.json")
        print("  - Per-question traces: research_trace_q*.json")
        
        return qa_results
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_index}: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return []


# ========== Main ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JITMind + LoCoMo evaluation")
    parser.add_argument("--data", type=str, default="/path/to/locomo/dataset.json", 
                        help="LoCoMo dataset path")
    parser.add_argument("--outdir", type=str, default="./results/locomo",
                        help="Output directory")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample index")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (exclusive); None for all")
    
    # Memory Generator config
    parser.add_argument("--memory-api-key", type=str, default="empty", help="Memory model API key")
    parser.add_argument("--memory-base-url", type=str, default="https://api.openai.com/v1", help="Memory model base URL")
    parser.add_argument("--memory-model", type=str, default="gpt-4o-mini", help="Memory model name")
    parser.add_argument("--memory-api-type", type=str, default="openai", choices=["openai", "vllm"], help="Memory API type")
    
    # Research Generator config
    parser.add_argument("--research-api-key", type=str, default="empty", help="Research model API key")
    parser.add_argument("--research-base-url", type=str, default="https://api.openai.com/v1", help="Research model base URL")
    parser.add_argument("--research-model", type=str, default="gpt-4o-mini", help="Research model name")
    parser.add_argument("--research-api-type", type=str, default="openai", choices=["openai", "vllm"], help="Research API type")
    parser.add_argument("--use-schema", type=bool, default=False, help="Use JSON schema")

    # Working Generator config
    parser.add_argument("--working-api-key", type=str, default="empty", help="Working model API key")
    parser.add_argument("--working-base-url", type=str, default="https://api.openai.com/v1", help="Working model base URL")
    parser.add_argument("--working-model", type=str, default="gpt-4o-mini", help="Working model name")
    parser.add_argument("--working-api-type", type=str, default="openai", choices=["openai", "vllm"], help="Working API type")

    args = parser.parse_args()
    
    print("=" * 60)
    print("JITMind + LoCoMo evaluation")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Output directory: {args.outdir}")
    print(f"Sample range: {args.start_idx} to {args.end_idx-1 if args.end_idx else 'all'} (total {args.end_idx - args.start_idx if args.end_idx else 'all'})")
    print("=" * 60)
    
    # Load data
    samples = load_locomo(args.data)
    print(f"Loaded {len(samples)} samples")
    
    # Reset end index after load
    if args.end_idx is None:
        args.end_idx = len(samples)
    
    print(f"Actual range: {args.start_idx} to {args.end_idx-1} (total {args.end_idx - args.start_idx})")
    
    # Validate index bounds
    if args.start_idx < 0 or args.start_idx >= len(samples):
        print(f"Error: start index {args.start_idx} is out of range (total samples: {len(samples)})")
        return
    
    if args.end_idx > len(samples):
        print(f"Warning: end index {args.end_idx} is out of range, clamped to {len(samples)}")
        args.end_idx = len(samples)
    
    if args.start_idx >= args.end_idx:
        print(f"Error: start index {args.start_idx} must be less than end index {args.end_idx}")
        return
    
    # Process samples sequentially
    sample_indices = list(range(args.start_idx, args.end_idx))
    
    print(f"Processing {len(sample_indices)} samples sequentially...")
    
    all_results = []
    
    # Process samples sequentially
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        sample = samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"Starting sample {sample_idx}/{len(samples)-1} (range: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            results = process_sample(
                sample, 
                sample_idx, 
                args.outdir,
                args.memory_api_key,
                args.memory_base_url,
                args.memory_model,
                args.research_api_key,
                args.research_base_url,
                args.research_model,
                args.working_api_key,
                args.working_base_url,
                args.working_model,
                args.use_schema,
                args.memory_api_type,
                args.research_api_type,
                args.working_api_type
            )
            print(f"[OK] Sample {sample_idx} completed")
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] Sample {sample_idx} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save aggregated results
    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Batch results saved: {summary_file}")
        
        # Compute metrics
        print(f"\n{'='*60}")
        print("Computing metrics...")
        print(f"{'='*60}")
        
        # Metrics for summary_answer
        pred_key = "summary_answer"
        pred_field = "answer"
        
        print(f"\n# LoCoMo Metrics for pred_key='{pred_key}', pred_field='{pred_field}'")
        summary, details = compute_metrics_by_category(all_results, pred_key=pred_key, pred_field=pred_field)
        
        # Print statistics
        print("\nBy-category summary:")
        for r in summary:
            print(f"Category {r['category']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}")
        
        # Compute overall averages
        all_f1_scores = [row["F1"] for row in details]
        all_bleu1_scores = [row["BLEU1"] for row in details]
        overall_f1_avg = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        overall_bleu1_avg = sum(all_bleu1_scores) / len(all_bleu1_scores) if all_bleu1_scores else 0.0
        
        print("\nOverall summary:")
        print(f"Total questions: {len(all_results)}")
        print(f"Overall avg F1: {overall_f1_avg:.4f}")
        print(f"Overall avg BLEU1: {overall_bleu1_avg:.4f}")
        
        # Save stats to JSON file (similar to hotpotqa_test.py)
        statistics = {
            "total_samples": args.end_idx - args.start_idx,
            "total_questions": len(all_results),
            "overall_f1_avg": overall_f1_avg,
            "overall_bleu1_avg": overall_bleu1_avg,
            "by_category": summary,
            "details": details,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx - 1
        }
        
        stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {stats_file}")
    
    print(f"\n{'='*60}")
    print("[OK] Batch evaluation completed")
    print(f"Samples processed: {args.end_idx - args.start_idx}")
    print(f"Questions processed: {len(all_results)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
