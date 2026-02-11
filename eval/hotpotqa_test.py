#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JITMind + HotpotQA evaluation script.

Based on test_gam_longbench.py, adapted to HotpotQA format.
HotpotQA format:
- context: str (long context)
- input: str (question)
- index: int
- answers: [] (gold answers, possibly multiple)
"""

import string
import sys
import os
import re
import json
from typing import Any, Counter, Dict, List, Optional, Tuple
from tqdm import tqdm


from jitmind import (
    MemoryAgent,
    ResearchAgent,
    VLLMGenerator,
    OpenAIGeneratorConfig,
    OpenAIGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

# ========== Data Loading ==========

def load_hotpotqa(json_path: str) -> List[Dict[str, Any]]:
    """
    Load HotpotQA JSON dataset.
    
    Args:
        json_path: JSON file path
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Extract all attributes
    data_all = [
        {
            "index": item.get("index", idx),
            "context": item.get("context", ""),
            "input": item.get("input", ""),
            "answers": item.get("answers", []),
            "_id": f"hotpotqa-{item.get('index', idx)}"  # unique ID
        }
        for idx, item in enumerate(dataset)
    ]
    
    return data_all

# ========== Long Text Chunking ==========

def build_context_chunks_for_sample(
    sample: Dict[str, Any], 
    max_tokens: int = 2000, 
    embedding_model_path: Optional[str] = None
) -> List[str]:
    """
    Split context into chunks by token count.
    Uses smarter boundary-aware splitting.
    
    Args:
        sample: sample with 'context'
        max_tokens: max tokens per chunk
        embedding_model_path: embedding model path for precise token counting (optional)
    """
    context_text = sample.get("context") or ""
    
    if not context_text:
        return []
    
    # Prefer precise split with embedding model if provided
    if embedding_model_path:
        try:
            chunks = _split_with_embedding_model(context_text, max_tokens, embedding_model_path)
            if chunks:
                return chunks
        except Exception as e:
            print(f"Warning: Embedding model splitting failed: {e}, falling back to tiktoken")
    
    # Use tiktoken for precise token split
    try:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = tokenizer.encode(context_text, disallowed_special=())
        
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{context_text}"]
        
        # Smart split by token count
        chunks = _smart_split_by_tokens(context_text, tokens, max_tokens, tokenizer)
        return chunks
        
    except ImportError:
        print("Warning: tiktoken not available, falling back to character-based splitting")
        return _fallback_char_split(context_text, max_tokens)

def _split_with_embedding_model(text: str, max_tokens: int, model_path: str) -> List[str]:
    """
    Precise token split using embedding model.
    """
    try:
        from transformers import AutoTokenizer
        
        # Use tokenizer from specified model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Encode text to tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{text}"]
        
        # Smart split
        chunks = _smart_split_by_tokens(text, tokens, max_tokens, tokenizer)
        return chunks
        
    except Exception as e:
        print(f"Error using embedding model: {e}")
        return []

def _smart_split_by_tokens(text: str, tokens: List[int], max_tokens: int, tokenizer) -> List[str]:
    """
    Simple split by token count: no boundary search, split by max_tokens.
    """
    chunks = []
    
    # If text is within max_tokens, return
    if len(tokens) <= max_tokens:
        return [f"[Session 1]\n{text}"]
    
    # Split directly by token indices
    session_id = 0
    start_idx = 0
    
    while start_idx < len(tokens):
        # Compute end token index for current chunk
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # Decode tokens back to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        if chunk_text.strip():
            chunks.append(f"[Session {session_id}]\n{chunk_text.strip()}")
            session_id += 1
        
        start_idx = end_idx
    
    return chunks

def _fallback_char_split(text: str, max_tokens: int) -> List[str]:
    """
    Fallback character-based splitting.
    """
    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return [f"[Session 1]\n{text}"]
    
    chunks = []
    current_start = 0
    session_id = 0
    
    while current_start < len(text):
        current_end = min(current_start + max_chars, len(text))
        
        # Try splitting at word boundary
        if current_end < len(text):
            # Find last newline
            last_newline = text.rfind('\n', current_start, current_end)
            if last_newline > current_start:
                current_end = last_newline
            else:
                # Find last space
                last_space = text.rfind(' ', current_start, current_end)
                if last_space > current_start:
                    current_end = last_space
        
        chunk_text = text[current_start:current_end].strip()
        if chunk_text:
            chunks.append(f"[Session {session_id}]\n{chunk_text}")
            session_id += 1
        
        current_start = current_end
    
    return chunks

# ========== Prompt Design ==========

def make_prompt(summary: str, question: str) -> str:
    """Create a unified prompt (open-domain QA format)."""
    prompt = f"""You are a careful multi-hop reading assistant. 
Use the given Context. 
Answer with ONLY the final answer string; no extra words.

Question:
{question}

Context:
{summary}

Answer:
"""
    return prompt

# ========== Answer Extraction and Evaluation ==========
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def _calculate_f1(pred_answer: str, gold_answers: List[str]) -> float:
    # Compute F1 against each gold answer and take the max
    max_f1 = 0.0
    for gold_answer in gold_answers:
        max_f1 = max(max_f1, qa_f1_score(pred_answer, gold_answer))
    return max_f1

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
    max_tokens: int = 2000,
    embedding_model_path: Optional[str] = None,
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
    sample_id = sample.get("_id", f"sample-{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"Processing sample #{sample_index}: {sample_id}")
    print(f"{'='*60}")
    
    try:
        # 1. Build context chunks
        context_chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_model_path)
        print(f"Context chunks: {len(context_chunks)}")
        if context_chunks:
            print(f"First chunk preview:\n{context_chunks[0][:400]}...")
        
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
        
        # 4. Build memory with MemoryAgent (each context chunk as one message)
        print("\nStep 2: Build memory with MemoryAgent")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator,
        )

        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            for i, context_chunk in enumerate(context_chunks, 1):
                print(f"  Processing chunk {i}/{len(context_chunks)}...")
                memory_update = memory_agent.memorize(context_chunk) 
        
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
        
        # 5. Create retrievers (for ResearchAgent)
        print("\nStep 3: Create retrievers (for ResearchAgent)")
        retrievers = {}
        
        # Index retriever
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
            # If index dir exists, remove it to avoid "Directory not empty"
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
            # If index dir exists, remove it to avoid "Directory not empty"
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
            # If index dir exists, remove it to avoid "Directory not empty"
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
        
        # 6. Create Research Generator and Working Generator
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
        
        # 7. Create ResearchAgent
        print("\nStep 5: Create ResearchAgent")
        research_agent = ResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=research_generator,
            max_iters=3
        )
        print("[OK] ResearchAgent created")
        
        # 8. Run QA
        print("\nStep 6: Run QA")
        
        # Extract question info
        question = sample.get("input", "")
        gold_answers = sample.get("answers", [])
        
        print(f"Question: {question}")
        print(f"Gold answers: {gold_answers}")
        
        # Save metadata
        result = {
            "_id": sample.get("_id", sample_id),
            "sample_id": sample_id,
            "index": sample.get("index", sample_index),
            "question": question,
            "answers": gold_answers,
            "gold_answers": gold_answers,  # keep for compatibility
        }

        try:
            # Run research via ResearchAgent
            print("Running deep research...")
            research_result = research_agent.research(question)
            research_summary = research_result.integrated_memory
            print(f"[OK] Research complete. Iterations: {len(research_result.raw_memory.get('iterations', []))}")
            print(f"Research summary: {research_summary[:200]}...")
            
            # Save research trace
            research_trace = {
                "question": question,
                "raw_memory": research_result.raw_memory,
                "integrated_memory": research_result.integrated_memory,
                "iterations": research_result.raw_memory.get("iterations", []),
                "search_plans": research_result.raw_memory.get("search_plans", []),
                "reflections": research_result.raw_memory.get("reflections", [])
            }
            
            trace_file = os.path.join(sample_results_dir, "research_trace.json")
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(research_trace, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Research trace saved: {trace_file}")
            
            result["research_summary"] = research_summary
            result["research_trace_file"] = trace_file
            
            # Generate answer with unified prompt
            print("Generating answer...")
            prompt = make_prompt(research_summary, question)
            response = working_generator.generate_single(prompt=prompt)
            answer_text = response.get("text", "").strip()
            
            print(f"Model response: {answer_text[:200]}...")
            
            # Extract answer
            pred_answer = answer_text
            result["response"] = answer_text
            result["pred"] = pred_answer
            
            # Compute F1 score
            f1_score = _calculate_f1(pred_answer, gold_answers) if pred_answer else 0.0
            result["f1"] = f1_score
            
            print(f"Predicted answer: {pred_answer}")
            print(f"Gold answers: {gold_answers}")
            print(f"F1 score: {f1_score:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process question: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        # Save result
        results_file = os.path.join(sample_results_dir, "qa_result.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Results saved to: {results_file}")
        
        # Summary
        print(f"\n{'='*60}")
        print("Processing summary")
        print(f"{'='*60}")
        print(f"Sample ID: {sample_id}")
        print(f"Context chunks: {len(context_chunks)}")
        if final_state:
            print(f"Memory abstracts: {len(final_state.abstracts)}")
        print(f"Predicted answer: {result.get('pred', 'N/A')}")
        print(f"Gold answers: {gold_answers}")
        print(f"F1 score: {result.get('f1', 0.0):.4f}")
        print(f"Results saved to: {sample_results_dir}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_index}: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "sample_id": sample.get("_id", f"sample-{sample_index}"),
            "error": error_msg
        }


# ========== Main ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JITMind + HotpotQA evaluation")
    parser.add_argument("--data", type=str, default="/path/to/hotpotqa/eval.json", 
                        help="HotpotQA dataset path")
    parser.add_argument("--outdir", type=str, default="./results/hotpotqa",
                        help="Output directory")
    parser.add_argument("--start-idx", type=int, default=0, help="Start sample index")
    parser.add_argument("--end-idx", type=int, default=None, help="End sample index (exclusive); None for all")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per context chunk")
    parser.add_argument("--embedding-model-path", type=str, default=None, 
                        help="Embedding model path for precise token count (optional)")
    
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
    print("JITMind + HotpotQA evaluation")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Output directory: {args.outdir}")
    print(f"Sample range: {args.start_idx} to {args.end_idx-1 if args.end_idx else 'all'}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 60)
    
    # Load data
    all_samples = load_hotpotqa(args.data)
    print(f"Loaded {len(all_samples)} samples")
    
    # Reset end index after load
    if args.end_idx is None:
        args.end_idx = len(all_samples)
    
    print(f"Actual range: {args.start_idx} to {args.end_idx-1} (total {args.end_idx - args.start_idx})")
    
    # Validate index range
    if args.start_idx < 0 or args.start_idx >= len(all_samples):
        print(f"Error: start index {args.start_idx} out of range (total {len(all_samples)})")
        return
    
    if args.end_idx > len(all_samples):
        print(f"Warning: end index {args.end_idx} out of range; adjusted to {len(all_samples)}")
        args.end_idx = len(all_samples)
    
    if args.start_idx >= args.end_idx:
        print(f"Error: start index {args.start_idx} must be less than end index {args.end_idx}")
        return
    
    # Process samples sequentially
    sample_indices = list(range(args.start_idx, args.end_idx))
    
    print("Starting sequential processing...")
    
    all_results = []
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        sample = all_samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"Processing sample {sample_idx}/{len(all_samples)-1} (range: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            result = process_sample(
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
                max_tokens=args.max_tokens,
                embedding_model_path=args.embedding_model_path,
                use_schema=args.use_schema,
                memory_api_type=args.memory_api_type,
                research_api_type=args.research_api_type,
                working_api_type=args.working_api_type
            )
            print(f"[OK] Sample {sample_idx} processed")
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Sample {sample_idx} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "sample_id": sample.get("_id", f"sample-{sample_idx}"),
                "error": str(e)
            })
    
    # Aggregate results
    f1_scores = []
    
    for result in all_results:
        if "f1" in result:
            f1_scores.append(result["f1"])
    
    # Save summary
    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Batch summary saved: {summary_file}")
        
        # Compute average F1
        if len(f1_scores) > 0:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            total_samples = args.end_idx - args.start_idx
            success_count = len(f1_scores)
            
            # Build statistics
            statistics = {
                "total_samples": total_samples,
                "success_count": success_count,
                "failed_count": total_samples - success_count,
                "success_rate": success_count / total_samples if total_samples > 0 else 0.0,
                "avg_f1": avg_f1,
                "f1_scores": f1_scores,
                "start_idx": args.start_idx,
                "end_idx": args.end_idx - 1
            }
            
            # Save statistics
            stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            print(f"[OK] Batch statistics saved: {stats_file}")
            
            # Print statistics
            print(f"\n{'='*60}")
            print("Batch statistics")
            print(f"{'='*60}")
            print(f"Samples processed: {total_samples}")
            print(f"Successful answers: {success_count}")
            print(f"Failed answers: {total_samples - success_count}")
            print(f"Success rate: {statistics['success_rate']:.2%}")
            print(f"Average F1: {avg_f1:.4f}")
            print(f"{'='*60}")

if __name__ == "__main__":
    main()
