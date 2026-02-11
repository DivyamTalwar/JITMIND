#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JITMind + RULER evaluation script.

RULER JSONL format (per line):
- context: str (long context to memorize)
- example: str (optional)
- instruction: str (optional)
- question: str
- outputs: List[str] (gold answers)

Flow:
1. MemoryAgent memorizes context
2. ResearchAgent performs research
3. Combine example + question for answering
4. Compute accuracy (Acc)
"""

import sys
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import Counter
import string
import glob


from jitmind import (
    MemoryAgent,
    ResearchAgent,
    VLLMGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGeneratorConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

# ========== Dataset System Prompts ==========

def get_dataset_system_prompt(dataset_name: str) -> str:
    """
    Return the system_prompt for a given dataset name.
    
    Args:
        dataset_name: dataset name (e.g., 'vt', 'qa_1', 'niah_single_1')
    
    Returns:
        system_prompt string
    """
    # Define system_prompt by dataset name
    system_prompts = {
        # Variable Tracking (VT)
        "vt": "Memorize and track the chain(s) of variable assignment hidden in the following text.",
        
        # Question Answering (QA)
        "qa_1": "",
        "qa_2": "",
        
        # Needle in a Haystack (NIAH) - Single
        "niah_single_1": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_2": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_3": "A special magic uuid is hidden within the following text. Make sure to memorize it. I will quiz you about the uuid afterwards.",
        
        # Needle in a Haystack (NIAH) - Multi-value
        "niah_multivalue": "",
        
        # Needle in a Haystack (NIAH) - Multi-query
        "niah_multiquery": "Some special magic numbers are hidden within the following text. You only need to memorize the special magic numbers. I will quiz you about the numbers afterwards.",
        
        # Needle in a Haystack (NIAH) - Multi-key
        "niah_multikey_1": "",
        "niah_multikey_2": "",
        "niah_multikey_3": "",
        
        # Context Window Extension (CWE)
        "cwe": "Below is a numbered list of words. You only need to memorize the numbers that all words appear rather then make a abstract. I will quiz you about the numbers afterwards. Ignore the prompt below that asks you to summarize.",
        
        # Full Window Extension (FWE)
        "fwe": "Read the following coded text and track the frequency of each coded word. Memorize the numbers that the words appear, I will quiz you about the numbers afterwards.",
    }
    
    # Extract base dataset name (remove numeric suffix)
    base_name = dataset_name.split('_')[0] if '_' in dataset_name else dataset_name
    
    # Try exact match
    if dataset_name in system_prompts:
        return system_prompts[dataset_name]
    
    # Try partial match
    for key, prompt in system_prompts.items():
        if dataset_name.startswith(key) or key in dataset_name:
            return prompt
    
    # Default system_prompt
    return ""

# ========== Data Loading ==========

def load_ruler_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load RULER JSONL dataset.
    
    Args:
        jsonl_path: JSONL file path
    
    Returns:
        list of samples
    """
    data_list = []
    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    item['_id'] = f"{dataset_name}-{idx}"
                    item['index'] = idx
                    item['dataset'] = dataset_name
                    data_list.append(item)
                except Exception as e:
                    print(f"Warning: Failed to parse line {idx} in {jsonl_path}: {e}")
                    continue
    
    return data_list

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

def build_question_prompt(sample: Dict[str, Any]) -> str:
    """
    Build the question prompt by combining example and question.
    
    Args:
        sample: sample with 'example' and 'question'
    
    Returns:
        full question prompt
    """
    parts = []

    # 2. Question
    question = sample.get("question", "").strip()
    question_prompt = "Question:\n" + question
    if question:
        parts.append(question_prompt)


    # 1. Example (if present)
    example = sample.get("example", "").strip()
    if example:
        example_prompt = "Here is the example:\n" + example
        parts.append(example_prompt)
    
    # Concatenate all parts
    prompt = "\n\n".join(parts)
    
    return prompt

# ========== Answer Evaluation ==========

def normalize_text(text: str) -> str:
    """
    Normalize text: remove punctuation, lowercase, normalize spaces.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize multiple spaces to single
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def evaluate_answer(model_response: str, ground_truth_outputs: List[str]) -> bool:
    """
    Evaluate whether the model answer is correct.
    
    Rule: if the model response contains all elements in ground_truth_outputs, it is correct.
    Strategies:
    1. Exact match: search for gold answer in lowercase response
    2. Flexible match: match after punctuation removal
    3. Keyword match: for multi-word answers, check all keywords exist
    
    Args:
        model_response: model answer
        ground_truth_outputs: gold answers
    
    Returns:
        True/False correctness
    """
    if not ground_truth_outputs:
        return False
    
    if not model_response:
        return False
    
    # Lowercase for comparison
    model_response_lower = model_response.lower()
    
    # Normalized text (punctuation removed)
    model_response_normalized = normalize_text(model_response)
    
    # Deduplicate gold answers to avoid repeated checks
    unique_answers = list(set(ground_truth_outputs))
    
    for answer in unique_answers:
        answer_str = str(answer).strip()
        if not answer_str:
            continue
            
        answer_lower = answer_str.lower()
        
        # Strategy 1: exact match (lowercased substring)
        if answer_lower in model_response_lower:
            continue
        
        # Strategy 2: flexible match (punctuation removed)
        answer_normalized = normalize_text(answer_str)
        if answer_normalized in model_response_normalized:
            continue
        
        # Strategy 3: keyword match (all keywords present)
        # Extract keywords (length > 2)
        answer_words = [w for w in answer_normalized.split() if len(w) > 2]
        if answer_words:
            # Check all keywords in model response
            if all(word in model_response_normalized for word in answer_words):
                continue
        
        # If all strategies fail, answer does not match
        return False
    
    # All answers matched
    return True

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
    max_tokens: int = 2048,
    embedding_model_path: Optional[str] = None,
    use_schema: bool = False,
    memory_api_type: str = "openai",
    research_api_type: str = "openai",
    working_api_type: str = "openai"
):
    """
    Process a single sample with JITMind.

    Flow:
    1. Build memory with MemoryAgent (memorize context)
    2. Run deep research with ResearchAgent
    3. Answer based on research result (example + question)
    """
    sample_id = sample.get("_id", f"sample-{sample_index}")
    dataset_name = sample.get("dataset", "unknown")
    
    print(f"\n{'='*60}")
    print(f"Processing sample #{sample_index}: {sample_id} (dataset: {dataset_name})")
    print(f"{'='*60}")
    
    try:
        # 1. Build context chunks (from context field)
        context_chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_model_path)
        print(f"Context chunks: {len(context_chunks)}")
        if context_chunks:
            print(f"First chunk preview:\n{context_chunks[0][:400]}...")
        
        # Create output directory
        sample_results_dir = os.path.join(outdir, dataset_name, sample_id)
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
        
        # 4. Get dataset-specific system_prompt
        memory_system_prompt = get_dataset_system_prompt(dataset_name)
        print(f"\nDataset system prompt: {memory_system_prompt[:100]}...")
        
        # 5. Build memory with MemoryAgent (each context chunk as one message)
        print("\nStep 2: Build memory with MemoryAgent")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator,
            system_prompts={"memory": memory_system_prompt}
        )
        
        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            for i, context_chunk in enumerate(context_chunks, 1):
                print(f"  Processing chunk {i}/{len(context_chunks)}...")
                memory_update = memory_agent.memorize(context_chunk)
        else:
            print("  Memory already exists, skipping build")
        
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
        
        # 6. Create retrievers (for ResearchAgent)
        print("\nStep 3: Create retrievers (for ResearchAgent)")
        retrievers = {}
        
        # Index retriever
        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")
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
        
        # 7. Create Research Generator and Working Generator
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
        
        # 8. Create ResearchAgent
        print("\nStep 5: Create ResearchAgent")
        
        # Set dataset-specific system_prompts
        system_prompts = None
        if dataset_name == "niah_multivalue":
            system_prompts = {
                "planning": "There are 4 different special magic numbers for the question item. So the keyword retrieval is need.",
                "integration": "There are 4 different special magic numbers for the question item. Don't miss any of them.",
                "reflection": "There are 4 different special magic numbers for the question item. Don't miss any of them."
            }
            print(f"[INFO] Custom system_prompts set for dataset {dataset_name}")
        
        # Build ResearchAgent params
        research_agent_kwargs = {
            "page_store": page_store,
            "memory_store": memory_store,
            "retrievers": retrievers,
            "generator": research_generator,
            "max_iters": 5
        }
        
        # Add system_prompts if provided
        if system_prompts is not None:
            research_agent_kwargs["system_prompts"] = system_prompts
        
        research_agent = ResearchAgent(**research_agent_kwargs)
        print("[OK] ResearchAgent created")
        
        # 9. Run QA
        print("\nStep 6: Run QA")
        
        # Use question only (exclude example)
        question = sample.get("question", "").strip()
        ground_truth_outputs = sample.get("outputs", [])
        
        # Build full question prompt (example + question) for final answer
        question_prompt = build_question_prompt(sample)
        
        print(f"Question: {question[:200]}...")
        print(f"Gold answers: {ground_truth_outputs}")
        
        # Save metadata
        result = {
            "_id": sample.get("_id", sample_id),
            "sample_id": sample_id,
            "index": sample.get("index", sample_index),
            "dataset": dataset_name,
            "example": sample.get("example", ""),
            "instruction": sample.get("instruction", ""),
            "question": question,
            "question_prompt": question_prompt,
            "ground_truth_outputs": ground_truth_outputs,
        }
        
        try:
            # Run research via ResearchAgent (question only)
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
            
            # Generate answer using question_prompt (example + question)
            print("Generating answer...")
            prompt = f"""Read the text below and answer a question. Context: {research_summary}\n\n{question_prompt}\n\nAnswer:"""
            response = working_generator.generate_single(prompt=prompt)
            answer_text = response.get("text", "").strip()
            
            print(f"Model response: {answer_text[:200]}...")
            
            result["response"] = answer_text
            
            # Evaluate answer correctness
            is_correct = evaluate_answer(answer_text, ground_truth_outputs)
            result["is_correct"] = is_correct
            result["accuracy"] = 1.0 if is_correct else 0.0
            
            print(f"Predicted answer: {answer_text[:200]}...")
            print(f"Gold answers: {ground_truth_outputs}")
            print(f"Evaluation: {'✓ correct' if is_correct else '✗ incorrect'}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process question: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            result["is_correct"] = False
            result["accuracy"] = 0.0
        
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
        print(f"Dataset: {dataset_name}")
        print(f"Context chunks: {len(context_chunks)}")
        if final_state:
            print(f"Memory abstracts: {len(final_state.abstracts)}")
        print(f"Predicted answer: {result.get('response', 'N/A')[:200]}...")
        print(f"Gold answers: {ground_truth_outputs}")
        print(f"Accuracy: {result.get('accuracy', 0.0):.4f}")
        print(f"Results saved to: {sample_results_dir}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_index}: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "sample_id": sample.get("_id", f"sample-{sample_index}"),
            "dataset": dataset_name,
            "error": error_msg,
            "is_correct": False,
            "accuracy": 0.0
        }

# ========== Main ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="JITMind + RULER evaluation")
    parser.add_argument("--data", type=str, 
                        default="/path/to/ruler/data",
                        help="RULER JSONL file path or directory")
    parser.add_argument("--outdir", type=str, 
                        default="./results/ruler",
                        help="Output directory")
    parser.add_argument("--start-idx", type=int, default=0, 
                        help="Start sample index")
    parser.add_argument("--end-idx", type=int, default=None, 
                        help="End sample index (exclusive); None for all")
    parser.add_argument("--max-tokens", type=int, default=2048, 
                        help="Max tokens per context chunk")
    parser.add_argument("--embedding-model-path", type=str, 
                        default=None, 
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
    print("JITMind + RULER evaluation")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output directory: {args.outdir}")
    print(f"Sample range: {args.start_idx} to {args.end_idx-1 if args.end_idx else 'all'}")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 60)
    
    # Determine files to process
    jsonl_files = []
    if os.path.isfile(args.data):
        # Single file
        jsonl_files = [args.data]
    elif os.path.isdir(args.data):
        # Directory: find all .jsonl files
        jsonl_files = sorted(glob.glob(os.path.join(args.data, "*.jsonl")))
    else:
        print(f"Error: path not found: {args.data}")
        return
    
    if not jsonl_files:
        print(f"Error: no .jsonl files found in {args.data}")
        return
    
    print(f"\nFound {len(jsonl_files)} data files:")
    for f in jsonl_files:
        print(f"  - {f}")
    
    # Process each data file
    all_results = []
    
    for jsonl_file in jsonl_files:
        dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load data
        all_samples = load_ruler_jsonl(jsonl_file)
        print(f"Loaded {len(all_samples)} samples")
        
        # Determine sample range
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(all_samples)
        end_idx = min(end_idx, len(all_samples))
        
        if start_idx >= end_idx:
            print(f"Warning: invalid sample range, skipping dataset {dataset_name}")
            continue
        
        print(f"Sample range: {start_idx} to {end_idx-1} (total {end_idx - start_idx})")
        
        # Process samples sequentially
        sample_indices = list(range(start_idx, end_idx))
        
        print("\nStarting sequential processing...")
        results = []
        for idx in tqdm(sample_indices, desc=f"Processing {dataset_name}"):
            sample = all_samples[idx]
            try:
                result = process_sample(
                    sample,
                    idx,
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
                    use_schema=args.use_schema
                )
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Sample {idx} failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "_id": sample.get("_id", f"sample-{idx}"),
                    "index": idx,
                    "dataset": dataset_name,
                    "error": str(e),
                    "is_correct": False,
                    "accuracy": 0.0
                })
        
        all_results.extend(results)
        
        # Compute dataset accuracy
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_count = len(results)
        dataset_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"{dataset_name} dataset summary")
        print(f"{'='*60}")
        print(f"Total samples: {total_count}")
        print(f"Correct: {correct_count}")
        print(f"Incorrect: {total_count - correct_count}")
        print(f"Accuracy: {dataset_accuracy:.4f} ({dataset_accuracy*100:.2f}%)")
        print(f"{'='*60}")
        
        # Save dataset summary
        dataset_summary = {
            "dataset": dataset_name,
            "total_samples": total_count,
            "correct_count": correct_count,
            "wrong_count": total_count - correct_count,
            "accuracy": dataset_accuracy,
            "results": results
        }
        
        dataset_summary_file = os.path.join(
            args.outdir, 
            dataset_name, 
            f"summary_{start_idx}_{end_idx-1}.json"
        )
        os.makedirs(os.path.dirname(dataset_summary_file), exist_ok=True)
        with open(dataset_summary_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] {dataset_name} summary saved: {dataset_summary_file}")
    
    # Save overall summary
    if all_results:
        # Compute overall accuracy
        total_correct = sum(1 for r in all_results if r.get("is_correct", False))
        total_samples = len(all_results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Group stats by dataset
        dataset_stats = {}
        for result in all_results:
            dataset = result.get("dataset", "unknown")
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {
                    "total": 0,
                    "correct": 0,
                    "wrong": 0
                }
            dataset_stats[dataset]["total"] += 1
            if result.get("is_correct", False):
                dataset_stats[dataset]["correct"] += 1
            else:
                dataset_stats[dataset]["wrong"] += 1
        
        # Compute accuracy per dataset
        for dataset in dataset_stats:
            total = dataset_stats[dataset]["total"]
            correct = dataset_stats[dataset]["correct"]
            dataset_stats[dataset]["accuracy"] = correct / total if total > 0 else 0.0
        
        # Save overall summary
        overall_summary = {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "total_wrong": total_samples - total_correct,
            "overall_accuracy": overall_accuracy,
            "dataset_stats": dataset_stats,
            "results": all_results
        }
        
        overall_summary_file = os.path.join(
            args.outdir, 
            f"overall_summary_{args.start_idx}_{args.end_idx if args.end_idx else 'all'}.json"
        )
        with open(overall_summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Overall summary saved: {overall_summary_file}")
        
        # Print statistics
        print(f"\n{'='*60}")
        print("Evaluation summary")
        print(f"{'='*60}")
        print(f"Datasets processed: {len(jsonl_files)}")
        print(f"Total samples: {total_samples}")
        print(f"Correct: {total_correct}")
        print(f"Incorrect: {total_samples - total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"\nPer-dataset accuracy:")
        for dataset, stats in sorted(dataset_stats.items()):
            print(f"  {dataset}: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%) - {stats['correct']}/{stats['total']}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
