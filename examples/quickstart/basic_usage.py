"""
Basic JITMind quickstart example.
Demonstrates memory construction, retrieval, and research end-to-end.
"""

import os
from jitmind import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    AdvancedMemoryStore,
    InMemoryPageStore,
    CohereDenseRetriever,
    CohereEmbedRetrieverConfig,
    CohereReranker,
    CohereRerankerConfig,
    BM25Retriever,
    BM25RetrieverConfig,
    IndexRetriever,
    IndexRetrieverConfig,
)


def basic_memory_example():
    """Basic memory construction example."""
    print("=== Basic Memory Construction ===\n")

    # 1. Configure and create generator
    gen_config = OpenAIGeneratorConfig(
        model_name="google/gemini-3-flash-preview",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
    )
    generator = OpenAIGenerator.from_config(gen_config)

    # 2. Create stores
    memory_store = AdvancedMemoryStore()
    page_store = InMemoryPageStore()

    # 3. Create MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store,
    )

    # 4. Documents to memorize (simulate long text)
    documents = [
        """Artificial intelligence (AI) is a branch of computer science focused on building systems that can perform tasks
        that typically require human intelligence. Machine learning is a subset of AI that enables systems to learn from data
        without being explicitly programmed.""",
        """Deep learning is a subset of machine learning that uses multi-layer neural networks. Natural language processing (NLP)
        is another key area of AI that focuses on understanding, interpreting, and generating human language.""",
        """Computer vision is a major AI field aimed at enabling machines to see and interpret visual information.
        Reinforcement learning is a machine learning approach that learns optimal behavior by interacting with an environment.""",
        """Neural networks are the foundation of deep learning and are composed of interconnected nodes (neurons).
        Convolutional neural networks (CNNs) work well for images, while recurrent neural networks (RNNs) are good for sequences.""",
        """The Transformer architecture transformed NLP and enabled large language models such as GPT and BERT.""",
    ]

    # 5. Memorize documents
    print(f"Memorizing {len(documents)} documents...")
    for i, doc in enumerate(documents, 1):
        print(f"  Memorizing document {i}/{len(documents)}...")
        memory_agent.memorize(doc)

    # 6. View memory state
    memory_state = memory_store.load()
    print("\n✅ Memory built successfully:")
    print(f"  - Memory abstracts: {len(memory_state.abstracts)}")

    return memory_store, page_store


def memory_research_example(memory_store, page_store):
    """Research example grounded in memory."""
    print("\n=== Memory-Grounded Research ===\n")

    # 1. Configure and create generator
    gen_config = OpenAIGeneratorConfig(
        model_name="google/gemini-3-flash-preview",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=2048,
    )
    generator = OpenAIGenerator.from_config(gen_config)

    # 2. Create retrievers
    retrievers = {}

    # Index retriever
    try:
        index_config = IndexRetrieverConfig(index_dir="./index/index")
        index_retriever = IndexRetriever(index_config.__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
        print("✅ Index retriever created")
    except Exception as e:
        print(f"[WARN] Failed to create index retriever: {e}")

    # BM25 retriever
    try:
        bm25_config = BM25RetrieverConfig(index_dir="./index/bm25")
        bm25_retriever = BM25Retriever(bm25_config.__dict__)
        bm25_retriever.build(page_store)
        retrievers["keyword"] = bm25_retriever
        print("✅ BM25 retriever created")
    except Exception as e:
        print(f"[WARN] Failed to create BM25 retriever: {e}")

    # Cohere dense retriever
    try:
        dense_config = CohereEmbedRetrieverConfig(
            index_dir="./index/cohere_dense",
            api_key=os.getenv("COHERE_API_KEY"),
        )
        dense_retriever = CohereDenseRetriever(dense_config.__dict__)
        dense_retriever.build(page_store)
        retrievers["vector"] = dense_retriever
        print("✅ Cohere dense retriever created")
    except Exception as e:
        print(f"[WARN] Failed to create Cohere dense retriever: {e}")

    # Optional reranker
    reranker = None
    try:
        rerank_config = CohereRerankerConfig(api_key=os.getenv("COHERE_API_KEY"))
        reranker = CohereReranker(rerank_config.__dict__)
        print("✅ Cohere reranker created")
    except Exception as e:
        print(f"[WARN] Failed to create Cohere reranker: {e}")

    # 3. Create ResearchAgent
    research_agent = ResearchAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store,
        retrievers=retrievers,
        reranker=reranker,
        max_iters=5,
    )

    # 4. Research
    question = "What are the key differences between machine learning and deep learning?"
    print(f"\nResearch question: {question}\n")
    research_result = research_agent.research(question, memory_state=memory_store.load())

    # 5. Output
    print("✅ Research completed:")
    print(f"  - Iterations: {len(research_result.raw_memory.get('iterations', []))}")
    print("\nResearch summary:")
    print(research_result.integrated_memory)


def main():
    """Main entry."""
    print("JITMind Quickstart Example")

    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Please set OPENROUTER_API_KEY")
        return
    if not os.getenv("COHERE_API_KEY"):
        print("⚠️  Please set COHERE_API_KEY")
        return

    try:
        # 1) Build memory
        memory_store, page_store = basic_memory_example()
        # 2) Research using memory
        memory_research_example(memory_store, page_store)
        print("\n✅ Example completed successfully")
        print("\nTips:")
        print("  - Modify the document content to test other scenarios")
        print("  - Try different questions to stress retrieval")
        print("  - Check eval/ for evaluation examples")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("  1. Network connectivity")
        print("  2. API keys")
        print("  3. Dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
