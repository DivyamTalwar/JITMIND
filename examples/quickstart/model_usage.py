"""
Examples of using different model backends with JITMind (OpenRouter API or local VLLM).
"""

import os
from jitmind import (
    MemoryAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
    AdvancedMemoryStore,
    InMemoryPageStore,
)


def openrouter_example():
    """OpenRouter API model usage example."""
    print("=== OpenRouter API Example ===\n")

    # 1. Configure OpenRouter generator
    gen_config = OpenAIGeneratorConfig(
        model_name="google/gemini-3-flash-preview",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
    )

    # 2. Create generator
    generator = OpenAIGenerator.from_config(gen_config)

    # 3. Create stores
    memory_store = AdvancedMemoryStore()
    page_store = InMemoryPageStore()

    # 4. Create MemoryAgent
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store,
    )

    # 5. Simple documents
    documents = [
        "Machine learning is a subset of AI.",
        "Deep learning uses multi-layer neural networks.",
        "NLP focuses on understanding human language.",
    ]

    print(f"Processing {len(documents)} documents...")
    for doc in documents:
        memory_agent.memorize(doc)

    memory_state = memory_store.load()
    print(f"✅ Built {len(memory_state.abstracts)} memory abstracts\n")


def custom_endpoint_example():
    """Custom OpenAI-compatible endpoint example."""
    print("=== Custom API Endpoint Example ===\n")

    # 1. Configure custom endpoint
    gen_config = OpenAIGeneratorConfig(
        model_name="your-model-name",
        api_key="your-api-key",
        base_url="https://your-custom-endpoint.com/v1",
        temperature=0.3,
    )

    # 2. Create generator
    _ = OpenAIGenerator.from_config(gen_config)

    # 3. Create stores
    _ = AdvancedMemoryStore()
    _ = InMemoryPageStore()

    # 4. Create MemoryAgent (not used here)
    print("✅ Custom endpoint configured")
    print(f"   Endpoint: {gen_config.base_url}")
    print(f"   Model: {gen_config.model_name}\n")


def vllm_example():
    """Local VLLM model usage example."""
    print("=== VLLM Local Model Example ===\n")

    try:
        # 1. Configure VLLM generator
        gen_config = VLLMGeneratorConfig(
            model_name="Qwen2.5-7B-Instruct",
            api_key="empty",
            base_url="http://localhost:8000/v1",
            temperature=0.7,
        )

        # 2. Create generator
        generator = VLLMGenerator.from_config(gen_config)

        # 3. Create stores
        memory_store = AdvancedMemoryStore()
        page_store = InMemoryPageStore()

        # 4. Create MemoryAgent
        memory_agent = MemoryAgent(
            generator=generator,
            memory_store=memory_store,
            page_store=page_store,
        )

        # 5. Simple documents
        documents = [
            "Machine learning is a subset of AI.",
            "Deep learning uses multi-layer neural networks.",
            "NLP focuses on understanding human language.",
        ]

        print(f"Processing {len(documents)} documents...")
        for doc in documents:
            memory_agent.memorize(doc)

        memory_state = memory_store.load()
        print(f"✅ Built {len(memory_state.abstracts)} memory abstracts\n")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install vllm>=0.6.0")
    except Exception as e:
        print(f"❌ Local model error: {e}")
        print("   Tip: if memory is limited, try a smaller model")


def model_comparison():
    """Model comparison overview."""
    print("\n=== Model Selection Guide ===\n")

    print("OpenRouter API models:")
    print("  Pros:")
    print("    - Fast to start, no local resources")
    print("    - Strong accuracy and performance")
    print("    - Managed updates and maintenance")
    print("  Cons:")
    print("    - Requires network access")
    print("    - Usage-based billing")
    print("    - Data sent to external servers")

    print("\nVLLM local models:")
    print("  Pros:")
    print("    - Fully offline")
    print("    - Privacy-preserving")
    print("    - No usage limits")
    print("  Cons:")
    print("    - Requires GPU resources")
    print("    - Must download and manage models")
    print("    - More setup and tuning")

    print("\nRecommendation:")
    print("  - Rapid prototyping: OpenRouter API")
    print("  - Privacy / offline: local VLLM")
    print("  - Large-scale usage: balance cost vs performance")


def main():
    """Main entry."""
    print("JITMind Model Usage Examples")

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not found")
        print("   Some examples will not run")
        print("   Set with: export OPENROUTER_API_KEY='your-api-key'\n")

    # OpenRouter API example
    if os.getenv("OPENROUTER_API_KEY"):
        try:
            openrouter_example()
        except Exception as e:
            print(f"OpenRouter API example failed: {e}\n")
    else:
        print("Skipping OpenRouter example (API key not set)\n")

    # Custom endpoint example (config only)
    custom_endpoint_example()

    # Optional VLLM test
    print("Test local VLLM model?")
    test_vllm = input("Type 'yes' to continue, or press Enter to skip: ").strip().lower()
    if test_vllm == "yes":
        vllm_example()
    else:
        print("Skipping VLLM example\n")

    # Model comparison
    model_comparison()

    print("\nSummary")
    if os.getenv("OPENROUTER_API_KEY"):
        print("✅ OpenRouter API: great for rapid prototyping and deployment")
    print("✅ Custom endpoints: useful for OpenAI-compatible providers")
    print("✅ VLLM local models: good for privacy/offline")

    print("\nMore info:")
    print("  - See eval/ for evaluation examples")
    print("  - See jitmind/generator/ for generator implementations")


if __name__ == "__main__":
    main()
