# Examples Directory

This folder contains runnable usage examples for onboarding and integration testing.

## Purpose

`examples/` is intended to answer one question fast:

How do I run JITMIND end-to-end with minimal code?

Current content:

- `quickstart/`

## Quickstart Track

Inside `quickstart/` you will find:

- `basic_usage.py`
  - memory construction and research flow
- `model_usage.py`
  - generator/model backend configuration patterns
- `ttl_usage.py`
  - TTL memory/page behavior examples
- `README.md`
  - per-example details and commands

## Run Path

From repository root:

```bash
python3 examples/quickstart/basic_usage.py
python3 examples/quickstart/model_usage.py
python3 examples/quickstart/ttl_usage.py
```

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install openai cohere neo4j faiss-cpu numpy pydantic tqdm scikit-learn pytest
```

Optional sparse retrieval dependency:

```bash
pip install pyserini
```

Environment variables commonly needed:

```bash
export OPENROUTER_API_KEY=\"...\"
export OPENROUTER_BASE_URL=\"https://openrouter.ai/api/v1\"
export COHERE_API_KEY=\"...\"
```

## Notes

1. Examples are practical onboarding scripts, not full benchmark harnesses.
2. Some features are optional and depend on installed dependencies (e.g., Neo4j, Pyserini).
3. For architecture-level explanation, read:
   - `README.md` (root)
   - `jitmind/README.md` (package handbook)

