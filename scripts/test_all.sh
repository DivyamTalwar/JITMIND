#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/2] Unit tests"
python3 -m pytest -q

echo
echo "[2/2] Live E2E (optional)"
if [[ -n "${OPENROUTER_API_KEY:-}" && -n "${COHERE_API_KEY:-}" ]]; then
  python3 scripts/e2e_live_test.py
else
  echo "Skipping live E2E: OPENROUTER_API_KEY and/or COHERE_API_KEY not set."
fi

