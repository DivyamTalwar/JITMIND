#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker compose -f docker-compose.neo4j.yml up -d
echo "Neo4j is up:"
echo "  http://localhost:7474"
echo "  bolt://localhost:7687"

