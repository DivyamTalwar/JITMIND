"""Persistent exact cosine index with opt-in trusted embedding callbacks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

from jitmind.storage.models import canonical_json, validate_identifier

from .base import SQLiteProjection
from .models import ProjectionError, ProjectionHit, QueryResult, integer


@dataclass(frozen=True)
class EmbeddingIdentity:
    model: str
    version: str
    dimensions: int
    normalization: str = "l2"

    def validated(self):
        validate_identifier(self.model)
        validate_identifier(self.version)
        integer(self.dimensions, 1, 4096)
        if self.normalization != "l2":
            raise ProjectionError("invalid_normalization")
        return self


class Embedder(Protocol):
    identity: EmbeddingIdentity

    def embed(self, text: str) -> list[float]: ...


class FeatureHashingEmbedder:
    """Deterministic lexical feature hashing, NOT a learned semantic model."""

    def __init__(self, dimensions=256):
        self.identity = EmbeddingIdentity(
            "lexical-feature-hashing", "sha256-word-v1", dimensions
        ).validated()

    def embed(self, text):
        canonical_json(text, max_bytes=131072)
        vector = [0.0] * self.identity.dimensions
        for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
            hashed = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(hashed[:8], "big") % len(vector)] += 1.0
        return vector


def normalized(raw, dimensions):
    if type(raw) not in (list, tuple) or len(raw) != dimensions:
        raise ProjectionError("embedding_dimension")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in raw):
        raise ProjectionError("embedding_nonfinite")
    # Scaling first avoids overflow even for finite huge callback components.
    scale = max(map(abs, raw), default=0)
    if scale == 0:
        return [0.0] * dimensions
    scaled = [v / scale for v in raw]
    norm = math.sqrt(sum(v * v for v in scaled))
    return [v / norm for v in scaled]


class SQLiteVectorProjection(SQLiteProjection):
    kind = "vector"
    app_id = 0x4A565031
    extra_schema = (
        "CREATE TABLE vectors (namespace TEXT NOT NULL, fact TEXT NOT NULL, vector TEXT NOT NULL, PRIMARY KEY(namespace,fact))",
    )

    def __init__(self, path, *, source_id, embedder=None, fault_hook=None):
        self.embedder = embedder if embedder is not None else FeatureHashingEmbedder()
        if type(self.embedder.identity) is not EmbeddingIdentity:
            raise ProjectionError("invalid_embedding_identity")
        self.embedding_identity = self.embedder.identity.validated()
        super().__init__(
            path,
            source_id=source_id,
            identity=self.embedding_identity.__dict__,
            fault_hook=fault_hook,
        )

    def _embed(self, text):
        if type(text) is not str:
            raise ProjectionError("invalid_query")
        canonical_json(text, max_bytes=131072)
        if self.embedder.identity != self.embedding_identity:
            raise ProjectionError("embedding_identity_changed")
        try:
            self._fault("before_compute")
            raw = self.embedder.embed(text)
            if self.embedder.identity != self.embedding_identity:
                raise ProjectionError("embedding_identity_changed")
            return normalized(raw, self.embedding_identity.dimensions)
        except ProjectionError:
            raise
        except Exception:  # noqa: BLE001 - sanitize trusted callback/validation failures
            raise ProjectionError("embedding_failed") from None

    def _prepare(self, fact):
        return self._embed(fact.content)

    def _remove(self, conn, namespace, fact):
        conn.execute(
            "DELETE FROM vectors WHERE namespace=? AND fact=?", (namespace, fact)
        )

    def _insert(self, conn, namespace, fact, prepared):
        conn.execute(
            "INSERT INTO vectors VALUES (?,?,?)",
            (namespace, fact.fact_id, canonical_json(prepared)),
        )

    def search(
        self,
        authority,
        scope,
        source,
        query,
        *,
        snapshots=(),
        limit=10,
        candidate_limit=1000,
    ):
        authority.require(scope)
        integer(limit, 1, 100)
        integer(candidate_limit, 1, 1000)
        try:
            snapshots, snap, reasons, generation = self._query_start(
                authority, scope, source, snapshots, candidate_limit
            )
            # Authority payload acquisition and authorization precede any callback.
            vector = self._embed(query)
            scores = []
            with self._connect() as conn:
                live = self._live(conn, scope.namespace_id, snap)
                for fact_id, fact in live.items():
                    authority.require(scope, fact.repo_id)
                    row = conn.execute(
                        "SELECT CASE WHEN length(CAST(vector AS BLOB))<=131072 THEN vector ELSE NULL END FROM vectors WHERE namespace=? AND fact=?",
                        (scope.namespace_id, fact_id),
                    ).fetchone()
                    if row is None:
                        reasons.append("coverage_gap")
                        continue
                    try:
                        stored = normalized(
                            json.loads(row[0]), self.embedding_identity.dimensions
                        )
                    except (ValueError, TypeError, OverflowError):
                        raise ProjectionError("index_corrupt") from None
                    score = max(
                        -1.0, min(1.0, sum(a * b for a, b in zip(vector, stored)))
                    )
                    scores.append(
                        ProjectionHit(
                            fact_id,
                            score,
                            fact.content,
                            fact.metadata_json,
                            fact.revision,
                            fact.repo_id,
                            fact.snapshot_id,
                        )
                    )
                if any(
                    f.visible and f.status == "active" and f.fact_id not in live
                    for f in snap.facts
                ):
                    reasons.append("coverage_gap")
            self._fault("before_return")
            self._query_finish(
                authority, scope, source, snapshots, snap, candidate_limit, generation
            )
            reasons = tuple(dict.fromkeys(reasons))
            hits = tuple(sorted(scores, key=lambda h: (-h.score, h.fact_id))[:limit])
            return QueryResult("partial" if reasons else "complete", reasons, hits=hits)
        except ProjectionError as exc:
            authority.require(scope)
            return QueryResult("unavailable", (exc.code,))
