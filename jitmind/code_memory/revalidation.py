"""Deterministic native revalidation, inspired by Koragraph's conservative guards.

No donor source is copied. See docs/code-anchors.md for the design reference.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass

from .binding_models import CodeBinding, EvidenceRef


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


@dataclass(frozen=True)
class Candidate:
    evidence: EvidenceRef
    kind: str
    signature_digest: str
    raw_source_digest: str
    body_digest: str


def candidate(evidence, qualified_name: str) -> Candidate:
    """Only the declaration's name is erased; literals, parameters and scope stay."""
    tree = ast.parse(textwrap.dedent(evidence.source))
    if len(tree.body) != 1 or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        raise ValueError("Unsupported declaration")
    node = tree.body[0]
    if node.name != evidence.name:
        raise ValueError("Declaration mismatch")
    node.name = "_binding_name_"
    body = digest(ast.dump(node, include_attributes=False))
    node.body = []
    signature = digest(ast.dump(node, include_attributes=False))
    return Candidate(
        EvidenceRef(
            source_id=evidence.source_id,
            path=evidence.path,
            qualified_name=qualified_name,
            start_line=evidence.start_line,
            end_line=evidence.end_line,
        ),
        evidence.kind,
        signature,
        digest(evidence.source),
        body,
    )


def decide(binding: CodeBinding, candidates: tuple[Candidate, ...], complete: bool):
    """Return state, reason, selected candidate, bounded review candidates."""
    if not complete:
        return "unknown", "source_unknown", None, ()
    same = [
        c
        for c in candidates
        if c.evidence.path == binding.path
        and c.evidence.qualified_name == binding.qualified_name
    ]
    if len(same) > 1:
        return "unknown", "ambiguous_candidates", None, tuple(same)
    if same:
        hit = same[0]
        if hit.kind != binding.kind or hit.signature_digest != binding.signature_digest:
            return "changed", "identity_changed", None, (hit,)
        if hit.body_digest != binding.body_digest:
            return "changed", "body_changed", None, (hit,)
        # A previously observed change is not semantic review of the fact.
        if binding.review_required:
            return "changed", "body_changed", None, (hit,)
        return "verified_current", "same_identity", hit, (hit,)
    compatible = tuple(
        c
        for c in candidates
        if c.kind == binding.kind and c.signature_digest == binding.signature_digest
    )
    matches = tuple(c for c in compatible if c.body_digest == binding.body_digest)
    if len(matches) > 1:
        return "unknown", "ambiguous_candidates", None, matches
    if len(matches) == 1:
        hit = matches[0]
        if binding.review_required:
            return "changed", "body_changed", None, matches
        if hit.evidence.path == binding.path:
            reason = "same_file_rename"
        elif hit.evidence.qualified_name == binding.qualified_name:
            reason = "file_move"
        else:
            reason = "rename_and_move"
        return "verified_current", reason, hit, matches
    plausible = tuple(
        c
        for c in candidates
        if c.kind == binding.kind
        and c.evidence.qualified_name.rsplit(".", 1)[-1]
        == binding.qualified_name.rsplit(".", 1)[-1]
    )
    if compatible or plausible:
        return "unknown", "plausible_candidates", None, compatible or plausible
    return "orphaned_confirmed", "symbol_absent", None, ()
