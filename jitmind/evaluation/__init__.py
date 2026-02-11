# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

try:
    from .ragas_eval import RAGASEvaluator
except Exception:
    RAGASEvaluator = None  # type: ignore
    warnings.warn("RAGASEvaluator not available (optional dependencies may be missing or incompatible)")

__all__ = ["RAGASEvaluator"]
