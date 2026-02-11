# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

try:
    from .consolidation import MemoryConsolidator
except ImportError:
    MemoryConsolidator = None  # type: ignore
    warnings.warn("MemoryConsolidator not available (optional dependencies may be missing)")

__all__ = ["MemoryConsolidator"]
