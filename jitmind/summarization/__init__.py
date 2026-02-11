# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

try:
    from .raptor import HierarchicalSummarizer
except ImportError:
    HierarchicalSummarizer = None  # type: ignore
    warnings.warn("HierarchicalSummarizer not available (optional dependencies may be missing)")

__all__ = ["HierarchicalSummarizer"]
