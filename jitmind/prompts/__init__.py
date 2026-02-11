# -*- coding: utf-8 -*-
"""
Prompts Module

This module contains all prompt templates and prompt-related tools for the JITMind (JITMind) framework.
Prompts are used to guide the behavior and responses of different agents (e.g., MemoryAgent, ResearchAgent) for various tasks, such as memory management and research reasoning.

Available Prompts:
- memory_prompts: Templates for memory management and updating.
- research_prompts: Templates for research, reasoning, and scientific inquiry.
"""
from .memory_prompts import MemoryAgent_PROMPT, MemoryOperation_PROMPT, ConflictCheck_PROMPT
from .research_prompts import (
    Planning_PROMPT,
    Integrate_PROMPT,
    InfoCheck_PROMPT,
    GenerateRequests_PROMPT,
    HYDE_PROMPT,
    SELF_RAG_PROMPT,
)
from .reflection_prompts import REFLECTION_PROMPT
from .summarization_prompts import HIERARCHICAL_SUMMARY_PROMPT, CONSOLIDATION_PROMPT
from .profile_prompts import PROFILE_UPDATE_PROMPT

__all__ = [
    "MemoryAgent_PROMPT",
    "MemoryOperation_PROMPT",
    "ConflictCheck_PROMPT",
    "Planning_PROMPT",
    "Integrate_PROMPT",
    "InfoCheck_PROMPT",
    "GenerateRequests_PROMPT",
    "HYDE_PROMPT",
    "SELF_RAG_PROMPT",
    "REFLECTION_PROMPT",
    "HIERARCHICAL_SUMMARY_PROMPT",
    "CONSOLIDATION_PROMPT",
    "PROFILE_UPDATE_PROMPT",
]
