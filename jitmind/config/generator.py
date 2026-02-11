from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union, Optional, Dict

@dataclass
class OpenAIGeneratorConfig:
    """OpenAI-compatible generator config (OpenRouter default)"""
    model_name: str = "google/gemini-3-flash-preview"
    api_key: Optional[str] = None
    base_url: Optional[str] = "https://openrouter.ai/api/v1"
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: float = 60.0
    use_schema: bool = False


@dataclass
class VLLMGeneratorConfig:
    """
    vLLM generator (local OpenAI-compatible endpoint /v1/chat/completions).
    Note: these fields are for client calls and differ from server startup params.
    """
    model_name: str = "Qwen2.5-7B-Instruct"   
    api_key: Optional[str] = "empty"          
    base_url: str = "http://localhost:8000/v1"
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: float = 60.0
    use_schema: bool = False
