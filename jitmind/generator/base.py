from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class AbsGenerator(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.config = config

    @abstractmethod
    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single response.
        Return format: {"text": str, "json": dict|None, "response": dict}
        Note: temperature and max_tokens are already set in config; no need to pass again.
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of responses.
        Return format: [{"text": str, "json": dict|None, "response": dict}, ...]
        Note: temperature and max_tokens are already set in config; no need to pass again.
        """
        pass
