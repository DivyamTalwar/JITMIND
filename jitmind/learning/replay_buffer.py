# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Any, List, Optional
import random


class ExperienceReplayBuffer:
    def __init__(self, max_size: int = 10000) -> None:
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_size)

    def add_experience(
        self,
        query: str,
        retrieved: List[str],
        response: Optional[str] = None,
        feedback: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.buffer.append({
            "query": query,
            "retrieved": retrieved,
            "response": response,
            "feedback": feedback,
            "meta": meta or {},
        })

    def sample_batch(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        if not self.buffer:
            return []
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
