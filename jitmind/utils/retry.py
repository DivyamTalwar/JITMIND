# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TypeVar
import random
import time


T = TypeVar("T")


@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    jitter_s: float = 0.2


def retry_call(
    fn: Callable[[], T],
    *,
    config: Optional[RetryConfig] = None,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
) -> T:
    """
    Call `fn()` with exponential backoff retries.

    Default behavior: retry all exceptions up to max_attempts.
    """
    cfg = config or RetryConfig()
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt < max(cfg.max_attempts, 1):
        attempt += 1
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if is_retryable is not None and not is_retryable(e):
                raise
            if attempt >= cfg.max_attempts:
                raise
            delay = min(cfg.max_delay_s, cfg.base_delay_s * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, cfg.jitter_s)
            time.sleep(delay)

    # Unreachable, but keep mypy happy.
    assert last_exc is not None
    raise last_exc

