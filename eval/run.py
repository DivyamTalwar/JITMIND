#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JITMind evaluation CLI.

This module exists primarily to back the console entrypoint defined in setup.py.
It dispatches to the existing evaluation scripts so we don't duplicate their
argument parsers.

Usage:
  jitmind-eval <task> [args...]

Tasks:
  hotpotqa
  locomo
  narrativeqa
  ruler
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


_TASK_TO_SCRIPT = {
    "hotpotqa": "hotpotqa_test.py",
    "locomo": "locomo_test.py",
    "narrativeqa": "narrativeqa_test.py",
    "ruler": "ruler_test.py",
}


def _help() -> str:
    tasks = "\n".join(f"  - {t}" for t in sorted(_TASK_TO_SCRIPT))
    return (
        "JITMind evaluation runner\n\n"
        "Usage:\n"
        "  jitmind-eval <task> [args...]\n\n"
        "Available tasks:\n"
        f"{tasks}\n\n"
        "Example:\n"
        "  jitmind-eval hotpotqa --help\n"
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_help())
        return 0

    task = sys.argv[1].strip().lower()
    script = _TASK_TO_SCRIPT.get(task)
    if not script:
        print(f"Unknown task: {task}\n")
        print(_help())
        return 2

    script_path = Path(__file__).resolve().parent / script
    if not script_path.exists():
        print(f"Evaluation script not found: {script_path}")
        return 2

    cmd = [sys.executable, str(script_path)] + sys.argv[2:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
