"""Installed benchmark dispatcher for JITMIND evaluation tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TASKS = {
    "hotpotqa": "hotpotqa_test.py",
    "locomo": "locomo_test.py",
    "narrativeqa": "narrativeqa_test.py",
    "ruler": "ruler_test.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a JITMIND evaluation task")
    parser.add_argument("task", choices=sorted(TASKS))
    args, forwarded = parser.parse_known_args(argv)
    script = Path(__file__).resolve().parents[2] / "eval" / TASKS[args.task]
    if not script.is_file():
        parser.error(
            "benchmark scripts are repository resources; run from a source checkout"
        )
    return subprocess.call([sys.executable, str(script), *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
