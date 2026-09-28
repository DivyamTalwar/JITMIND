"""Executable installed-wheel smoke. No pytest dependency in the installed venv.

Run only with: <fresh-venv>/bin/python -I <checkout>/tests/test_standalone_install.py
  --installed-smoke <fresh-venv> <new-result.json>
A pytest run of this file tests smoke guards; it is NOT installed acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile

SMOKE_NODE = "tests/test_standalone_install.py::installed_smoke"
CHECKS = {
    "isolated_origin",
    "optional_dependencies_absent",
    "legacy_memory_research",
    "optional_adapter_import",
    "missing_adapter_capability",
}


def verify_isolation(venv: Path, origin: Path) -> None:
    assert sys.flags.isolated == 1, "isolated_interpreter_required"
    assert "PYTHONPATH" not in os.environ, "pythonpath_must_be_absent"
    assert Path(sys.prefix).resolve() == venv.resolve(), "wrong_virtual_environment"
    assert sys.prefix != sys.base_prefix, "fresh_virtual_environment_required"
    assert origin.resolve().is_relative_to(venv.resolve()), "checkout_import_refused"
    checkout = Path(__file__).resolve().parents[1]
    assert not Path.cwd().resolve().is_relative_to(checkout), "checkout_cwd_refused"
    assert not any(Path(p).resolve() == checkout for p in sys.path if p), (
        "checkout_path_refused"
    )


def legacy_roundtrip(directory: Path) -> None:
    from jitmind import MemoryAgent, ResearchAgent
    from jitmind.schemas import InMemoryMemoryStore, InMemoryPageStore

    class FakeGenerator:
        def __init__(self):
            self.calls = []

        def generate_single(self, prompt=None, schema=None, **kwargs):
            self.calls.append(schema)
            properties = (schema or {}).get("properties", {})
            if "operation" in properties:
                return {"json": {"operation": "add"}}
            if "info_needs" in properties:
                return {"json": {"tools": ["page_index"], "page_index": [0]}}
            if "content" in properties:
                assert "cobalt" in prompt, "retrieved_page_missing_from_prompt"
                return {"json": {"content": "cobalt preference", "sources": ["0"]}}
            if "enough" in properties:
                return {"json": {"enough": True}}
            if schema is None:
                return {"text": "cobalt preference"}
            raise AssertionError("unexpected_generator_contract")

    generator = FakeGenerator()
    memory = InMemoryMemoryStore(dir_path=str(directory))
    pages = InMemoryPageStore(dir_path=str(directory))
    agent = MemoryAgent(memory_store=memory, page_store=pages, generator=generator)
    result = agent.memorize("cobalt is the user's preferred color")
    assert result.new_state.abstracts == ["cobalt preference"]
    memory = InMemoryMemoryStore(dir_path=str(directory))
    pages = InMemoryPageStore(dir_path=str(directory))
    assert memory.load().abstracts == ["cobalt preference"]
    assert pages.get(0).content == "cobalt is the user's preferred color"
    research = ResearchAgent(
        page_store=pages,
        memory_store=memory,
        generator=generator,
        max_iters=1,
        enable_hyde=False,
        enable_self_rag=False,
    )
    answer = research.research("preferred color")
    assert answer.integrated_memory == "cobalt preference"
    assert answer.raw_memory["temp_memory"]["sources"] == ["0"]
    assert len(generator.calls) == 5, "public_api_did_not_use_expected_generator_calls"


def missing_adapter_capability(directory: Path) -> None:
    """Exercise actual public missing-runtime handling, never a mocked parser.

    This deliberately fails until the assembled adapter supports an absent runtime
    with a typed capability failure or an unavailable result. Arbitrary IO/import
    exceptions and AttributeError are not meaningful capability errors.
    """
    from jitmind.code_context import NodeParser
    from jitmind.code_context.process import Unavailable

    try:
        NodeParser(
            node_binary=str(directory / "absent-node"),
            adapter_dir=str(directory / "absent-adapter"),
        )
    except Unavailable as error:
        assert error.reason == "node_executable_unavailable"
    else:
        raise AssertionError("missing_runtime_requires_capability_error")


def installed_smoke(venv: Path) -> dict:
    results = {name: "not_run" for name in sorted(CHECKS)}
    report = {
        "schema": "jitmind.installed-smoke/v1",
        "node_id": SMOKE_NODE,
        "checks": results,
        "status": "failed",
        "kind": "installed_package",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": None,
        "sqlite_pragmas": None,
        "product_storage_diagnostics": None,
        "provider_calls": 0,
        "network_policy": "socket_connections_denied",
    }

    def no_network(*args, **kwargs):
        raise AssertionError("network_forbidden_in_standalone_smoke")

    # Prevent accidental live provider connections even if a regression adds one.
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    try:
        results["isolated_origin"] = "failed"
        import jitmind

        verify_isolation(venv, Path(jitmind.__file__))
        results["isolated_origin"] = "passed"
        report["origin"] = str(
            Path(jitmind.__file__).resolve().relative_to(venv.resolve())
        )
        report["dependencies"] = sorted(
            [
                [d.metadata["Name"], d.version]
                for d in importlib.metadata.distributions()
            ]
        )
        results["optional_dependencies_absent"] = "failed"
        assert all(
            importlib.util.find_spec(name) is None
            for name in ("cohere", "neo4j", "faiss")
        )
        assert shutil.which("node") is None, "node_must_be_absent_from_smoke_path"
        results["optional_dependencies_absent"] = "passed"
        with tempfile.TemporaryDirectory(prefix="jitmind-installed-") as temp:
            directory = Path(temp)
            results["legacy_memory_research"] = "failed"
            legacy_roundtrip(directory)
            results["legacy_memory_research"] = "passed"
            with sqlite3.connect(directory / "probe.sqlite") as connection:
                report["sqlite_pragmas"] = {
                    key: connection.execute("PRAGMA " + key).fetchone()[0]
                    for key in ("journal_mode", "synchronous", "foreign_keys")
                }
            if importlib.util.find_spec("jitmind.storage") is not None:
                from jitmind.storage import SQLiteDurableStore

                report["product_storage_diagnostics"] = SQLiteDurableStore(
                    directory / "authority.sqlite"
                ).diagnostics()
            # Import must work independently of the optional native dependency.
            results["optional_adapter_import"] = "failed"
            import jitmind.code_context  # noqa: F401

            results["optional_adapter_import"] = "passed"
            results["missing_adapter_capability"] = "failed"
            missing_adapter_capability(directory)
            results["missing_adapter_capability"] = "passed"
        report["status"] = "passed"
    except Exception:
        # Do not expose paths, provider payloads, or arbitrary exception strings.
        report["error"] = "installed_contract_failed"
    return report


def test_checkout_cannot_masquerade_as_installed_package(tmp_path):
    import pytest

    with pytest.raises(AssertionError):
        verify_isolation(tmp_path, Path(__file__))


def test_legacy_fake_generator_roundtrip_is_a_checkout_unit_check(tmp_path):
    legacy_roundtrip(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-smoke", action="store_true", required=True)
    parser.add_argument("venv", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = installed_smoke(args.venv)
    with args.result.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True, indent=2)
    print(
        json.dumps(
            {"status": result["status"], "checks": result["checks"]}, sort_keys=True
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
