"""Disposable process interruption and import boundaries for real consumers."""

import os
import subprocess
import sys

import pytest

from jitmind.projections import SQLiteProjectionSource, SQLiteVectorProjection
from jitmind.scope import ScopeAuthority
from jitmind.storage import SQLiteDurableStore


@pytest.mark.parametrize("stage", ["before_commit", "after_commit"])
def test_actual_process_exit_and_reconcile(tmp_path, stage):
    code = r"""
import os
import sys
from pathlib import Path
from jitmind.projections import SQLiteProjectionSource, SQLiteVectorProjection
from jitmind.scope import ScopeAuthority
from jitmind.storage import SQLiteDurableStore, IngestRequest, Proposal
root = Path(sys.argv[1])
store = SQLiteDurableStore(root / "authority")
store.ingest(IngestRequest.create("ns", "one", "tea"), lambda _: Proposal(
    abstract="tea", header="tea", decorated="tea", decision={"operation":"add"}))
authority = ScopeAuthority()
authority.grant("host", "ns", [])
scope = authority.context("host", "ns", "child")
source = SQLiteProjectionSource(store, source_id="fixture")
def fault(current):
    if current == sys.argv[2]:
        os._exit(23)
vector = SQLiteVectorProjection(root / "index", source_id="fixture", fault_hook=fault)
vector.deliver(authority, scope, source, source.events(authority, scope)[0])
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), stage],
        check=False,
        capture_output=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.returncode == 23, result.stderr.decode()
    a = ScopeAuthority()
    a.grant("host", "ns", [])
    s = a.context("host", "ns", "parent")
    source = SQLiteProjectionSource(
        SQLiteDurableStore(tmp_path / "authority"), source_id="fixture"
    )
    vector = SQLiteVectorProjection(tmp_path / "index", source_id="fixture")
    if stage == "before_commit":
        assert vector.status(a, s, source)["watermark"] == 0
    vector.deliver(a, s, source, source.events(a, s)[0])
    assert vector.status(a, s, source)["watermark"] == 1
    assert vector.search(a, s, source, "tea").hits[0].content == "tea"


def test_import_performs_no_network_or_model_construction():
    code = r"""
import socket
calls = []
def forbidden(*args, **kwargs):
    calls.append("network")
    raise RuntimeError("no network in fixture")
socket.socket.connect = forbidden
socket.create_connection = forbidden
import jitmind.projections as projections
assert not calls
assert projections.FeatureHashingEmbedder().identity.model == "lexical-feature-hashing"
assert "openai" not in projections.__dict__
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr.decode()
