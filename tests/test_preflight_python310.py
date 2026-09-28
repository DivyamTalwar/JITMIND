"""Keep deferred contention distinct from corrupt/unavailable storage on 3.10."""

import sqlite3

import pytest

from jitmind.code_memory.preflight import PreflightPolicy
from jitmind.code_memory.work_storage import Budget, Deferred, WorkDatabase, WorkError


@pytest.mark.parametrize(
    "message,deferred",
    [
        ("database is locked", True),
        ("database table is locked", True),
        ("database table is locked: lessons", True),
        ("database schema is locked: main", True),
        ("interrupted", True),
        ("database or disk is full", False),
        ("file is not a database", False),
        ("near 'database is locked': syntax error", False),
    ],
)
def test_metadata_free_errors_preserve_public_state(tmp_path, message, deferred):
    database = WorkDatabase(tmp_path / "work.db")
    error = sqlite3.OperationalError(message)
    assert not hasattr(error, "sqlite_errorcode")
    with pytest.raises(Deferred if deferred else WorkError) as caught:
        with database.connect(Budget(5)):
            raise error
    assert isinstance(caught.value, Deferred) is deferred
    assert str(caught.value) in {"deadline_or_contention", "storage_unavailable"}


def test_production_deadline_is_not_relaxed():
    assert PreflightPolicy().deadline_seconds == 0.15
