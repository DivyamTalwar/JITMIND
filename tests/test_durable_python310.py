"""Exercise SQLite error handling when Python 3.11 metadata is unavailable."""

import sqlite3

import pytest

from jitmind.storage import StorageBusy, StorageFailure
from jitmind.storage.sqlite import _db_error


@pytest.mark.parametrize(
    "message,expected",
    [
        ("database is locked", StorageBusy),
        ("database table is locked", StorageBusy),
        ("database table is locked: items", StorageBusy),
        ("database schema is locked: main", StorageBusy),
        ("database or disk is full", StorageFailure),
        ("file is not a database", StorageFailure),
        ("SQL failed near 'database is locked'", StorageFailure),
    ],
)
def test_python310_errors_without_module_aliases(monkeypatch, message, expected):
    monkeypatch.delattr(sqlite3, "SQLITE_BUSY", raising=False)
    monkeypatch.delattr(sqlite3, "SQLITE_LOCKED", raising=False)
    error = sqlite3.OperationalError(message)
    assert not hasattr(error, "sqlite_errorcode")
    result = _db_error(error)
    assert isinstance(result, expected)
    assert str(result) in {"storage_busy", "storage_failure"}


@pytest.mark.parametrize(
    "code,expected",
    [
        (5, StorageBusy),
        (6, StorageBusy),
        (5 | (2 << 8), StorageBusy),
        (13, StorageFailure),
        (0, StorageFailure),
        (True, StorageFailure),
        (-251, StorageFailure),
        ("5", StorageFailure),
    ],
)
def test_numeric_codes_take_precedence_over_message(monkeypatch, code, expected):
    monkeypatch.delattr(sqlite3, "SQLITE_BUSY", raising=False)
    monkeypatch.delattr(sqlite3, "SQLITE_LOCKED", raising=False)
    error = sqlite3.OperationalError("database is locked")
    error.sqlite_errorcode = code
    assert isinstance(_db_error(error), expected)


def test_non_operational_error_is_not_retried_as_contention(monkeypatch):
    monkeypatch.delattr(sqlite3, "SQLITE_BUSY", raising=False)
    monkeypatch.delattr(sqlite3, "SQLITE_LOCKED", raising=False)
    assert isinstance(
        _db_error(sqlite3.IntegrityError("database is locked")), StorageFailure
    )
