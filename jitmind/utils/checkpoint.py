"""Identity-safe state checkpoints, including read compatibility with v1 JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from jitmind.utils.atomic_io import (
    CorruptStoreError,
    PersistenceError,
    StorageError,
    _sync_directory,
    atomic_write_json,
)
from jitmind.utils.file_lock import file_lock


class CheckpointFileError(StorageError):
    """Sanitized checkpoint file failure (never equivalent to absence)."""


class CheckpointCorrupt(CheckpointFileError, CorruptStoreError):
    def __init__(self, message: str):
        # Keep specific diagnostics and the legacy generic corruption catch.
        StorageError.__init__(self, message)


class CheckpointSchemaError(CheckpointCorrupt):
    pass


class CheckpointIdentityError(CheckpointFileError):
    pass


class CheckpointIOError(CheckpointFileError):
    pass


def _identifier(value: str) -> None:
    if (
        type(value) is not str
        or not 0 < len(value) <= 1024
        or not value.strip()
        or any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
    ):
        raise CheckpointIdentityError("Invalid checkpoint identity")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


def _validate_keys(value, depth=0):
    # json.dumps coerces keys before the duplicate-checking reader sees them.
    if depth > 64:
        raise ValueError()
    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError()
            _validate_keys(child, depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_keys(child, depth + 1)


class CheckpointManager:
    """The directory is trusted host configuration, never a request parameter.

    v2 filenames hash the exact [namespace, thread_id]. JSON state is always
    legacy_unvalidated: only the optional event ledger can validate source ranges.
    """

    def __init__(self, dir_path: str = "./checkpoints") -> None:
        self._dir = Path(dir_path)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str, *, namespace: str = "default") -> Path:
        _identifier(thread_id)
        _identifier(namespace)
        identity = json.dumps(
            [namespace, thread_id], ensure_ascii=True, separators=(",", ":")
        )
        return self._dir / f"v2-{sha256(identity.encode('utf-8')).hexdigest()}.json"

    def _legacy_path(self, thread_id: str) -> Path | None:
        safe_id = "".join(c for c in thread_id if c.isalnum() or c in ("-", "_"))
        if len(safe_id.encode("utf-8")) > 250:
            return None  # Such an alias could never have been saved on supported filesystems.
        return self._dir / f"{safe_id}.json"

    def _read(
        self, path: Path, thread_id: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        try:
            if path.is_symlink():
                raise CheckpointIOError("Checkpoint links are not supported")
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(
                    handle,
                    object_pairs_hook=_unique_object,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
                )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise CheckpointCorrupt("Invalid checkpoint JSON") from exc
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise CheckpointIOError("Checkpoint read failed") from exc
        if (
            type(data) is not dict
            or type(data.get("state")) is not dict
            or type(data.get("thread_id")) is not str
            or type(data.get("timestamp")) is not str
        ):
            raise CheckpointSchemaError("Invalid checkpoint schema")
        version = data.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise CheckpointSchemaError("Unsupported checkpoint schema")
        if version == 2 and (
            "namespace" not in data or data.get("validation") != "legacy_unvalidated"
        ):
            raise CheckpointSchemaError("Invalid checkpoint schema")
        stored_namespace = data.get("namespace", "default")
        try:
            _identifier(data["thread_id"])
            _identifier(stored_namespace)
        except CheckpointIdentityError as exc:
            # Malformed stored identities are corruption, not a valid alias
            # collision that a save may leave behind while creating v2 state.
            raise CheckpointSchemaError("Invalid checkpoint identity schema") from exc
        if (thread_id is not None and data["thread_id"] != thread_id) or (
            namespace is not None and stored_namespace != namespace
        ):
            raise CheckpointIdentityError("Checkpoint identity mismatch")
        return {
            **data,
            "namespace": stored_namespace,
            "validation": "legacy_unvalidated",
        }

    def save_checkpoint(
        self, thread_id: str, state: dict[str, Any], *, namespace: str = "default"
    ) -> None:
        path = self._path(thread_id, namespace=namespace)
        if type(state) is not dict:
            raise CheckpointSchemaError("Checkpoint state must be an object")
        payload = {
            "schema_version": 2,
            "namespace": namespace,
            "thread_id": thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "validation": "legacy_unvalidated",
        }
        # Validate before taking the writer lock; reject non-JSON/NaN state.
        try:
            _validate_keys(payload)
            # Publish a detached representation verified by the reader's rules.
            payload = json.loads(
                json.dumps(payload, allow_nan=False),
                object_pairs_hook=_unique_object,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            )
        except (TypeError, ValueError, RecursionError):
            raise CheckpointSchemaError("Checkpoint state must be JSON data") from None
        try:
            with file_lock(self._dir / ".checkpoints.lock"):
                found = False
                if path.exists() or path.is_symlink():
                    try:
                        self._read(path, thread_id, namespace)
                        found = True
                    except FileNotFoundError:
                        pass
                if not found and namespace == "default":
                    legacy = self._legacy_path(thread_id)
                    if legacy is not None and (legacy.exists() or legacy.is_symlink()):
                        try:
                            self._read(legacy, thread_id, namespace)
                        except (CheckpointIdentityError, FileNotFoundError):
                            pass  # Safe new filename; the other identity's alias stays intact.
                atomic_write_json(path, payload, ensure_ascii=False, indent=2)
        except (PersistenceError, CheckpointFileError):
            raise
        except (OSError, TimeoutError, StorageError) as exc:
            raise CheckpointIOError("Checkpoint save failed") from exc

    def load_checkpoint_record(
        self, thread_id: str, *, namespace: str = "default"
    ) -> dict[str, Any] | None:
        path = self._path(thread_id, namespace=namespace)
        try:
            with file_lock(self._dir / ".checkpoints.lock"):
                candidates = [path]
                if namespace == "default":
                    legacy = self._legacy_path(thread_id)
                    if legacy is not None and legacy != path:
                        candidates.append(legacy)
                for candidate in candidates:
                    try:
                        return self._read(candidate, thread_id, namespace)
                    except FileNotFoundError:
                        continue
                return None
        except (PersistenceError, CheckpointFileError):
            raise
        except (OSError, TimeoutError, StorageError) as exc:
            raise CheckpointIOError("Checkpoint load failed") from exc

    def load_checkpoint(
        self, thread_id: str, *, namespace: str = "default"
    ) -> dict[str, Any] | None:
        record = self.load_checkpoint_record(thread_id, namespace=namespace)
        return None if record is None else record["state"]

    def list_checkpoints(self, *, namespace: str = "default") -> list[str]:
        """v2 listing returns exact IDs, not lossy v1 filename aliases. See migration docs."""
        _identifier(namespace)
        try:
            with file_lock(self._dir / ".checkpoints.lock"):
                identities = set()
                for path in self._dir.glob("*.json"):
                    try:
                        data = self._read(path)
                    except FileNotFoundError:
                        continue
                    if data["namespace"] == namespace:
                        identities.add(data["thread_id"])
                return sorted(identities)
        except (PersistenceError, CheckpointFileError):
            raise
        except (OSError, TimeoutError, StorageError) as exc:
            raise CheckpointIOError("Checkpoint listing failed") from exc

    def delete_checkpoint(self, thread_id: str, *, namespace: str = "default") -> bool:
        path = self._path(thread_id, namespace=namespace)
        try:
            with file_lock(self._dir / ".checkpoints.lock"):
                candidates = [path]
                if namespace == "default":
                    legacy = self._legacy_path(thread_id)
                    if legacy is not None and legacy != path:
                        candidates.append(legacy)
                matching = []
                for candidate in candidates:
                    if candidate.exists() or candidate.is_symlink():
                        try:
                            self._read(candidate, thread_id, namespace)
                        except FileNotFoundError:
                            continue
                        except CheckpointIdentityError:
                            # A v1 alias may belong to someone else. Never delete it.
                            if candidate == path or not matching:
                                raise
                            continue
                        matching.append(candidate)
                # Validate all files before deleting any of them.
                removed = False
                try:
                    for candidate in matching:
                        try:
                            candidate.unlink()
                        except FileNotFoundError:
                            continue
                        removed = True
                    if removed:
                        _sync_directory(self._dir)
                except Exception as exc:
                    # Includes partial multi-copy deletion and late sync failures.
                    raise PersistenceError(outcome_uncertain=removed) from exc
                return removed
        except (PersistenceError, CheckpointFileError):
            raise
        except (OSError, TimeoutError, StorageError) as exc:
            raise CheckpointIOError("Checkpoint delete failed") from exc
