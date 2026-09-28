"""Host-issued, revocable authorization contexts (no remote identity service)."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import ref


class ScopeDenied(PermissionError):
    """Deliberately does not disclose principals, paths, grants, or content."""

    def __init__(self) -> None:
        super().__init__("Scope denied")


def _identifier(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 1024
        or value != value.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ScopeDenied()
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ScopeDenied() from None
    return value


@dataclass(frozen=True)
class ScopeContext:
    principal_id: str
    namespace_id: str
    authorized_repo_ids: tuple[str, ...]
    authorization_version: int
    request_id: str

    def __post_init__(self) -> None:
        for value in (self.principal_id, self.namespace_id, self.request_id):
            _identifier(value)
        if (
            type(self.authorized_repo_ids) is not tuple
            or len(self.authorized_repo_ids) > 1024
            or type(self.authorization_version) is not int
            or not 0 < self.authorization_version < 2**63
        ):
            raise ScopeDenied()
        for repo_id in self.authorized_repo_ids:
            _identifier(repo_id)
        if len(set(self.authorized_repo_ids)) != len(self.authorized_repo_ids):
            raise ScopeDenied()


class ScopeAuthority:
    """Owned by a trusted host, which authenticates callers before issuing contexts.

    Administrative methods must never be exposed as user-request RPCs. Contexts
    are process-local capabilities, not serialized bearer credentials.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._versions: dict[tuple[str, str], int] = {}
        self._grants: dict[tuple[str, str], tuple[tuple[str, ...], int]] = {}
        self._issued: dict[int, tuple] = {}

    def grant(self, principal_id: str, namespace_id: str, repo_ids) -> None:
        key = (_identifier(principal_id), _identifier(namespace_id))
        # Do not consume arbitrary or infinite iterables.
        if type(repo_ids) not in (list, tuple) or len(repo_ids) > 1024:
            raise ScopeDenied()
        repos = tuple(_identifier(repo) for repo in repo_ids)
        if len(set(repos)) != len(repos):
            raise ScopeDenied()
        with self._lock:
            version = self._versions.get(key, 0) + 1
            if version >= 2**63:
                raise ScopeDenied()
            self._versions[key] = version
            self._grants[key] = (repos, version)

    def context(
        self, principal_id: str, namespace_id: str, request_id: str
    ) -> ScopeContext:
        key = (_identifier(principal_id), _identifier(namespace_id))
        _identifier(request_id)
        with self._lock:
            grant = self._grants.get(key)
            if grant is None:
                raise ScopeDenied()
            scope = ScopeContext(*key, grant[0], grant[1], request_id)
            identity = id(scope)
            owner = ref(self)

            def discard(_):
                authority = owner()
                if authority is not None:
                    with authority._lock:
                        authority._issued.pop(identity, None)

            self._issued[identity] = (ref(scope, discard), self._claims(scope))
            return scope

    @staticmethod
    def _claims(scope):
        return (
            scope.principal_id,
            scope.namespace_id,
            scope.authorized_repo_ids,
            scope.authorization_version,
            scope.request_id,
        )

    def require(self, scope: ScopeContext, repo_id: str | None = None) -> None:
        with self._lock:
            issued = self._issued.get(id(scope))
            if (
                type(scope) is not ScopeContext
                or issued is None
                or issued[0]() is not scope
                or issued[1] != self._claims(scope)
            ):
                raise ScopeDenied()
            grant = self._grants.get((scope.principal_id, scope.namespace_id))
            if grant != (scope.authorized_repo_ids, scope.authorization_version):
                raise ScopeDenied()
            if repo_id is not None and _identifier(repo_id) not in grant[0]:
                raise ScopeDenied()

    def revoke(self, principal_id: str, namespace_id: str) -> None:
        key = (_identifier(principal_id), _identifier(namespace_id))
        with self._lock:
            self._grants.pop(key, None)
            self._versions[key] = self._versions.get(key, 0) + 1


__all__ = ["ScopeAuthority", "ScopeContext", "ScopeDenied"]
