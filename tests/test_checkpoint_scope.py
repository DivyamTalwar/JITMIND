from dataclasses import FrozenInstanceError, replace

import pytest

from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied


def test_authoritative_scope_revocation_and_forgery():
    authority = ScopeAuthority()
    authority.grant("alice", "a/b", ["repo"])
    scope = authority.context("alice", "a/b", "request")
    authority.require(scope, "repo")
    for forged in (
        replace(scope, namespace_id="ab"),
        replace(scope, principal_id="mallory"),
        replace(scope, authorized_repo_ids=("other",)),
        replace(scope, authorization_version=2),
    ):
        with pytest.raises(ScopeDenied, match="^Scope denied$"):
            authority.require(forged)
    with pytest.raises(ScopeDenied):
        authority.require(scope, "other")
    with pytest.raises(FrozenInstanceError):
        scope.namespace_id = "ab"
    authority.revoke("alice", "a/b")
    with pytest.raises(ScopeDenied):
        authority.require(scope)
    with pytest.raises(ScopeDenied):
        authority.context("alice", "a/b", "request")
    authority.grant("alice", "a/b", ["repo"])
    with pytest.raises(ScopeDenied):
        authority.require(scope)
    assert authority.context("alice", "a/b", "new").authorization_version == 3


@pytest.mark.parametrize("value", ["", " ", "a\x00b", "a\nb", "x" * 1025, 1, None])
def test_scope_invalid_identifiers(value):
    authority = ScopeAuthority()
    with pytest.raises(ScopeDenied):
        authority.grant("alice", value, [])


@pytest.mark.parametrize(
    "repos", [["a"] * 2, ["x"] * 1025, "repo", [None], (x for x in ["a"])]
)
def test_scope_bounded_repository_grants(repos):
    with pytest.raises(ScopeDenied):
        ScopeAuthority().grant("alice", "space", repos)


@pytest.mark.parametrize("version", [True, 0, -1, 2**63, 1.0])
def test_scope_rejects_invalid_revision(version):
    with pytest.raises(ScopeDenied):
        ScopeContext("alice", "space", (), version, "r")
