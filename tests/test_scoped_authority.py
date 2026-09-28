from dataclasses import FrozenInstanceError, replace

import pytest

from jitmind.scope import ScopeAuthority, ScopeContext, ScopeDenied


def test_issued_context_required_and_revocation_regrant_monotonic():
    authority = ScopeAuthority()
    authority.grant("alice", "team/a", ["repo-a"])
    scope = authority.context("alice", "team/a", "request")
    authority.require(scope, "repo-a")
    with pytest.raises(FrozenInstanceError):
        scope.namespace_id = "team/b"
    for forged in (
        replace(scope),
        replace(scope, principal_id="bob"),
        object(),
        "alice",
    ):
        with pytest.raises(ScopeDenied, match="^Scope denied$"):
            authority.require(forged)
    authority.revoke("alice", "team/a")
    with pytest.raises(ScopeDenied):
        authority.require(scope)
    with pytest.raises(ScopeDenied):
        authority.context("alice", "team/a", "new")
    authority.grant("alice", "team/a", ["repo-a"])
    new = authority.context("alice", "team/a", "new")
    assert new.authorization_version > scope.authorization_version
    with pytest.raises(ScopeDenied):
        authority.require(scope)
    authority.require(new)
    with pytest.raises(ScopeDenied):
        authority.require(new, "repo-a-sibling")


@pytest.mark.parametrize(
    "bad", ["", " ", "a\0b", "a\nb", " a", "a ", None, True, 3, "a" * 1025]
)
def test_invalid_identifiers(bad):
    with pytest.raises(ScopeDenied):
        ScopeAuthority().grant(bad, "namespace", [])


@pytest.mark.parametrize(
    "repos", ["repo", iter(["repo"]), ["x"] * 1025, ["x", "x"], [False]]
)
def test_bounded_repository_identifiers(repos):
    with pytest.raises(ScopeDenied):
        ScopeAuthority().grant("p", "n", repos)


@pytest.mark.parametrize("version", [True, 0, -1, 1.1, "1", 2**63])
def test_invalid_revision(version):
    with pytest.raises(ScopeDenied):
        ScopeContext("p", "n", (), version, "r")


def test_namespace_is_an_exact_opaque_identifier():
    authority = ScopeAuthority()
    authority.grant("p", "../opaque/a", [])
    authority.require(authority.context("p", "../opaque/a", "r"))
    for other in ("a", "opaque/a", "../opaque/a-sibling"):
        with pytest.raises(ScopeDenied):
            authority.context("p", other, "r")


def test_authority_remembers_issued_claims_even_if_frozen_dataclass_is_bypassed():
    authority = ScopeAuthority()
    authority.grant("alice", "a", [])
    authority.grant("bob", "b", [])
    scope = authority.context("alice", "a", "r")
    object.__setattr__(scope, "principal_id", "bob")
    object.__setattr__(scope, "namespace_id", "b")
    with pytest.raises(ScopeDenied):
        authority.require(scope)


def test_malformed_unicode_identifier_is_generic():
    with pytest.raises(ScopeDenied, match="^Scope denied$"):
        ScopeAuthority().grant("\ud800", "n", [])
