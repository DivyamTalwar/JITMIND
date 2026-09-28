"""Reporting-contract controls, not installed-product acceptance evidence."""

import pytest

from jitmind.code_context.process import Unavailable
from test_standalone_install import missing_adapter_capability


def test_smoke_accepts_real_typed_missing_runtime(tmp_path):
    missing_adapter_capability(tmp_path)


@pytest.mark.parametrize(
    "error,expected",
    [
        (Unavailable("unrelated_failure"), AssertionError),
        (Unavailable("runtime_unavailable"), AssertionError),
        (FileNotFoundError("node_executable_unavailable"), FileNotFoundError),
        (AttributeError("node_executable_unavailable"), AttributeError),
    ],
)
def test_smoke_does_not_accept_wrong_reason_or_untyped_error(
    tmp_path, monkeypatch, error, expected
):
    import jitmind.code_context as module

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(module, "NodeParser", fail)
    with pytest.raises(expected):
        missing_adapter_capability(tmp_path)
