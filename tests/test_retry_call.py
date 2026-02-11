from __future__ import annotations


def test_retry_call_retries_then_succeeds(monkeypatch):
    # Avoid real sleeps during tests.
    import jitmind.utils.retry as r
    monkeypatch.setattr(r.time, "sleep", lambda _s: None)

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    out = r.retry_call(fn, config=r.RetryConfig(max_attempts=5, base_delay_s=0.0, jitter_s=0.0))
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_call_respects_is_retryable(monkeypatch):
    import jitmind.utils.retry as r
    monkeypatch.setattr(r.time, "sleep", lambda _s: None)

    calls = {"n": 0}

    class Fatal(Exception):
        pass

    def fn():
        calls["n"] += 1
        raise Fatal("nope")

    try:
        r.retry_call(fn, config=r.RetryConfig(max_attempts=5), is_retryable=lambda e: not isinstance(e, Fatal))
        assert False, "expected exception"
    except Fatal:
        pass

    assert calls["n"] == 1

