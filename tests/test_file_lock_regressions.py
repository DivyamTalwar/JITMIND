"""Bounded spawn contention and real POSIX fork descriptor regressions."""

import errno
import importlib
import multiprocessing as mp
import os
import select
import threading
from pathlib import Path

import pytest

from jitmind.utils.file_lock import file_lock

CTX = mp.get_context("spawn")


def observe_contention(event):
    import fcntl

    original = fcntl.flock

    def probe(fd, flags):
        try:
            return original(fd, flags)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                event.set()
            raise

    fcntl.flock = probe


def hold_lock(path, acquired, release, contended):
    observe_contention(contended)
    with file_lock(Path(path), timeout_s=12, poll_s=0.01, stale_s=0.001):
        acquired.set()
        assert release.wait(12), "parent did not release holder"


def finish(processes):
    for process in processes:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(5)
            assert not process.is_alive()


@pytest.mark.skipif(os.name != "posix", reason="POSIX strategy; Windows unverified")
def test_aged_live_owner_kill_successor_and_third_owner(tmp_path):
    path = tmp_path / "state.lock"
    acquired = [CTX.Event() for _ in range(3)]
    release = [CTX.Event() for _ in range(3)]
    contended = [CTX.Event() for _ in range(3)]
    processes = [
        CTX.Process(
            target=hold_lock, args=(str(path), acquired[i], release[i], contended[i])
        )
        for i in range(3)
    ]
    try:
        processes[0].start()
        assert acquired[0].wait(8)
        inode = path.stat().st_ino
        os.utime(path, (1, 1))
        processes[1].start()
        assert contended[1].wait(8)
        assert not acquired[1].is_set()
        processes[0].kill()
        processes[0].join(5)
        assert processes[0].exitcode is not None
        assert acquired[1].wait(8)
        processes[2].start()
        assert contended[2].wait(8)
        assert not acquired[2].is_set()
        release[1].set()
        processes[1].join(8)
        assert processes[1].exitcode == 0
        assert acquired[2].wait(8)
        release[2].set()
        processes[2].join(8)
        assert processes[2].exitcode == 0
        assert path.stat().st_ino == inode
    finally:
        # Do not signal a multiprocessing.Condition whose waiter was killed:
        # its semaphore accounting is not recoverable. Terminate live children.
        finish(processes)


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in ("timeout_s", "poll_s")
        for value in (-1, float("nan"), float("inf"), -float("inf"), True, False)
    ]
    + [("poll_s", 0)],
)
def test_invalid_waits(tmp_path, name, value):
    with pytest.raises(ValueError):
        with file_lock(tmp_path / "x.lock", **{name: value}):
            pytest.fail("invalid wait acquired a lock")
    assert not (tmp_path / "x.lock").exists()


def test_nested_exception_releases_and_threads_contend(tmp_path):
    path = tmp_path / "x.lock"
    results = []
    done = threading.Event()

    def contender():
        try:
            with file_lock(path, timeout_s=0.1, poll_s=0.005):
                results.append("incorrect acquisition")
        except TimeoutError as exc:
            results.append(str(exc))
        finally:
            done.set()

    with pytest.raises(RuntimeError, match="body"):
        with file_lock(path):
            with file_lock(tmp_path / "." / "x.lock", timeout_s=0):
                thread = threading.Thread(target=contender)
                thread.start()
                assert done.wait(3)
                thread.join(3)
                assert not thread.is_alive()
                assert results == ["Timed out waiting for persistence lock"]
                raise RuntimeError("body")
    with file_lock(path, timeout_s=0):
        assert path.exists()
    assert path.exists()


def test_unsupported_platform_is_explicit(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("jitmind.utils.file_lock")
    path = tmp_path / "x.lock"
    monkeypatch.setattr(module.os, "name", "nt")
    with pytest.raises(NotImplementedError, match="POSIX"), file_lock(path):
        pytest.fail("unsupported platform acquired lock")


def test_wall_clock_is_not_used(tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module("jitmind.utils.file_lock")

    def fail():
        raise AssertionError("wall-clock time used")

    monkeypatch.setattr(module.time, "time", fail)
    with file_lock(tmp_path / "x.lock", timeout_s=0):
        pass


def fork_window_probe(directory, window):
    """Isolate at-fork instrumentation; the test parent supplies a hard bound."""
    import fcntl

    module = importlib.import_module("jitmind.utils.file_lock")
    path = Path(directory) / "fork.lock"
    paused, resume = threading.Event(), threading.Event()
    acquired, release = threading.Event(), threading.Event()
    preparing, forked = threading.Event(), threading.Event()
    failures, pids, descriptors = [], [], []
    ready_r, ready_w = os.pipe()
    exit_r, exit_w = os.pipe()
    real_open, real_close = os.open, os.close
    real_guard = module._descriptor_guard

    class ObservedGuard:
        def acquire(self):
            if threading.current_thread() is forker:
                # Prove the paused operation is inside the same exclusion that
                # the actual registered before-fork callback acquires.
                if real_guard.acquire(blocking=False):
                    failures.append("fork did not wait for descriptor operation")
                    preparing.set()
                    return True
                preparing.set()
            return real_guard.acquire()

        def release(self):
            real_guard.release()

        def __enter__(self):
            self.acquire()

        def __exit__(self, *args):
            self.release()

    def opened(name, *args, **kwargs):
        fd = real_open(name, *args, **kwargs)
        if name == path and threading.current_thread() is owner:
            descriptors.append(fd)
            if window == "acquisition":
                paused.set()
                assert resume.wait(5)
        return fd

    def closed(fd):
        if (
            window == "release"
            and descriptors == [fd]
            and threading.current_thread() is owner
        ):
            paused.set()
            assert resume.wait(5)
        return real_close(fd)

    def own():
        try:
            with file_lock(path):
                acquired.set()
                assert release.wait(5)
        except BaseException as exc:
            failures.append(repr(exc))

    def fork():
        try:
            pid = os.fork()
            if pid == 0:
                # Remain idle while the parent verifies ownership/release. Use
                # only bounded pipe waits, and never unwind inherited contexts.
                status = 1
                try:
                    real_close(ready_r)
                    real_close(exit_w)
                    if window == "acquisition":
                        try:
                            os.fstat(descriptors[0])
                        except OSError as exc:
                            assert exc.errno == errno.EBADF
                        else:
                            raise AssertionError("inherited descriptor still open")
                    assert not module._descriptors
                    with file_lock(Path(directory) / "child.lock", timeout_s=0):
                        pass
                    os.write(ready_w, b"R")
                    assert select.select([exit_r], [], [], 5)[0]
                    assert os.read(exit_r, 1) == b"X"
                    status = 0
                finally:
                    os._exit(status)
            pids.append(pid)
        except BaseException as exc:
            failures.append(repr(exc))
        finally:
            forked.set()

    owner = threading.Thread(target=own)
    forker = threading.Thread(target=fork)
    module._descriptor_guard = ObservedGuard()
    os.open, os.close = opened, closed
    try:
        owner.start()
        if window == "release":
            assert acquired.wait(5)
            release.set()
        assert paused.wait(5)
        forker.start()
        assert preparing.wait(5)
        # Preparation is blocked on the lifecycle guard. Release the paused
        # descriptor operation BEFORE waiting for preparation/fork to finish.
        resume.set()
        assert forked.wait(5)
        forker.join(5)
        assert not forker.is_alive() and not failures
        assert select.select([ready_r], [], [], 5)[0]
        assert os.read(ready_r, 1) == b"R"
        if window == "acquisition":
            assert acquired.wait(5)
            contender = real_open(path, os.O_RDWR)
            try:
                # Child cleanup must close, never LOCK_UN the parent's lock.
                with pytest.raises(OSError) as raised:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert raised.value.errno in (errno.EACCES, errno.EAGAIN)
            finally:
                real_close(contender)
        release.set()
        owner.join(5)
        assert not owner.is_alive() and not failures
        # The child is still alive; its exit cannot explain successful locking.
        assert os.waitpid(pids[0], os.WNOHANG) == (0, 0)
        with file_lock(path, timeout_s=0):
            pass
    finally:
        resume.set()
        release.set()
        owner.join(5)
        if forker.ident is not None:
            forker.join(5)
        os.open, os.close = real_open, real_close
        module._descriptor_guard = real_guard
        try:
            os.write(exit_w, b"X")
        except BrokenPipeError:
            pass
        for fd in (ready_r, ready_w, exit_r, exit_w):
            real_close(fd)
        for pid in pids:
            waited, status = os.waitpid(pid, 0)  # child's pipe wait is bounded
            assert waited == pid and os.waitstatus_to_exitcode(status) == 0
    assert not failures


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
@pytest.mark.parametrize("window", ["acquisition", "release"])
def test_fork_waits_for_descriptor_lifecycle(tmp_path, window):
    process = CTX.Process(target=fork_window_probe, args=(str(tmp_path), window))
    try:
        process.start()
        process.join(25)
        assert not process.is_alive(), "fork lifecycle probe exceeded its bound"
        assert process.exitcode == 0
    finally:
        finish([process])


@pytest.mark.parametrize("contention", ["thread", "kernel"])
def test_huge_finite_waits_use_bounded_sleep(tmp_path, monkeypatch, contention):
    import fcntl

    module = importlib.import_module("jitmind.utils.file_lock")
    path = tmp_path / "huge.lock"
    sleeps, errors = [], []

    class StopWaiting(Exception):
        pass

    def sleep(delay):
        sleeps.append(delay)
        assert 0 < delay <= 1
        # Kernel contention must not retain the global descriptor guard.
        assert module._descriptor_guard.acquire(blocking=False)
        module._descriptor_guard.release()
        with file_lock(tmp_path / "unrelated.lock", timeout_s=0):
            pass
        raise StopWaiting

    def contend():
        try:
            with file_lock(path, timeout_s=1e300, poll_s=1e300):
                errors.append("unexpected acquisition")
        except StopWaiting:
            pass
        except BaseException as exc:
            errors.append(repr(exc))

    monkeypatch.setattr(module.time, "sleep", sleep)
    if contention == "thread":
        with file_lock(path):
            thread = threading.Thread(target=contend)
            thread.start()
            thread.join(5)
            assert not thread.is_alive()
    else:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            contend()
        finally:
            os.close(fd)
    assert sleeps == [1.0] and not errors
