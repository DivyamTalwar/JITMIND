"""POSIX streaming subprocess boundary; never passes a shell or host environment."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event

from .models import MAX_RESPONSE_BYTES


class Unavailable(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class CapabilityUnavailable(Unavailable):
    """The trusted optional runtime is absent or unusable at configuration time.

    Fixed reason codes are safe to display; no host path or OS error is exposed.
    Construction checks paths only and never starts Node or a provider.
    """


def checkpoint(deadline: float, cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Unavailable("cancelled")
    if time.monotonic() >= deadline:
        raise Unavailable("deadline")


def run_bounded(
    argv: tuple[str, ...], payload: bytes, deadline: float, cancel: Event | None = None
) -> bytes:
    """argv is constructed by trusted host code only, never from CodeQuery."""
    checkpoint(deadline, cancel)
    if os.name != "posix" or len(payload) > 8 * 1024 * 1024:
        raise Unavailable("runtime_unavailable")
    output = bytearray()
    emitted = 0
    # Empty private cwd: all source bytes are from the immutable captured snapshot.
    with tempfile.TemporaryDirectory(prefix="jitmind-code-") as cwd:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
                env={
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "HOME": cwd,
                    "TMPDIR": cwd,
                },
                close_fds=True,
            )
        except OSError:
            raise Unavailable("runtime_unavailable") from None
        with selectors.DefaultSelector() as selector:
            try:
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    os.set_blocking(pipe.fileno(), False)
                selector.register(proc.stdin, selectors.EVENT_WRITE, "input")
                selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
                selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
                offset = 0
                while selector.get_map():
                    checkpoint(deadline, cancel)
                    for key, _ in selector.select(
                        min(0.02, max(0, deadline - time.monotonic()))
                    ):
                        pipe = key.fileobj
                        if key.data == "input":
                            try:
                                offset += os.write(
                                    pipe.fileno(), payload[offset : offset + 16384]
                                )
                            except BlockingIOError:
                                continue
                            except BrokenPipeError:
                                offset = len(payload)
                            if offset == len(payload):
                                selector.unregister(pipe)
                                pipe.close()
                        else:
                            try:
                                chunk = os.read(pipe.fileno(), 16384)
                            except BlockingIOError:
                                continue
                            if not chunk:
                                selector.unregister(pipe)
                                pipe.close()
                                continue
                            emitted += len(chunk)
                            if emitted > MAX_RESPONSE_BYTES:
                                raise Unavailable("adapter_output_budget")
                            if key.data == "stdout":
                                output.extend(chunk)
                while proc.poll() is None:
                    checkpoint(deadline, cancel)
                    time.sleep(min(0.01, max(0, deadline - time.monotonic())))
                if proc.returncode:
                    raise Unavailable("adapter_failed")
                return bytes(output)
            finally:
                # Kill group even if leader exited: a grandchild may still own pipes.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    pipe.close()


class NodeParser:
    """Application configuration, not a request field. Runtime provisioned separately."""

    def __init__(self, node_binary: str, adapter_dir: str):
        node = Path(node_binary)
        directory = Path(adapter_dir)
        if not node.is_absolute() or not directory.is_absolute():
            raise ValueError("Trusted absolute runtime paths required")
        try:
            node = node.resolve(strict=True)
            if not node.is_file() or not os.access(node, os.X_OK):
                raise CapabilityUnavailable("node_executable_unavailable")
        except (OSError, RuntimeError):
            raise CapabilityUnavailable("node_executable_unavailable") from None
        try:
            self.directory = directory.resolve(strict=True)
            if not self.directory.is_dir():
                raise CapabilityUnavailable("adapter_directory_unavailable")
        except (OSError, RuntimeError):
            raise CapabilityUnavailable("adapter_directory_unavailable") from None
        self.node = str(node)
        runner = self.directory / "runner.mjs"
        try:
            if (
                runner.is_symlink()
                or not runner.is_file()
                or not os.access(runner, os.R_OK)
            ):
                raise CapabilityUnavailable("adapter_runner_unavailable")
        except OSError:
            raise CapabilityUnavailable("adapter_runner_unavailable") from None
        self.argv = (self.node, "--max-old-space-size=128", str(runner))

    def parse(
        self, payload: bytes, deadline: float, cancel: Event | None = None
    ) -> bytes:
        return run_bounded(self.argv, payload, deadline, cancel)
