"""
src/height/fit_client.py
========================
Talks to src/fit/fit_worker.py, which runs in a *different* Python environment
(Python 3.7 + TensorFlow 1.15). The main pipeline keeps the webcam and the window;
this client only ships a frame + height to the worker and stores the reply.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2


def _send(sock, header, blob=b""):
    j = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">II", len(j), len(blob)) + j + blob)


def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("fit worker closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock):
    jl, bl = struct.unpack(">II", _recvn(sock, 8))
    header = json.loads(_recvn(sock, jl).decode("utf-8"))
    blob = _recvn(sock, bl) if bl else b""
    return header, blob


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FitClient:
    """States: starting -> ready <-> running ; error (worker unusable)."""

    def __init__(self, hmr_python: str, fit_dir: str, extra_args: Optional[List[str]] = None,
                 connect_timeout_s: float = 900.0):
        self.hmr_python = hmr_python
        self.fit_dir = fit_dir
        self.extra_args = extra_args or []
        self.connect_timeout_s = connect_timeout_s
        self.log_path = os.path.join(fit_dir, "fit_worker.log")

        self._lock = threading.Lock()
        self._state = "starting"
        self._message = ""
        self._result: Optional[Dict[str, Any]] = None
        self._sock: Optional[socket.socket] = None
        self._proc: Optional[subprocess.Popen] = None
        self._port = _free_port()

    # ---- lifecycle ----
    def start(self) -> None:
        script = os.path.join(self.fit_dir, "fit_worker.py")
        if not os.path.isfile(self.hmr_python):
            self._fail(f"HMR python not found: {self.hmr_python}")
            return
        if not os.path.isfile(script):
            self._fail(f"fit_worker.py not found in {self.fit_dir}")
            return
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        log = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            [self.hmr_python, script, "--port", str(self._port)] + self.extra_args,
            cwd=self.fit_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
        threading.Thread(target=self._connect_loop, daemon=True).start()

    def _connect_loop(self) -> None:
        deadline = time.time() + self.connect_timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._fail(f"fit worker exited (code {self._proc.returncode}); see {self.log_path}")
                return
            try:
                sock = socket.create_connection(("127.0.0.1", self._port), timeout=2)
                sock.settimeout(900)
                with self._lock:
                    self._sock = sock
                    self._state = "ready"
                return
            except OSError:
                time.sleep(1.0)
        self._fail("timed out waiting for the fit worker to load its models")

    def close(self) -> None:
        try:
            if self._sock is not None:
                try:
                    _send(self._sock, {"cmd": "quit"})
                except OSError:
                    pass
                self._sock.close()
        finally:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

    # ---- requests ----
    def request(self, frame_bgr, height_cm: float) -> bool:
        """Start a fit job on a background thread. Returns False if the worker is not ready."""
        with self._lock:
            if self._state != "ready":
                return False
            self._state = "running"
            self._message = ""
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            with self._lock:
                self._state = "ready"
                self._message = "could not encode frame"
            return False
        threading.Thread(target=self._do_request, args=(jpg.tobytes(), float(height_cm)),
                         daemon=True).start()
        return True

    def _do_request(self, jpg: bytes, height_cm: float) -> None:
        try:
            _send(self._sock, {"cmd": "measure", "height_cm": height_cm}, jpg)
            reply, _ = _recv(self._sock)
        except Exception as e:  # worker died / socket error
            self._fail(f"{type(e).__name__}: {e} (see {self.log_path})")
            return
        with self._lock:
            if reply.get("ok"):
                self._result = {"height_cm": height_cm,
                                "measurements": reply.get("measurements", {}),
                                "elapsed_s": reply.get("elapsed_s")}
                self._message = ""
            else:
                self._message = str(reply.get("error", "fit failed"))
            self._state = "ready"

    def _fail(self, msg: str) -> None:
        with self._lock:
            self._state = "error"
            self._message = msg

    def set_note(self, msg: str) -> None:
        with self._lock:
            if self._state != "error":
                self._message = msg

    def snapshot(self) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        with self._lock:
            return self._state, self._message, self._result
