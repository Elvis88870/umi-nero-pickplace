"""Client for the isolated CPU image decoder process."""

from __future__ import annotations

from multiprocessing.connection import Client
import time
from typing import Any, Dict, List, Sequence

try:
    from .d405_shm_reader import FrameMetadata
except ImportError:
    from d405_shm_reader import FrameMetadata


def _frame_request(frame, role: str) -> Dict[str, Any]:
    return {
        "role": role,
        "sequence": int(frame.sequence),
        "timestamp": float(frame.timestamp),
        "width": int(getattr(frame, "width", 0)),
        "height": int(getattr(frame, "height", 0)),
        "stride": int(getattr(frame, "stride", 0)),
        "payload": frame.payload,
    }


class ImageDecoderClient:
    def __init__(
        self,
        socket_path: str,
        timeout_sec: float = 2.0,
    ):
        self.socket_path = socket_path
        self.timeout_sec = float(timeout_sec)
        self.connection = Client(socket_path, family="AF_UNIX")
        self.request_id = 0
        self.last_queue_wait_ms = None
        self.last_main_wait_ms = None
        self.last_decode_total_ms = None
        self.decoded_frames = {"fisheye": 0, "d405": 0}
        self.total_fish_decode_ms = 0.0
        self.total_yuyv_decode_ms = 0.0

    def decode(
        self,
        fisheye_frames: Sequence[Any],
        d405_frames: Sequence[Any],
    ) -> Dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        request = {
            "request_id": request_id,
            "enqueued_monotonic_ns": time.monotonic_ns(),
            "fisheye_frames": [
                _frame_request(frame, "fisheye")
                for frame in fisheye_frames
            ],
            "d405_frames": [
                _frame_request(frame, "d405")
                for frame in d405_frames
            ],
        }
        started = time.perf_counter_ns()
        self.connection.send(request)
        poller = __import__("multiprocessing.connection", fromlist=["wait"])
        if not poller.wait(
            [self.connection], timeout=self.timeout_sec
        ):
            raise TimeoutError(
                f"decoder worker exceeded {self.timeout_sec:.3f}s")
        response = self.connection.recv()
        self.last_main_wait_ms = (
            time.perf_counter_ns() - started) / 1e6
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"decoder request id mismatch: "
                f"{response.get('request_id')} != {request_id}")
        if not response.get("ok"):
            raise RuntimeError(
                f"decoder worker failed: {response.get('error')}")
        expected = {
            (frame["role"], frame["sequence"], frame["timestamp"])
            for frame in request["fisheye_frames"] + request["d405_frames"]
        }
        returned = {
            (
                frame["role"], int(frame["sequence"]), float(frame["timestamp"]))
            for frame in response["frames"]
        }
        if expected != returned:
            raise RuntimeError(
                "decoder response sequence/timestamp mismatch")
        self.last_queue_wait_ms = float(response["queue_wait_ms"])
        self.last_decode_total_ms = float(response["decode_total_ms"])
        for frame in response["frames"]:
            role = frame["role"]
            self.decoded_frames[role] += 1
            if role == "fisheye":
                self.total_fish_decode_ms += float(frame["decode_ms"])
            else:
                self.total_yuyv_decode_ms += float(frame["decode_ms"])
        return response

    def diagnostics(self) -> Dict[str, Any]:
        total = sum(self.decoded_frames.values())
        return {
            "requests": self.request_id,
            "decoded_frames": dict(self.decoded_frames),
            "last_queue_wait_ms": self.last_queue_wait_ms,
            "last_main_wait_ms": self.last_main_wait_ms,
            "last_decode_total_ms": self.last_decode_total_ms,
            "mean_fisheye_decode_ms": (
                self.total_fish_decode_ms / self.decoded_frames["fisheye"]
                if self.decoded_frames["fisheye"] else None),
            "mean_d405_decode_ms": (
                self.total_yuyv_decode_ms / self.decoded_frames["d405"]
                if self.decoded_frames["d405"] else None),
            "total_decoded_frames": total,
        }

    def close(self) -> None:
        try:
            self.connection.send(None)
        except Exception:
            pass
        self.connection.close()
