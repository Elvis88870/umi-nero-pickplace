#!/usr/bin/env python3
"""Standalone CPU image decoder for selected fisheye/D405 frames."""

from __future__ import annotations

import argparse
from multiprocessing.connection import Listener
import os
import time
from typing import Any

import cv2
import numpy as np


def decode_request(request: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic_ns()
    decode_started = time.perf_counter_ns()
    response_frames = []
    fish_decode_ms = []
    yuyv_decode_ms = []
    try:
        for frame in request["fisheye_frames"]:
            tick = time.perf_counter_ns()
            encoded = np.frombuffer(frame["payload"], dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(
                    f"JPEG decode failed for sequence {frame['sequence']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elapsed = (time.perf_counter_ns() - tick) / 1e6
            fish_decode_ms.append(elapsed)
            response_frames.append({
                "role": "fisheye",
                "sequence": int(frame["sequence"]),
                "timestamp": float(frame["timestamp"]),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "rgb": np.ascontiguousarray(rgb).tobytes(),
                "decode_ms": elapsed,
            })
        for frame in request["d405_frames"]:
            tick = time.perf_counter_ns()
            payload = np.frombuffer(frame["payload"], dtype=np.uint8)
            rows = payload.reshape(int(frame["height"]), int(frame["stride"]))
            yuyv = rows[:, :int(frame["width"]) * 2].reshape(
                int(frame["height"]), int(frame["width"]), 2)
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elapsed = (time.perf_counter_ns() - tick) / 1e6
            yuyv_decode_ms.append(elapsed)
            response_frames.append({
                "role": "d405",
                "sequence": int(frame["sequence"]),
                "timestamp": float(frame["timestamp"]),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "rgb": np.ascontiguousarray(rgb).tobytes(),
                "decode_ms": elapsed,
            })
    except Exception as exc:
        return {
            "request_id": request.get("request_id"),
            "ok": False,
            "error": str(exc),
            "queue_wait_ms": (
                decode_started - int(request["enqueued_monotonic_ns"])) / 1e6,
            "worker_total_ms": (time.perf_counter_ns() - decode_started) / 1e6,
        }
    return {
        "request_id": request.get("request_id"),
        "ok": True,
        "frames": response_frames,
        "queue_wait_ms": (
            decode_started - int(request["enqueued_monotonic_ns"])) / 1e6,
        "decode_total_ms": (time.perf_counter_ns() - decode_started) / 1e6,
        "fish_decode_ms": fish_decode_ms,
        "yuyv_decode_ms": yuyv_decode_ms,
        "worker_total_ms": (time.monotonic_ns() - started) / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/tmp/umi_image_decoder.sock")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    try:
        os.unlink(args.socket)
    except FileNotFoundError:
        pass
    listener = Listener(args.socket, family="AF_UNIX")
    print(
        f"[decoder] pid={os.getpid()} affinity="
        f"{sorted(os.sched_getaffinity(0))} socket={args.socket}",
        flush=True,
    )
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    if request is None:
                        break
                    connection.send(decode_request(request))
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass
            finally:
                connection.close()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        try:
            os.unlink(args.socket)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
