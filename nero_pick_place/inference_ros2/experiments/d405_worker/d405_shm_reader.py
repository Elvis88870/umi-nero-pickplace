"""Read D405 YUYV frames from the C++ shared-memory ring buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from multiprocessing import resource_tracker, shared_memory
import struct
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


MAGIC = 0x4434303553484D31
VERSION = 1
HEADER_SIZE = 128
SLOT_HEADER_SIZE = 128
ENCODING_YUYV = 1
STATE_READY = 2

HEADER_FORMAT = "<QIIIIIIQQQQQQQ"
SLOT_FORMAT = "<QIIQQQQIIIIQQII"


@dataclass(frozen=True)
class FrameMetadata:
    slot_index: int
    generation: int
    sequence: int
    capture_ros_ns: int
    capture_monotonic_ns: int
    v4l2_timestamp_ns: int
    width: int
    height: int
    stride: int
    encoding: int
    payload_size: int
    checksum: int
    timestamp_flag: int

    @property
    def timestamp(self) -> float:
        return float(self.capture_ros_ns) * 1e-9


@dataclass(frozen=True)
class SharedImageRecord:
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class HistoryPlan:
    latest_sequence: int
    targets: Tuple[float, ...]
    frames: Tuple[FrameMetadata, ...]


def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    p50, p95, p99 = np.percentile(array, [50.0, 95.0, 99.0])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(array.max()),
    }


class SharedMemoryD405Reader:
    """Cross-process reader for the D405 seqlock ring."""

    def __init__(self, name: str, open_timeout_sec: float = 3.0):
        normalized = str(name)
        if not normalized.startswith("/"):
            raise ValueError("shared-memory name must begin with '/'")
        shared_name = normalized[1:]
        deadline = time.monotonic() + max(0.0, open_timeout_sec)
        last_error: Optional[Exception] = None
        while True:
            try:
                self._shm = shared_memory.SharedMemory(
                    name=shared_name, create=False)
                # The writer owns the POSIX shm lifetime. Prevent this reader
                # from unlinking it when the process exits.
                resource_tracker.unregister(
                    self._shm._name, "shared_memory")
                break
            except FileNotFoundError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"D405 shared memory {normalized!r} is unavailable; "
                        "start d405_shm_writer first") from exc
                time.sleep(0.05)
        self.name = normalized
        self._closed = False
        self._read_attempts = 0
        self._decoded_frames = 0
        self._ring_overwrite_retries = 0
        self._history_rejections = 0
        self._last_total_frames: Optional[int] = None
        self._last_writer_time: Optional[float] = None
        self._writer_fps: Optional[float] = None
        self._frame_ages_ms: deque[float] = deque(maxlen=100000)
        self._first_read_time: Optional[float] = None
        self._last_read_time: Optional[float] = None
        self._last_master_sequence: Optional[int] = None
        self._validate_header()

    def _header(self) -> Tuple[int, ...]:
        return struct.unpack_from(HEADER_FORMAT, self._shm.buf, 0)

    def _validate_header(self) -> None:
        values = self._header()
        (
            magic, version, header_size, slot_count, width, height, encoding,
            payload_capacity, slot_stride, _latest, _total, _heartbeat,
            _errors, _created,
        ) = values
        if magic != MAGIC:
            raise RuntimeError("D405 shared-memory magic does not match")
        if version != VERSION:
            raise RuntimeError(
                f"D405 shared-memory version {version} != {VERSION}")
        if header_size != HEADER_SIZE or slot_count < 2:
            raise RuntimeError("D405 shared-memory header is invalid")
        if encoding != ENCODING_YUYV:
            raise RuntimeError(
                f"unsupported D405 shared-memory encoding {encoding}")
        if width <= 0 or height <= 0 or payload_capacity < width * height * 2:
            raise RuntimeError("D405 shared-memory dimensions are invalid")
        if slot_stride < SLOT_HEADER_SIZE + payload_capacity:
            raise RuntimeError("D405 shared-memory slot stride is invalid")
        required_size = HEADER_SIZE + slot_count * slot_stride
        if self._shm.size < required_size:
            raise RuntimeError(
                f"D405 shared memory is short: {self._shm.size} < {required_size}")
        self.slot_count = int(slot_count)
        self.width = int(width)
        self.height = int(height)
        self.payload_capacity = int(payload_capacity)
        self.slot_stride = int(slot_stride)

    def _slot_offset(self, slot_index: int) -> int:
        return HEADER_SIZE + slot_index * self.slot_stride

    def _payload_offset(self, slot_index: int) -> int:
        return self._slot_offset(slot_index) + SLOT_HEADER_SIZE

    def _slot_metadata(
        self,
        slot_index: int,
        generation: Optional[int] = None,
    ) -> Optional[FrameMetadata]:
        offset = self._slot_offset(slot_index)
        values = struct.unpack_from(SLOT_FORMAT, self._shm.buf, offset)
        (
            current_generation, state, _reserved0, capture_ros_ns,
            capture_monotonic_ns, v4l2_timestamp_ns, sequence, width, height,
            stride, encoding, payload_size, checksum, timestamp_flag,
            _reserved1,
        ) = values
        if state != STATE_READY or current_generation <= 0 or (
                current_generation & 1):
            return None
        if generation is not None and current_generation != generation:
            return None
        if (
            width != self.width
            or height != self.height
            or encoding != ENCODING_YUYV
            or stride < width * 2
            or payload_size < stride * height
            or payload_size > self.payload_capacity
        ):
            return None
        return FrameMetadata(
            slot_index=slot_index,
            generation=int(current_generation),
            sequence=int(sequence),
            capture_ros_ns=int(capture_ros_ns),
            capture_monotonic_ns=int(capture_monotonic_ns),
            v4l2_timestamp_ns=int(v4l2_timestamp_ns),
            width=int(width),
            height=int(height),
            stride=int(stride),
            encoding=int(encoding),
            payload_size=int(payload_size),
            checksum=int(checksum),
            timestamp_flag=int(timestamp_flag),
        )

    def snapshot_metadata(self) -> List[FrameMetadata]:
        frames: List[FrameMetadata] = []
        for slot_index in range(self.slot_count):
            offset = self._slot_offset(slot_index)
            generation, state = struct.unpack_from(
                "<QI", self._shm.buf, offset)
            if state != STATE_READY or generation <= 0 or generation & 1:
                continue
            item = self._slot_metadata(slot_index, generation=int(generation))
            if item is not None:
                frames.append(item)
        frames.sort(key=lambda item: (item.capture_ros_ns, item.sequence))
        return frames

    def latest_metadata(self) -> Optional[FrameMetadata]:
        frames = self.snapshot_metadata()
        return frames[-1] if frames else None

    def _read_payload(self, item: FrameMetadata) -> Optional[np.ndarray]:
        for _attempt in range(5):
            offset = self._slot_offset(item.slot_index)
            generation_before, state_before = struct.unpack_from(
                "<QI", self._shm.buf, offset)
            if (
                state_before != STATE_READY
                or generation_before != item.generation
                or generation_before & 1
            ):
                self._ring_overwrite_retries += 1
                continue
            payload = np.frombuffer(
                self._shm.buf,
                dtype=np.uint8,
                count=item.payload_size,
                offset=self._payload_offset(item.slot_index),
            ).copy()
            generation_after, state_after = struct.unpack_from(
                "<QI", self._shm.buf, offset)
            if (
                state_after == STATE_READY
                and generation_after == generation_before
            ):
                return payload
            self._ring_overwrite_retries += 1
        return None

    def decode_frame(
        self,
        item: FrameMetadata,
    ) -> Optional[SharedImageRecord]:
        payload = self._read_payload(item)
        if payload is None:
            return None
        rows = payload.reshape(item.height, item.stride)
        yuyv = rows[:, :item.width * 2].reshape(
            item.height, item.width, 2)
        try:
            rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
        except cv2.error:
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._decoded_frames += 1
        return SharedImageRecord(
            timestamp=item.timestamp,
            image=np.ascontiguousarray(rgb),
        )

    @staticmethod
    def _nearest_index(timestamps: np.ndarray, target: float) -> int:
        position = int(np.searchsorted(timestamps, target))
        candidates: List[int] = []
        if position < timestamps.size:
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        if not candidates:
            raise RuntimeError("empty D405 metadata history")
        return min(candidates, key=lambda index: abs(
            float(timestamps[index]) - target))

    def plan_history(
        self,
        history_horizon: int,
        history_interval_sec: float,
    ) -> Optional[HistoryPlan]:
        """Select real frames by timestamp without copying image payloads."""
        frames = self.snapshot_metadata()
        if not frames:
            self._history_rejections += 1
            return None
        latest = max(frames, key=lambda item: item.sequence)
        targets = latest.timestamp - (
            np.arange(history_horizon - 1, -1, -1, dtype=np.float64)
            * history_interval_sec)
        timestamps = np.asarray(
            [item.timestamp for item in frames], dtype=np.float64)
        selected_frames: List[FrameMetadata] = []
        for target in targets:
            selected_frames.append(frames[
                self._nearest_index(timestamps, float(target))])
        unique_frames = list(dict.fromkeys(selected_frames))
        return HistoryPlan(
            latest_sequence=latest.sequence,
            targets=tuple(float(value) for value in targets),
            frames=tuple(unique_frames),
        )

    def decode_plan(self, plan: HistoryPlan) -> List[SharedImageRecord]:
        if plan.latest_sequence == self._last_master_sequence:
            return []
        records = []
        for frame in plan.frames:
            record = self.decode_frame(frame)
            if record is not None:
                records.append(record)
        if not records:
            self._history_rejections += 1
            return []
        records.sort(key=lambda item: item.timestamp)
        self._last_master_sequence = plan.latest_sequence
        self._read_attempts += 1
        now = time.monotonic()
        if self._first_read_time is None:
            self._first_read_time = now
        self._last_read_time = now
        self._frame_ages_ms.append(
            (time.monotonic_ns() - plan.frames[-1].capture_monotonic_ns) / 1e6)
        return records

    def read_history(
        self,
        history_horizon: int,
        history_interval_sec: float,
    ) -> List[SharedImageRecord]:
        """Plan and decode a history; mainly for standalone diagnostics."""
        plan = self.plan_history(
            history_horizon, history_interval_sec)
        return [] if plan is None else self.decode_plan(plan)

    def _update_writer_stats(self) -> None:
        header = self._header()
        total_frames = int(header[10])
        now = time.monotonic()
        if self._last_total_frames is not None and self._last_writer_time is not None:
            elapsed = now - self._last_writer_time
            if elapsed > 0.0:
                self._writer_fps = (
                    total_frames - self._last_total_frames) / elapsed
        self._last_total_frames = total_frames
        self._last_writer_time = now

    def diagnostics(self) -> Dict[str, Any]:
        self._update_writer_stats()
        elapsed = 0.0
        if self._first_read_time is not None and self._last_read_time is not None:
            elapsed = self._last_read_time - self._first_read_time
        reader_fps = (
            self._decoded_frames / elapsed if elapsed > 0.0 else None)
        frames = self.snapshot_metadata()
        latest_age_ms = None
        if frames:
            latest_age_ms = (
                time.monotonic_ns() - frames[-1].capture_monotonic_ns) / 1e6
        header = self._header()
        return {
            "name": self.name,
            "writer_fps": self._writer_fps,
            "writer_total_frames": int(header[10]),
            "latest_sequence": int(header[9]),
            "latest_frame_age_ms": latest_age_ms,
            "reader_history_requests": self._read_attempts,
            "reader_decoded_frames": self._decoded_frames,
            "reader_decoded_fps": reader_fps,
            "ring_overwrite_retries": self._ring_overwrite_retries,
            "history_rejections": self._history_rejections,
            "selected_frame_age_ms": _percentiles(self._frame_ages_ms),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()
