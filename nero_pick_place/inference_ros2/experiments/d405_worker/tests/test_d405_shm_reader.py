import os
import mmap
from pathlib import Path
import struct
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d405_shm_reader import (
    ENCODING_YUYV,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAGIC,
    SLOT_FORMAT,
    SLOT_HEADER_SIZE,
    STATE_READY,
    VERSION,
    SharedMemoryD405Reader,
)


def _write_slot(buffer, slot_offset, sequence, ros_ns, payload):
    struct.pack_into(
        "<QI", buffer, slot_offset, 1, 1)
    struct.pack_into(
        SLOT_FORMAT,
        buffer,
        slot_offset,
        1,
        2,
        0,
        ros_ns,
        time.monotonic_ns(),
        time.monotonic_ns(),
        sequence,
        4,
        2,
        8,
        ENCODING_YUYV,
        len(payload),
        0,
        0x2000,
        0,
    )
    start = slot_offset + SLOT_HEADER_SIZE
    buffer[start:start + len(payload)] = payload
    struct.pack_into(
        "<QI", buffer, slot_offset, 2, STATE_READY)


class D405SharedMemoryReaderTest(unittest.TestCase):
    def test_reads_real_yuyv_frames_and_preserves_timestamp(self):
        name = f"umi_d405_test_{os.getpid()}"
        slot_stride = 4096
        path = f"/dev/shm/{name}"
        size = HEADER_SIZE + 2 * slot_stride
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.ftruncate(fd, size)
        shm_buffer = mmap.mmap(fd, size)
        try:
            struct.pack_into(
                HEADER_FORMAT, shm_buffer, 0, MAGIC, VERSION, HEADER_SIZE,
                2, 4, 2, ENCODING_YUYV, 16, slot_stride,
                2, 2, time.monotonic_ns(), 0, time.monotonic_ns())
            yuyv = np.full(2 * 8, 128, dtype=np.uint8)
            _write_slot(
                shm_buffer, HEADER_SIZE, 1, 123456789000, yuyv.tobytes())
            _write_slot(
                shm_buffer, HEADER_SIZE + slot_stride, 2,
                123456889000, yuyv.tobytes())

            reader = SharedMemoryD405Reader("/" + name)
            frames = reader.read_history(2, 0.1)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].image.shape, (2, 4, 3))
            self.assertAlmostEqual(
                frames[-1].timestamp, 123456889000 * 1e-9)
            self.assertEqual(reader.read_history(2, 0.1), [])
            diagnostics = reader.diagnostics()
            self.assertEqual(diagnostics["reader_decoded_frames"], 2)
            self.assertEqual(diagnostics["ring_overwrite_retries"], 0)
            reader.close()
        finally:
            shm_buffer.close()
            os.close(fd)
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
