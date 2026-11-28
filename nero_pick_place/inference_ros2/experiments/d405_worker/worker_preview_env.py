"""Preview-only environment using the isolated image decoder."""

from __future__ import annotations

import threading

try:
    from .worker_observation_buffer import WorkerObservationBuffer
except ImportError:
    from worker_observation_buffer import WorkerObservationBuffer


class WorkerPreviewEnv:
    def __init__(self, config):
        import rclpy

        self.rclpy = rclpy
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.buffer = WorkerObservationBuffer(config)
        self.executor = None
        self.thread = None
        self.started = False

    def start(self):
        if self.started:
            return
        from rclpy.executors import SingleThreadedExecutor

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.buffer.node)
        self.thread = threading.Thread(
            target=self._spin, name="worker-preview-state", daemon=True)
        self.thread.start()
        self.started = True

    def _spin(self):
        from rclpy.executors import ExternalShutdownException

        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    def stop(self):
        self.started = False
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        self.buffer.destroy()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()

    def get_obs(self, count_attempt: bool = True):
        return self.buffer.get_obs(count_attempt=count_attempt)

    def wait_for_obs(self, timeout_sec: float):
        return self.buffer.wait_for_obs(timeout_sec)

    def diagnostics(self):
        return self.buffer.diagnostics()

    @property
    def latest_tcp(self):
        return self.buffer.tcp[-1] if self.buffer.tcp else None

    @property
    def latest_gripper(self):
        return self.buffer.gripper[-1] if self.buffer.gripper else None
