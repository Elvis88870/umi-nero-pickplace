"""Original UMI-style 10 Hz policy and replanning loop."""

from __future__ import annotations

from argparse import Namespace
import json
import time
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from env.nero_ros_umi_env import NeroRosUmiEnv
from env.ros_action_scheduler import PlanRejected
from env.ros_observation_buffer import ObservationUnavailable
from runtime.policy_runtime import PolicyRuntime
from runtime.preview_logger import PreviewLogger


def episode_start_pose(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Build the relative-pose anchor expected by the UMI policy."""
    return np.concatenate([
        obs["robot0_eef_pos"][-1],
        obs["robot0_eef_rot_axis_angle"][-1],
    ]).astype(np.float64)


class UmiControlLoop:
    """Run inference at the original UMI cadence and hand plans to the env."""

    def __init__(
        self,
        runtime: PolicyRuntime,
        env: NeroRosUmiEnv,
        logger: PreviewLogger,
        config: Dict[str, Any],
        args: Namespace,
    ) -> None:
        self.runtime = runtime
        self.env = env
        self.logger = logger
        self.config = config
        self.args = args
        self.action_interval = self._validate_timing()
        self.diagnostic_interval = float(
            config["observation"]["diagnostic_interval_sec"])

    def _validate_timing(self) -> float:
        action_interval = float(
            self.config["action"]["action_interval_sec"])
        expected_interval = 1.0 / self.args.frequency
        if abs(action_interval - expected_interval) > 1e-6:
            raise ValueError(
                f"checkpoint action interval {action_interval:.6f}s does not "
                f"match --frequency {self.args.frequency:g} Hz")
        expected_replan_interval = (
            self.args.steps_per_inference * action_interval)
        configured_replan_interval = float(
            self.config["action"]["replan_interval_sec"])
        if abs(configured_replan_interval - expected_replan_interval) > 1e-6:
            raise ValueError(
                f"checkpoint replan interval {configured_replan_interval:.6f}s "
                f"does not match steps_per_inference/frequency "
                f"{self.args.steps_per_inference}/{self.args.frequency:g}="
                f"{expected_replan_interval:.6f}s")
        return action_interval

    def _print_action_diagnostics(self, actions: np.ndarray) -> None:
        xyz = actions[:, :3]
        step_sizes = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        rotations = Rotation.from_rotvec(actions[:, 3:6])
        rotation_steps = (
            (rotations[:-1].inv() * rotations[1:]).magnitude()
            if len(rotations) > 1 else np.asarray([]))
        print(
            "[action diagnostics] "
            f"xyz_min={np.min(xyz, axis=0).round(5).tolist()} "
            f"xyz_max={np.max(xyz, axis=0).round(5).tolist()} "
            f"first5_xyz={xyz[:5].round(5).tolist()} "
            f"max_step_m="
            f"{float(step_sizes.max()) if step_sizes.size else 0.0:.6f} "
            f"max_rotation_step_rad="
            f"{float(rotation_steps.max()) if rotation_steps.size else 0.0:.6f} "
            f"gripper_min={float(actions[:, 6].min()):.5f} "
            f"gripper_max={float(actions[:, 6].max()):.5f}",
            flush=True)

    def _print_diagnostics(self, name: str) -> None:
        diagnostics = self.env.diagnostics()
        print(
            f"[{name}] "
            + json.dumps(diagnostics, separators=(",", ":")),
            flush=True)

    def _write_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        if self.args.diagnostic_json is None:
            return
        path = self.args.diagnostic_json.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[observation diagnostics] wrote {path}", flush=True)

    def run(self) -> int:
        index = 0
        controller_primed = False
        last_diagnostic = time.monotonic()
        self.env.start()
        try:
            if self.args.control:
                verification_facts = self.env.verify_live_control_facts(
                    self.args.timeout)
                print(
                    f"[control verification] tcp_frame="
                    f"{verification_facts['tcp_frame']} "
                    f"gripper={verification_facts['gripper_value']:.6f}; "
                    "configured control publishers created",
                    flush=True)

            first_obs = self.env.wait_for_obs(self.args.timeout)
            start_pose = episode_start_pose(first_obs)
            warmup_actions, warmup_ms = self.runtime.infer(
                first_obs, start_pose)
            del warmup_actions
            print(
                f"[warmup] inference={warmup_ms:.1f}ms; "
                "not submitted to the robot",
                flush=True)

            try:
                obs = self.env.get_obs()
            except ObservationUnavailable as exc:
                print(
                    f"[observation rejected after warmup] {exc}",
                    flush=True)
                obs = self.env.wait_for_obs(self.args.timeout)
            control_loop_start = time.monotonic()
            control_iteration = 0
            while True:
                if self.args.max_cycles and index >= self.args.max_cycles:
                    break
                deadline_extension = 0.0
                if self.args.control and not controller_primed:
                    prime = self.env.prime_control_from_feedback()
                    controller_primed = True
                    print(
                        f"[controller priming] tcp={prime['tcp_xyz']} "
                        f"gripper={prime['gripper']:.6f} "
                        f"hold={prime['hold_sec']:.2f}s "
                        f"messages={prime['published_messages']}",
                        flush=True)
                    # Prime consumes wall time. Rebuild the observation so the
                    # model action timestamps start from the post-prime state.
                    try:
                        obs = self.env.get_obs()
                    except ObservationUnavailable as exc:
                        print(
                            f"[observation rejected after priming] {exc}",
                            flush=True)
                        obs = self.env.wait_for_obs(self.args.timeout)
                actions, inference_ms = self.runtime.infer(obs, start_pose)
                action_timestamps = (
                    float(obs["timestamp"][-1])
                    + np.arange(actions.shape[0], dtype=np.float64)
                    * self.action_interval)
                self.logger.log(
                    index, obs, actions, action_timestamps, inference_ms)
                if self.args.print_action_diagnostics:
                    self._print_action_diagnostics(actions)
                print(
                    f"[infer {index:04d}] obs={obs['timestamp'][-1]:.6f} "
                    f"inference={inference_ms:.1f}ms "
                    f"first_action={action_timestamps[0]:.6f} "
                    f"mode={'CONTROL' if self.args.control else 'PREVIEW'}",
                    flush=True)
                if self.args.control:
                    try:
                        report = self.env.exec_actions(
                            actions,
                            action_timestamps,
                            observation_timestamp=float(obs["timestamp"][-1]),
                        )
                    except PlanRejected as exc:
                        print(f"[schedule] rejected: {exc}", flush=True)
                        # A published hold keeps QP initialized. If control was
                        # actually released, seed measured feedback before the
                        # next plan so startup cannot see a far-ahead target.
                        controller_primed = self.env.control_is_active
                    else:
                        print(
                            f"[schedule] input={report.input_actions} "
                            f"accepted={report.accepted_actions} "
                            f"expired={report.dropped_expired} "
                            f"minimum_lead={report.dropped_minimum_lead} "
                            f"replaced_future="
                            f"{report.replaced_future_actions} "
                            f"first_exec="
                            f"{report.first_execution_timestamp:.6f} "
                            f"last_exec="
                            f"{report.last_execution_timestamp:.6f}",
                            flush=True)
                        if report.translation_limited_actions:
                            print(
                                "[schedule] translation limited: "
                                f"actions={report.translation_limited_actions} "
                                f"input_max={report.input_max_translation_step_m:.4f}m "
                                f"output_max={report.output_max_translation_step_m:.4f}m",
                                flush=True)
                        if report.rotation_limited_actions:
                            print(
                                "[schedule] rotation limited: "
                                f"actions={report.rotation_limited_actions} "
                                f"input_max={report.input_max_rotation_step_rad:.4f}rad "
                                f"output_max={report.output_max_rotation_step_rad:.4f}rad",
                                flush=True)
                        if self.args.control_action_limit > 0.0:
                            remaining = max(
                                0.0,
                                report.last_execution_timestamp
                                - report.scheduler_now,
                            )
                            publish_period = 1.0 / float(
                                self.config["action"]["publish_rate_hz"])
                            deadline_extension = min(
                                2.0, remaining + publish_period)

                index += 1
                control_iteration += self.args.steps_per_inference
                if (
                    self.diagnostic_interval > 0.0
                    and time.monotonic() - last_diagnostic
                    >= self.diagnostic_interval
                ):
                    self._print_diagnostics("observation diagnostics")
                    last_diagnostic = time.monotonic()
                deadline = (
                    control_loop_start
                    + control_iteration * self.action_interval
                    - self.args.frame_latency)
                if deadline_extension > 0.0:
                    deadline = max(
                        deadline,
                        time.monotonic() + deadline_extension,
                    )
                while time.monotonic() < deadline:
                    self.env.spin_once(
                        min(0.02, deadline - time.monotonic()))
                try:
                    obs = self.env.get_obs()
                except ObservationUnavailable as exc:
                    print(f"[observation rejected] {exc}", flush=True)
                    obs = self.env.wait_for_obs(self.args.timeout)
        except KeyboardInterrupt:
            print("Interrupted.", flush=True)
        finally:
            diagnostics = self.env.diagnostics()
            print(
                "[observation diagnostics final] "
                + json.dumps(diagnostics, separators=(",", ":")),
                flush=True)
            self._write_diagnostics(diagnostics)
            self.env.stop()

        if self.args.preview:
            print("[preview only] no robot command was published", flush=True)
        else:
            print(
                "[control] scheduler closed; teleop_active=False",
                flush=True)
        return 0
