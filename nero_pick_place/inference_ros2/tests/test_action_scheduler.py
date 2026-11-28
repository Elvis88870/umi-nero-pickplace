from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.nero_ros_umi_env import validate_control_request  # noqa: E402
from env.grasp_verifier import GraspVerifier  # noqa: E402
from env.grasp_lift_guard import GraspLiftGuard  # noqa: E402
from env.ros_action_scheduler import (  # noqa: E402
    ActionScheduler,
    PlanRejected,
    RosActionScheduler,
    find_grasp_transition,
)


def make_configs():
    action = {
        "action_horizon": 16,
        "action_interval_sec": 0.1,
        "replan_interval_sec": 0.6,
        "publish_rate_hz": 50.0,
        "minimum_command_lead_sec": 0.02,
        "compensation": {
            "camera_timestamp_sec": 0.0,
            "inference_latency_sec": 0.0,
            "ros_transport_latency_sec": 0.0,
            "zmq_transport_latency_sec": 0.0,
            "qp_latency_sec": 0.0,
            "gripper_latency_sec": 0.0,
        },
    }
    safety = {
        "workspace_min_xyz_m": [-1.0, -1.0, -1.0],
        "workspace_max_xyz_m": [1.0, 1.0, 1.0],
        "max_translation_step_m": 0.5,
        "max_rotation_step_rad": 2.0,
        "gripper_min_m": 0.0,
        "gripper_max_m": 0.1,
        "required_action_interval_tolerance_sec": 0.005,
    }
    return action, safety


def action_batch(times, xyz=None, yaw=None):
    count = len(times)
    actions = np.zeros((count, 7), dtype=np.float64)
    actions[:, 2] = 0.2
    actions[:, 6] = 0.04
    if xyz is not None:
        actions[:, :3] = np.asarray(xyz, dtype=np.float64)
    if yaw is not None:
        actions[:, 5] = np.asarray(yaw, dtype=np.float64)
    return actions


class ActionSchedulerTest(unittest.TestCase):
    def test_disabled_pose_checks_pass_model_geometry_and_bypass_both_limiters(self):
        action, safety = make_configs()
        action["rotation_step_limiter"] = {"enabled": True}
        action["translation_step_limiter"] = {"enabled": True}
        safety.update(translation_step_check_enabled=False,
                      rotation_step_check_enabled=False,
                      max_translation_step_m=None, max_rotation_step_rad=None)
        scheduler = ActionScheduler(action, safety)
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times, xyz=[[0, 0, .2], [.6, 0, .2], [.8, 0, .2]],
                               yaw=[0, .2, .5])
        report = scheduler.replace_plan(actions, times, now=10)
        self.assertEqual(report.rotation_limited_actions, 0)
        self.assertEqual(report.translation_limited_actions, 0)
        np.testing.assert_allclose([p.xyz for p in scheduler.plan], actions[:, :3])
        np.testing.assert_allclose([p.quat_xyzw for p in scheduler.plan],
                                   Rotation.from_rotvec(actions[:, 3:6]).as_quat())
        actions[1, 6] = .2
        with self.assertRaisesRegex(PlanRejected, "gripper range"):
            scheduler.replace_plan(actions, times, now=10)

    def rotation_limited_scheduler(self):
        action, safety = make_configs()
        action["rotation_step_limiter"] = {
            "enabled": True, "max_input_step_ratio": 1.5}
        safety["translation_step_check_enabled"] = False
        safety["max_rotation_step_rad"] = 0.080
        return ActionScheduler(action, safety)

    def test_small_rotation_overshoot_slows_pose_and_jaw_in_phase(self):
        scheduler = self.rotation_limited_scheduler()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.01, 0.02]
        actions[:, 5] = [0.0, 0.0838, 0.1676]
        actions[:, 6] = [0.096, 0.080, 0.064]
        report = scheduler.replace_plan(actions, times, now=10.0)
        self.assertEqual(report.rotation_limited_actions, 2)
        self.assertAlmostEqual(report.input_max_rotation_step_rad, 0.0838)
        self.assertLessEqual(report.output_max_rotation_step_rad, 0.080)
        alpha = 0.080 / 0.0838
        self.assertAlmostEqual(scheduler.plan[1].xyz[0], 0.01 * alpha)
        self.assertAlmostEqual(scheduler.plan[1].gripper, 0.096 - 0.016 * alpha)
        np.testing.assert_allclose([p.raw_timestamp for p in scheduler.plan], times)

    def test_rotation_limiter_keeps_nominal_actions_and_large_rejection(self):
        scheduler = self.rotation_limited_scheduler()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times, yaw=[0.0, 0.05, 0.10])
        report = scheduler.replace_plan(actions, times, now=10.0)
        self.assertEqual(report.rotation_limited_actions, 0)
        np.testing.assert_allclose(
            [p.quat_xyzw for p in scheduler.plan],
            Rotation.from_rotvec(actions[:, 3:6]).as_quat())
        actions[1, 5] = 0.13
        with self.assertRaisesRegex(PlanRejected, "rotation step"):
            scheduler.replace_plan(actions, times, now=10.0)
        self.assertFalse(scheduler.has_plan)

    def test_rotation_smoothing_cannot_hide_workspace_or_jaw_errors(self):
        for error in ("workspace", "jaw"):
            scheduler = self.rotation_limited_scheduler()
            times = np.asarray([10.1, 10.2])
            actions = action_batch(times, yaw=[0.0, 0.0838])
            if error == "workspace":
                actions[1, 0] = 2.0
            else:
                actions[1, 6] = 0.2
            with self.assertRaises(PlanRejected):
                scheduler.replace_plan(actions, times, now=10.0)

    def test_disabled_translation_check_preserves_model_waypoints(self):
        action_cfg, safety_cfg = make_configs()
        # Disabling the check also bypasses the optional path limiter.
        action_cfg["translation_step_limiter"] = {"enabled": True}
        safety_cfg["translation_step_check_enabled"] = False
        safety_cfg["max_translation_step_m"] = None
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.060, 0.140]
        report = scheduler.replace_plan(actions, times, now=10.0)
        self.assertEqual(report.accepted_actions, 3)
        self.assertEqual(report.translation_limited_actions, 0)
        self.assertFalse(scheduler.translation_limiter_enabled)
        np.testing.assert_array_equal(
            [p.xyz for p in scheduler.plan], actions[:, :3])
        np.testing.assert_array_equal(
            [p.raw_timestamp for p in scheduler.plan], times)

    def test_disabling_translation_check_keeps_other_rejections(self):
        for violation in ("rotation", "gripper", "workspace", "nan", "interval"):
            with self.subTest(violation=violation):
                action_cfg, safety_cfg = make_configs()
                safety_cfg["translation_step_check_enabled"] = False
                scheduler = ActionScheduler(action_cfg, safety_cfg)
                times = np.asarray([10.1, 10.2])
                actions = action_batch(times)
                actions[1, 0] = 0.060
                scheduler.replace_plan(actions, times, now=10.0)
                if violation == "rotation":
                    actions[1, 5] = 2.5
                elif violation == "gripper":
                    actions[1, 6] = 0.2
                elif violation == "workspace":
                    actions[:, 0] += 2.0
                elif violation == "nan":
                    actions[1, 0] = np.nan
                else:
                    times[1] += 0.1
                with self.assertRaises(PlanRejected):
                    scheduler.replace_plan(actions, times, now=10.0)
                self.assertFalse(scheduler.has_plan)

    def make_limited_scheduler(self):
        action_cfg, safety_cfg = make_configs()
        action_cfg["translation_step_limiter"] = {
            "enabled": True, "max_input_step_ratio": 1.5,
        }
        safety_cfg["max_translation_step_m"] = 0.035
        return ActionScheduler(action_cfg, safety_cfg)

    def test_modest_translation_overshoot_is_slowed_with_same_timestamps(self):
        scheduler = self.make_limited_scheduler()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.0455, 0.091]
        actions[:, 5] = [0.0, 0.05, 0.10]
        actions[:, 6] = [0.096, 0.080, 0.064]
        original = actions.copy()
        report = scheduler.replace_plan(actions, times, now=10.0)
        self.assertEqual(report.translation_limited_actions, 2)
        self.assertAlmostEqual(report.input_max_translation_step_m, 0.0455)
        self.assertLessEqual(report.output_max_translation_step_m, 0.035)
        np.testing.assert_allclose([p.xyz[0] for p in scheduler.plan],
                                   [0.0, 0.035, 0.070])
        np.testing.assert_allclose([p.raw_timestamp for p in scheduler.plan], times)
        # Pose rotation and gripper must slow with the same source-path phase.
        np.testing.assert_allclose(
            [p.gripper for p in scheduler.plan],
            [0.096, 0.096 - 0.016 * 0.035 / 0.0455,
             0.080 - 0.016 * (0.070 - 0.0455) / 0.0455])
        np.testing.assert_array_equal(actions, original)

    def test_nominal_plan_is_unchanged_with_limiter_enabled(self):
        scheduler = self.make_limited_scheduler()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.02, 0.04]
        report = scheduler.replace_plan(actions, times, now=10.0)
        self.assertEqual(report.translation_limited_actions, 0)
        np.testing.assert_array_equal([p.xyz for p in scheduler.plan], actions[:, :3])

    def test_large_translation_jump_still_clears_plan(self):
        scheduler = self.make_limited_scheduler()
        times = np.asarray([10.1, 10.2])
        scheduler.replace_plan(action_batch(times), times, now=10.0)
        actions = action_batch(times)
        actions[1, 0] = 0.060
        with self.assertRaisesRegex(PlanRejected, "translation step"):
            scheduler.replace_plan(actions, times, now=10.0)
        self.assertFalse(scheduler.has_plan)

    def test_limiter_does_not_hide_workspace_rotation_or_gripper_violations(self):
        for violation in ("workspace", "rotation", "gripper"):
            with self.subTest(violation=violation):
                scheduler = self.make_limited_scheduler()
                times = np.asarray([10.1, 10.2])
                actions = action_batch(times)
                actions[1, 0] = 0.040
                if violation == "workspace":
                    actions[:, 0] += 2.0
                elif violation == "rotation":
                    actions[1, 5] = 2.5
                else:
                    actions[1, 6] = 0.2
                with self.assertRaises(PlanRejected):
                    scheduler.replace_plan(actions, times, now=10.0)
                self.assertFalse(scheduler.has_plan)

    def test_curved_path_output_steps_remain_bounded(self):
        scheduler = self.make_limited_scheduler()
        times = 10.1 + np.arange(16) * 0.1
        angles = np.arange(15) * 0.7
        increments = 0.0455 * np.column_stack([
            np.cos(angles), np.sin(angles), np.zeros(15)])
        actions = action_batch(times)
        actions[:, :3] = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
        report = scheduler.replace_plan(actions, times, now=10.0)
        self.assertLessEqual(report.output_max_translation_step_m, 0.035)
        self.assertGreater(report.translation_limited_actions, 0)

    def test_find_grasp_transition(self):
        timestamps = np.arange(4, dtype=np.float64) * 0.1
        actions = action_batch(timestamps)
        actions[:, 6] = [0.096, 0.090, 0.074, 0.063]
        self.assertEqual(
            find_grasp_transition(actions, 0.096, 0.080), 2)

    @patch("env.ros_action_scheduler.time.time", return_value=10.0)
    def test_default_now_uses_ros_epoch_clock_domain(self, _mock_time):
        action_cfg, safety_cfg = make_configs()
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([9.9, 10.0, 10.1])
        report = scheduler.replace_plan(
            action_batch(timestamps), timestamps)
        self.assertEqual(report.dropped_expired, 2)
        self.assertEqual(report.accepted_actions, 1)
        self.assertEqual(scheduler.plan[0].raw_timestamp, 10.1)
        self.assertAlmostEqual(report.scheduler_now, 10.0)
        self.assertAlmostEqual(report.first_execution_timestamp, 10.1)
        self.assertAlmostEqual(report.last_execution_timestamp, 10.1)

    def test_expired_actions_are_dropped(self):
        action_cfg, safety_cfg = make_configs()
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([9.7, 9.8, 9.9, 10.0, 10.1])
        report = scheduler.replace_plan(
            action_batch(timestamps), timestamps, now=10.0)
        self.assertEqual(report.dropped_expired, 4)
        self.assertEqual(report.accepted_actions, 1)
        self.assertEqual(scheduler.plan[0].raw_timestamp, 10.1)

    def test_new_plan_replaces_only_future_actions(self):
        action_cfg, safety_cfg = make_configs()
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        old_times = np.asarray([10.1, 10.2])
        scheduler.replace_plan(
            action_batch(old_times), old_times, now=10.0)
        new_times = np.asarray([10.15, 10.25])
        report = scheduler.replace_plan(
            action_batch(new_times), new_times, now=10.05)
        self.assertEqual(report.replaced_future_actions, 2)
        self.assertEqual(
            [item.raw_timestamp for item in scheduler.plan],
            [10.15, 10.25])

    def test_control_action_limit_applies_after_expiration_filtering(self):
        action_cfg, safety_cfg = make_configs()
        action_cfg["max_actions_per_plan"] = 2
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([9.9, 10.0, 10.1, 10.2, 10.3])
        report = scheduler.replace_plan(
            action_batch(timestamps), timestamps, now=10.0)
        self.assertEqual(report.dropped_expired, 2)
        self.assertEqual(report.accepted_actions, 2)
        self.assertEqual(
            [item.raw_timestamp for item in scheduler.plan],
            [10.1, 10.2])

    def test_control_action_limit_ignores_unsafe_unsubmitted_actions(self):
        action_cfg, safety_cfg = make_configs()
        action_cfg["max_actions_per_plan"] = 2
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(timestamps)
        actions[2, 0] = 2.0
        report = scheduler.replace_plan(actions, timestamps, now=10.0)
        self.assertEqual(report.accepted_actions, 2)
        self.assertEqual(len(scheduler.plan), 2)

    def test_disabled_workspace_permits_outside_bounds_but_keeps_step_limit(self):
        action_cfg, safety_cfg = make_configs()
        safety_cfg.update({
            "workspace_enabled": False,
            "workspace_min_xyz_m": None,
            "workspace_max_xyz_m": None,
        })
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([10.1, 10.2])
        actions = action_batch(timestamps)
        actions[:, 0] = 2.0
        report = scheduler.replace_plan(actions, timestamps, now=10.0)
        self.assertEqual(report.accepted_actions, 2)
        unsafe_step = actions.copy()
        unsafe_step[1, 0] = 2.6
        with self.assertRaisesRegex(PlanRejected, "translation step"):
            scheduler.replace_plan(
                unsafe_step, timestamps, now=10.05)

    def test_scheduler_slerps_pose_between_waypoints(self):
        action_cfg, safety_cfg = make_configs()
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([10.0, 10.1])
        actions = action_batch(
            timestamps, yaw=[0.0, 1.0])
        scheduler.replace_plan(actions, timestamps, now=9.9)
        sampled = scheduler.sample(10.05)
        self.assertIsNotNone(sampled)
        self.assertAlmostEqual(
            sampled.pose_xyz_rotvec[5], 0.5, places=6)
        self.assertAlmostEqual(sampled.gripper, 0.04)

    def test_single_action_plan_holds_final_waypoint(self):
        action_cfg, safety_cfg = make_configs()
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([10.0])
        scheduler.replace_plan(
            action_batch(timestamps), timestamps, now=9.9)
        at_target = scheduler.sample(10.0)
        after_target = scheduler.sample(10.5)
        self.assertIsNotNone(at_target)
        self.assertIsNotNone(after_target)
        self.assertTrue(scheduler.has_plan)
        np.testing.assert_allclose(
            after_target.pose_xyz_rotvec,
            at_target.pose_xyz_rotvec,
        )
        self.assertAlmostEqual(
            after_target.gripper, at_target.gripper, places=9)

    def test_safety_violation_clears_plan(self):
        action_cfg, safety_cfg = make_configs()
        scheduler = ActionScheduler(action_cfg, safety_cfg)
        timestamps = np.asarray([10.0, 10.1])
        scheduler.replace_plan(
            action_batch(timestamps), timestamps, now=9.9)
        unsafe = action_batch(timestamps)
        unsafe[1, 0] = 2.0
        with self.assertRaisesRegex(PlanRejected, "translation step|workspace"):
            scheduler.replace_plan(unsafe, timestamps, now=9.95)
        self.assertFalse(scheduler.has_plan)

    def test_unverified_control_request_is_rejected(self):
        config = {
            "hardware_verification": {
                "operator_confirmed": False,
            },
            "safety": {
                "workspace_min_xyz_m": None,
                "workspace_max_xyz_m": None,
                "max_translation_step_m": None,
                "max_rotation_step_rad": None,
                "gripper_min_m": None,
                "gripper_max_m": None,
            },
            "control_output": {
                "mode": "direct_ros_pose",
                "pose_topic": "/right/delta_pose",
                "gripper_control_topic":
                    "/right/gripper/joint_states_ctrl",
                "teleop_active_topic": "/right/teleop_active",
                "expected_pose_frame": "right_base_link",
            },
        }
        report = validate_control_request(config)
        self.assertFalse(report.ready)
        self.assertTrue(any(
            "operator_confirmed" in issue for issue in report.issues))


class GraspReplanTest(unittest.TestCase):
    def make_wrapper(self):
        wrapper = RosActionScheduler.__new__(RosActionScheduler)
        wrapper._plan_lock = threading.RLock()
        action_cfg, safety_cfg = make_configs()
        action_cfg["translation_step_limiter"] = {
            "enabled": True, "max_input_step_ratio": 1.5,
        }
        safety_cfg["max_translation_step_m"] = 0.035
        wrapper.core = ActionScheduler(action_cfg, safety_cfg)
        wrapper._priming = False
        wrapper._last_published_action = None
        wrapper._rejected_plan_hold = None
        wrapper._grasp_sync_enabled = True
        wrapper._grasp_model_driven_pose = False
        wrapper._lift_guard = GraspLiftGuard({})
        wrapper._lift_guard_last_reason = None
        wrapper._tcp_feedback_received = None
        wrapper._tcp_feedback_stamp = None
        wrapper._gripper_extra_close = 0.0
        wrapper._gripper_last_tick = None
        wrapper._gripper_previous_model_width = None
        wrapper._gripper_model_widening = False
        wrapper._gripper_close_started = None
        wrapper._gripper_contact_reported = False
        wrapper._grasp_complete = False
        wrapper._grasp_awaiting_fresh_plan = False
        wrapper._grasp_contact_epoch = None
        wrapper._grasp_target_pose = None
        wrapper._grasp_open_width = None
        wrapper._grasp_close_width = None
        wrapper._grasp_settle_started = None
        wrapper._grasp_tighten_started = None
        wrapper._grasp_close_threshold = 0.080
        wrapper._grasp_pose_width = 0.070
        wrapper._grasp_pose_width_reached = False
        wrapper._grasp_max_extra_close = 0.003
        wrapper._grasp_tighten_floor = 0.060
        wrapper._grasp_tighten_rate = 0.005
        wrapper._grasp_target_width_tolerance = 0.001
        wrapper._grasp_always_tighten = False
        wrapper._grasp_require_fresh_plan = True
        wrapper._grasp_position_tolerance = 0.012
        wrapper._grasp_settle_sec = 0.6
        wrapper._grasp_close_rate = 0.02
        wrapper._grasp_close_timeout = 5.0
        wrapper._grasp_failed = False
        wrapper._grasp_hold_width = 0.096
        wrapper._grasp_reason = "not started"
        wrapper._grasp_verifier = GraspVerifier({
            "effort_calibrated": True, "contact_effort_min": 0.5,
        })
        wrapper._actual_gripper = 0.096
        wrapper._actual_tcp_xyz = np.asarray([0.0, 0.0, 0.2])
        wrapper._set_inactive = Mock()
        wrapper.node = Mock()
        return wrapper

    def test_rejected_plan_does_not_latch_grasp(self):
        wrapper = self.make_wrapper()
        times = np.asarray([10.1, 10.2])
        actions = action_batch(times)
        actions[:, 6] = [0.096, 0.070]
        actions[1, 0] = 0.060
        with self.assertRaises(PlanRejected):
            wrapper.replace_plan(actions, times, now=10.0)
        self.assertIsNone(wrapper._grasp_target_pose)

    def test_model_driven_pose_does_not_latch_or_freeze_on_closing(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_model_driven_pose = True
        times = np.asarray([1000.1, 1000.2, 1000.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.01, 0.02]
        actions[:, 2] = [0.20, 0.18, 0.16]
        actions[:, 6] = [0.096, 0.067, 0.063]
        wrapper.replace_plan(actions, times, now=1000.0)
        self.assertIsNone(wrapper._grasp_target_pose)
        self.tick(wrapper, 100.0, epoch_now=1000.3)
        pose = wrapper.pose_pub.publish.call_args.args[0].pose.position
        np.testing.assert_allclose([pose.x, pose.y, pose.z], actions[-1, :3])

        # Lift proceeds on the model's timestamps even without contact data.
        actions[:, 2] = [0.16, 0.18, 0.20]
        wrapper.replace_plan(actions, times + 0.6, now=1000.6)
        self.tick(wrapper, 100.6, epoch_now=1000.9)
        pose = wrapper.pose_pub.publish.call_args.args[0].pose.position
        np.testing.assert_allclose([pose.x, pose.y, pose.z], actions[-1, :3])
        self.assertFalse(wrapper._grasp_failed)
        self.assertIsNone(wrapper._grasp_target_pose)

    def test_lift_guard_blocks_model_upward_plan_until_grip_and_fresh_replan(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_model_driven_pose = True
        wrapper._lift_guard = GraspLiftGuard({"enabled": True})
        wrapper._actual_tcp_xyz = np.asarray([0, 0, .08])
        wrapper._grasp_verifier.check = Mock(return_value=(True, "stable contact"))
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 2] = .08
        actions[:, 6] = .063
        wrapper.replace_plan(actions, times, now=1000.0)
        self.tick(wrapper, 100)

        actions[:, 2] = .10  # Model now requests lifting before the jaw is ready.
        wrapper.replace_plan(actions, times, now=1000.0)
        for i in range(1, 14):
            now = 100 + i * .1
            wrapper._tcp_feedback_received = now
            wrapper._tcp_feedback_stamp = 1000.7
            self.tick(wrapper, now)
            self.assertAlmostEqual(wrapper.pose_pub.publish.call_args.args[0].pose.position.z, .08)
        self.assertIsNotNone(wrapper._lift_guard.confirmed_epoch)
        self.assertAlmostEqual(wrapper.gripper_pub.publish.call_args.args[0].position[0], .060)

        wrapper.replace_plan(actions, times + 1, now=1001,
                             observation_timestamp=1000.6)
        self.assertTrue(wrapper._lift_guard.holding)
        wrapper.replace_plan(actions, times + 1, now=1001,
                             observation_timestamp=1000.8)
        self.tick(wrapper, 101.4, epoch_now=1001.3)
        self.assertAlmostEqual(wrapper.pose_pub.publish.call_args.args[0].pose.position.z, .10)
        self.assertIsNone(wrapper._grasp_target_pose)  # No width-selected grasp.

    def test_jaw_compensation_is_bounded_and_model_reopening_releases(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_model_driven_pose = True
        self.assertAlmostEqual(wrapper._compensate_model_gripper(0.063, 100), 0.063)
        for i in range(1, 11):
            command = wrapper._compensate_model_gripper(0.063, 100 + i * 0.1)
        self.assertAlmostEqual(command, 0.060)
        self.assertAlmostEqual(wrapper._compensate_model_gripper(0.070, 101.1), 0.067)
        self.assertAlmostEqual(wrapper._compensate_model_gripper(0.096, 101.2), 0.096)
        # A new model-driven closing sequence starts with no retained offset.
        self.assertAlmostEqual(wrapper._compensate_model_gripper(0.063, 101.2), 0.063)
        self.assertAlmostEqual(wrapper._compensate_model_gripper(0.050, 101.3), 0.050)

    def test_visual_replan_updates_unconfirmed_guard_pose_and_allows_reopening(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_model_driven_pose = True
        guard = wrapper._lift_guard = GraspLiftGuard({"enabled": True})
        guard.observe_model(np.asarray([0, 0, .08, 0, 0, 0.]), .063, .080, 100)
        guard.observe_model(np.asarray([0, 0, .10, 0, 0, 0.]), .063, .080, 100.1)
        guard.hold_epoch = 1000
        times = np.asarray([1000.3, 1000.4, 1000.5])
        actions = action_batch(times)
        actions[:, :3] = [[.01, .01, .09], [.02, .01, .075], [.03, .01, .06]]
        actions[:, 6] = .063
        with patch("env.ros_action_scheduler.time.monotonic", return_value=100.2):
            wrapper.replace_plan(actions, times, now=1000.2,
                                 observation_timestamp=1000.1)
        self.tick(wrapper, 100.3)
        position = wrapper.pose_pub.publish.call_args.args[0].pose.position
        np.testing.assert_allclose([position.x, position.y, position.z], [.03, .01, .06])
        self.assertTrue(guard.holding)
        self.assertIsNone(guard.confirmed_epoch)

        # An opening visual correction must reach the actual publisher; old
        # base-width latching must not silently close the jaw again.
        actions[:, :3] = [.02, .02, .065]
        actions[:, 6] = .096
        with patch("env.ros_action_scheduler.time.monotonic", return_value=100.4):
            wrapper.replace_plan(actions, times + .3, now=1000.5,
                                 observation_timestamp=1000.4)
        self.tick(wrapper, 100.5, epoch_now=1000.8)
        self.assertAlmostEqual(wrapper.gripper_pub.publish.call_args.args[0].position[0], .096)
        self.assertTrue(guard.holding)

    def test_small_model_reopening_is_not_cancelled_by_extra_tightening(self):
        wrapper = self.make_wrapper()
        wrapper._compensate_model_gripper(0.063, 100.0)
        closed = wrapper._compensate_model_gripper(0.063, 100.1)
        offset = wrapper._gripper_extra_close
        reopened = wrapper._compensate_model_gripper(0.0632, 100.2)
        self.assertAlmostEqual(reopened - closed, 0.0002)
        self.assertAlmostEqual(wrapper._gripper_extra_close, offset)
        # A flat prediction after reopening must not silently tighten back.
        self.assertAlmostEqual(wrapper._compensate_model_gripper(0.0632, 100.3), reopened)

    def test_model_plan_rejection_keeps_qp_active_without_extra_jaw_closure(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_model_driven_pose = True
        times = np.asarray([1000.1, 1000.2, 1000.3])
        safe = action_batch(times)
        safe[:, 6] = 0.066
        safe[:, 0] = [0, 0.01, 0.02]
        wrapper.replace_plan(safe, times, now=1000.0)
        for i in range(11):
            self.tick(wrapper, 100 + i * 0.1)
        held_pose = wrapper._last_published_action.pose_xyz_rotvec.copy()
        held_width = wrapper._last_published_action.gripper
        self.assertAlmostEqual(held_width, 0.063)

        unsafe = safe.copy()
        unsafe[1, 5] = 3.0
        for i in range(3):
            with self.assertRaisesRegex(PlanRejected, "rotation step"):
                wrapper.replace_plan(unsafe, times, now=1000.0)
            self.assertFalse(wrapper.core.has_plan)
            self.tick(wrapper, 102 + i * 0.1)
            np.testing.assert_allclose(
                wrapper._last_published_action.pose_xyz_rotvec, held_pose)
            self.assertAlmostEqual(wrapper._last_published_action.gripper, held_width)
            self.assertTrue(wrapper.active_pub.publish.call_args.args[0].data)
        wrapper._set_inactive.assert_not_called()

        # A valid replacement resumes normal model execution without a QP
        # false/true transition or a controller reinitialization.
        safe[:, 2] = 0.25
        wrapper.replace_plan(safe, times + 0.6, now=1000.6)
        self.assertIsNone(wrapper._rejected_plan_hold)
        self.tick(wrapper, 103, epoch_now=1000.9)
        self.assertAlmostEqual(wrapper._last_published_action.pose_xyz_rotvec[2], 0.25)

    def test_rejected_replan_preserves_active_grasp_hold(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_target_pose = np.asarray(
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        times = np.asarray([10.1, 10.2])
        safe = action_batch(times)
        safe[:, 6] = 0.063
        wrapper.core.replace_plan(safe, times, now=10.0)
        previous_plan = list(wrapper.core.plan)

        unsafe = safe.copy()
        unsafe[1, 3:6] = [0.0, 0.0, 3.0]
        with self.assertRaisesRegex(PlanRejected, "rotation step"):
            wrapper.replace_plan(unsafe, times, now=10.0)
        self.assertEqual(wrapper.core.plan, previous_plan)
        wrapper._set_inactive.assert_not_called()

    def test_bounded_tightening_gets_independent_timeout_window(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_always_tighten = True
        wrapper._grasp_target_pose = np.asarray(
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        wrapper._grasp_hold_width = 0.096
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)
        for t in np.arange(106.0, 106.36, 0.05):
            wrapper._grasp_verifier.update(t, t + 894.0, 0.063, 0.1, True)

        # Initial phase is already older than five seconds. Starting the 60 mm
        # correction now must not fail in the same timer callback.
        self.tick(wrapper, 106.35, epoch_now=1000.35)
        self.assertEqual(wrapper._grasp_tighten_started, 106.35)
        self.assertFalse(wrapper._grasp_failed)

        # The correction receives its own full timeout window.
        self.tick(wrapper, 111.36, epoch_now=1005.36)
        self.assertTrue(wrapper._grasp_failed)

    def test_expired_grasp_target_is_not_latched(self):
        wrapper = self.make_wrapper()
        times = np.asarray([9.9, 10.0, 10.1, 10.2])
        actions = action_batch(times)
        actions[:, 6] = [0.096, 0.070, 0.096, 0.096]
        wrapper.replace_plan(actions, times, now=10.0)
        self.assertIsNone(wrapper._grasp_target_pose)

    def test_latched_grasp_uses_slowed_future_pose(self):
        wrapper = self.make_wrapper()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.0455, 0.091]
        actions[:, 6] = [0.096, 0.070, 0.065]
        wrapper.replace_plan(actions, times, now=10.0)
        # The limiter produces 35 mm and 70 mm future positions. The 70 mm jaw
        # trigger is interpolated between their associated closing widths.
        self.assertGreater(wrapper._grasp_target_pose[0], 0.035)
        self.assertLess(wrapper._grasp_target_pose[0], 0.070)

    def test_first_crossing_is_not_used_as_final_close_width(self):
        wrapper = self.make_wrapper()
        times = 10.1 + np.arange(6) * 0.1
        actions = action_batch(times)
        actions[:, 0] = np.arange(6) * 0.005
        actions[:, 6] = [0.096, 0.079, 0.070, 0.063, 0.090, 0.040]
        wrapper.replace_plan(actions, times, now=10.0)
        self.assertAlmostEqual(wrapper._grasp_close_width, 0.063)
        self.assertAlmostEqual(wrapper._grasp_target_pose[0], 0.010)
        self.assertTrue(wrapper._grasp_pose_width_reached)

    def test_later_replan_updates_incomplete_close_pose_and_width_together(self):
        wrapper = self.make_wrapper()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.005, 0.010]
        actions[:, 6] = [0.096, 0.079, 0.075]
        wrapper.replace_plan(actions, times, now=10.0)
        self.assertAlmostEqual(wrapper._grasp_target_pose[0], 0.010)
        self.assertAlmostEqual(wrapper._grasp_close_width, 0.075)
        self.assertFalse(wrapper._grasp_pose_width_reached)

        actions[:, 0] = [0.010, 0.015, 0.020]
        actions[:, 6] = [0.074, 0.070, 0.063]
        wrapper.replace_plan(actions, times + 0.3, now=10.3)
        self.assertAlmostEqual(wrapper._grasp_target_pose[0], 0.015)
        self.assertAlmostEqual(wrapper._grasp_close_width, 0.063)
        self.assertTrue(wrapper._grasp_pose_width_reached)

    def test_model_close_width_survives_replan_after_closing_starts(self):
        wrapper = self.make_wrapper()
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 6] = [0.096, 0.079, 0.063]
        wrapper.replace_plan(actions, times, now=10.0)
        self.assertEqual(wrapper._grasp_close_width, 0.063)
        wrapper._grasp_settle_started = 100.0
        actions[:, 6] = [0.079, 0.055, 0.050]
        wrapper.replace_plan(actions, times, now=10.0)
        self.assertEqual(wrapper._grasp_close_width, 0.063)

    def test_verified_width_holds_during_transport_but_allows_release(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_complete = True
        wrapper._grasp_hold_width = 0.060
        times = np.asarray([1000.1, 1000.2])
        for predicted, expected in (
            (0.070, 0.060), (0.055, 0.055), (0.096, 0.096)
        ):
            actions = action_batch(times)
            actions[:, 6] = predicted
            wrapper.core.replace_plan(actions, times, now=1000.0)
            self.tick(wrapper, 100.7)
            self.assertEqual(
                wrapper.gripper_pub.publish.call_args.args[0].position, [expected])

    def test_post_grasp_waits_for_observation_newer_than_contact(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_complete = True
        wrapper._grasp_awaiting_fresh_plan = True
        wrapper._grasp_contact_epoch = 1000.0
        wrapper._grasp_hold_width = 0.060
        wrapper._grasp_target_pose = np.asarray(
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=float)
        times = np.asarray([1000.6, 1000.7])
        actions = action_batch(times)
        actions[:, 0] = 0.1
        actions[:, 2] = 0.3
        actions[:, 6] = 0.070

        wrapper.replace_plan(
            actions,
            times,
            now=1000.5,
            observation_timestamp=999.9,
        )
        self.assertTrue(wrapper._grasp_awaiting_fresh_plan)
        self.tick(wrapper, 100.0)
        pose = wrapper.pose_pub.publish.call_args.args[0].pose.position
        np.testing.assert_allclose([pose.x, pose.y, pose.z], [0.0, 0.0, 0.2])
        self.assertEqual(
            wrapper.gripper_pub.publish.call_args.args[0].position, [0.060])

        wrapper.replace_plan(
            actions,
            times,
            now=1000.5,
            observation_timestamp=1000.1,
        )
        self.assertFalse(wrapper._grasp_awaiting_fresh_plan)
        self.tick(wrapper, 100.1)
        pose = wrapper.pose_pub.publish.call_args.args[0].pose.position
        np.testing.assert_allclose([pose.x, pose.y, pose.z], [0.1, 0.0, 0.3])

    def test_stable_no_contact_starts_bounded_tightening(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_target_pose = np.asarray([0, 0, 0.2, 0, 0, 0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        wrapper._grasp_hold_width = 0.096
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)
        for t in np.arange(101.35, 101.71, 0.05):
            wrapper._grasp_verifier.update(t, t + 899, 0.063, 0.1, True)
        self.tick(wrapper, 101.7)
        self.assertEqual(wrapper._grasp_tighten_started, 101.7)
        self.assertEqual(
            wrapper.gripper_pub.publish.call_args.args[0].position, [0.063])
        self.tick(wrapper, 102.4)
        self.assertEqual(
            wrapper.gripper_pub.publish.call_args.args[0].position, [0.060])

    def test_stale_feedback_cannot_start_tightening(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_target_pose = np.asarray([0, 0, 0.2, 0, 0, 0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        wrapper._grasp_hold_width = 0.096
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)
        self.tick(wrapper, 100.7)
        self.assertIsNone(wrapper._grasp_tighten_started)
        np.testing.assert_allclose(
            wrapper.gripper_pub.publish.call_args.args[0].position, [0.082])

    def test_contact_after_tightening_latches_command_not_measured_width(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_target_pose = np.asarray([0, 0, 0.2, 0, 0, 0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        wrapper._grasp_tighten_started = 100.0
        wrapper._grasp_hold_width = 0.096
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)
        for t in np.arange(100.35, 100.71, 0.05):
            wrapper._grasp_verifier.update(t, t + 900, 0.0612, 0.8, True)
        self.tick(wrapper, 100.7)
        self.assertTrue(wrapper._grasp_complete)
        self.assertEqual(wrapper._grasp_hold_width, 0.060)

    def test_current_object_requires_full_bounded_tightening_before_contact(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_always_tighten = True
        wrapper._grasp_target_pose = np.asarray([0, 0, 0.2, 0, 0, 0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        wrapper._grasp_hold_width = 0.096
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)

        # Strong evidence at the model width must not release the lift gate.
        for t in np.arange(101.35, 101.71, 0.05):
            wrapper._grasp_verifier.update(t, t + 899, 0.063, 0.8, True)
        self.tick(wrapper, 101.7)
        self.assertFalse(wrapper._grasp_complete)
        self.assertEqual(wrapper._grasp_tighten_started, 101.7)

        # Only stable evidence after the command reaches 60 mm may confirm.
        for t in np.arange(102.35, 102.71, 0.05):
            wrapper._grasp_verifier.update(t, t + 899, 0.0612, 0.8, True)
        self.tick(wrapper, 102.7, epoch_now=1001.7)
        self.assertTrue(wrapper._grasp_complete)
        self.assertEqual(wrapper._grasp_hold_width, 0.060)

    def test_replans_during_grasp_settle_do_not_clear_or_release_control(self):
        self.assert_replans_finish_settling(0.070)

    def test_open_gripper_replan_cannot_interrupt_started_settle(self):
        self.assert_replans_finish_settling(0.096)

    def assert_replans_finish_settling(self, predicted_width):
        wrapper = self.make_wrapper()
        target = np.asarray([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        wrapper._grasp_target_pose = target.copy()
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.070
        wrapper._grasp_settle_started = 100.0
        times = np.asarray([10.1, 10.2, 10.3])
        actions = action_batch(times)
        actions[:, 0] = [0.0, 0.0455, 0.091]
        actions[:, 6] = predicted_width
        for shift in (0.0, 0.6, 1.2):
            report = wrapper.replace_plan(actions, times + shift, now=10.0 + shift)
            self.assertTrue(wrapper.core.has_plan)
            self.assertGreater(report.translation_limited_actions, 0)
            self.assertEqual(wrapper._grasp_settle_started, 100.0)
            np.testing.assert_array_equal(wrapper._grasp_target_pose, target)
        wrapper._set_inactive.assert_not_called()

        for t in np.arange(100.35, 100.71, 0.05):
            wrapper._grasp_verifier.update(t, t + 900, 0.070, 0.8, True)
        self.tick(wrapper, 100.7)
        self.assertTrue(wrapper._grasp_complete)
        wrapper._set_inactive.assert_not_called()
        self.assertTrue(wrapper.active_pub.publish.call_args.args[0].data)

    def test_elapsed_wait_alone_does_not_authorize_lift(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_target_pose = np.asarray([0, 0, 0.2, 0, 0, 0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)
        self.tick(wrapper, 100.7)
        self.assertFalse(wrapper._grasp_complete)
        np.testing.assert_allclose(
            wrapper.gripper_pub.publish.call_args.args[0].position, [0.082])

    def test_unconfirmed_timeout_latches_hold_across_replans(self):
        wrapper = self.make_wrapper()
        wrapper._grasp_target_pose = np.asarray([0, 0, 0.2, 0, 0, 0], dtype=float)
        wrapper._grasp_open_width = 0.096
        wrapper._grasp_close_width = 0.063
        wrapper._grasp_settle_started = 100.0
        wrapper._actual_gripper = 0.065
        times = np.asarray([1000.1, 1000.2])
        actions = action_batch(times)
        actions[:, 6] = 0.063
        wrapper.core.replace_plan(actions, times, now=1000.0)
        self.tick(wrapper, 105.1)
        self.assertTrue(wrapper._grasp_failed)
        self.assertFalse(wrapper._grasp_complete)
        self.assertEqual(wrapper.gripper_pub.publish.call_args.args[0].position, [0.065])
        actions[:, 6] = 0.096
        wrapper.replace_plan(actions, times, now=1000.0)
        self.tick(wrapper, 106.0)
        self.assertFalse(wrapper._grasp_complete)
        self.assertEqual(wrapper.gripper_pub.publish.call_args.args[0].position, [0.065])

    def tick(self, wrapper, monotonic_now, epoch_now=1000.7):
        # Exercise the timer with fake ROS messages; no ROS/hardware is used.
        def pose_message():
            return SimpleNamespace(
                header=SimpleNamespace(),
                pose=SimpleNamespace(position=SimpleNamespace(),
                                     orientation=SimpleNamespace()))

        wrapper._clock_now_sec = lambda: epoch_now
        wrapper.node.get_clock().now().to_msg.return_value = None
        wrapper.pose_pub = Mock()
        wrapper.gripper_pub = Mock()
        wrapper.active_pub = Mock()
        wrapper.pose_frame = "right_base_link"
        wrapper.gripper_joint_name = "right_gripper_joint"
        wrapper.was_active = True
        messages = {
            "geometry_msgs.msg": SimpleNamespace(PoseStamped=pose_message),
            "sensor_msgs.msg": SimpleNamespace(
                JointState=lambda: SimpleNamespace(header=SimpleNamespace())),
            "std_msgs.msg": SimpleNamespace(Bool=lambda **kw: SimpleNamespace(**kw)),
        }
        with patch.dict(sys.modules, messages), patch(
            "env.ros_action_scheduler.time.monotonic", return_value=monotonic_now,
        ):
            wrapper._timer_callback()


if __name__ == "__main__":
    unittest.main()
