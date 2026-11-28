"""Pause a model-requested lift until the model grasp point and jaw are ready."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


class GraspLiftGuard:
    def __init__(self, config):
        self.enabled = bool(config.get("enabled", False))
        self.rise = float(config.get("rise_detection_m", 0.002))
        self.position_tolerance = float(config.get("position_tolerance_m", 0.004))
        self.timeout = float(config.get("timeout_sec", 5.0))
        self.max_age = float(config.get("max_tcp_age_sec", 0.25))
        self.max_upward_correction = float(config.get("max_upward_correction_m", 0.010))
        if not all(math.isfinite(v) and v > 0 for v in (
                self.rise, self.position_tolerance, self.timeout, self.max_age,
                self.max_upward_correction)):
            raise ValueError("lift guard tolerances and timing must be positive")
        self.reset()

    def reset(self):
        self.low_pose = None
        self.base_width = None
        self.holding = False
        self.failed = False
        self.released = False
        self.confirmed_epoch = None
        self.started = None
        self.command_since = None
        self.last_command = None
        self.last_observation = None
        self.correction_ceiling = None
        self.reopening = False
        self.target_stable_since = None
        self.frozen_command = None
        self.hold_epoch = None
        self.reason = "following model approach"

    def observe_model(self, pose, width, open_threshold, now):
        if self.failed or self.holding:
            return
        if width >= open_threshold:
            self.reset()
            return
        if self.released:
            return
        self.base_width = (width if self.base_width is None
                           else min(self.base_width, width))
        if self.low_pose is None or pose[2] < self.low_pose[2]:
            self.low_pose = pose.copy()
        if pose[2] > self.low_pose[2] + self.rise:
            self.holding = True
            self.started = now
            self.target_stable_since = now
            self.correction_ceiling = self.low_pose[2] + self.max_upward_correction
            self.reason = "model requested lift; waiting for jaw and TCP"

    def update_approach(self, actions, observation_stamp, open_threshold, now):
        """Retarget an unconfirmed grasp using a fresh visual approach forecast.

        Use the first closing segment's approach valley only if that segment
        actually descends. A lift-only forecast cannot move the held target
        into transport. An opening/reapproach forecast keeps jaw corrections.
        The upward correction ceiling is anchored once, never ratcheted up.
        """
        if (not self.holding or self.failed or self.confirmed_epoch is not None
                or observation_stamp is None):
            return False
        stamp = float(observation_stamp)
        if (not math.isfinite(stamp)
                or (self.hold_epoch is not None and stamp <= self.hold_epoch)
                or (self.last_observation is not None and stamp <= self.last_observation)):
            return False
        self.last_observation = stamp
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
            return False
        if not np.all(np.isfinite(actions)):
            return False
        immediate_width = float(actions[0, 6])
        self.reopening = (
            immediate_width >= open_threshold
            or immediate_width > self.base_width + 0.0002
            or (self.reopening and immediate_width >= self.base_width - 0.0002))
        closed = np.flatnonzero(actions[:, 6] < open_threshold)
        candidate = None
        width = self.base_width
        if len(closed):
            start = int(closed[0])
            end = start + 1
            while end < len(actions) and actions[end, 6] < open_threshold:
                end += 1
            valley = start
            for index in range(start + 1, end):
                if actions[index, 2] < actions[valley, 2]:
                    valley = index
                elif actions[index, 2] > actions[valley, 2] + self.rise:
                    break
            # Descent may precede jaw closure (open approach -> close valley).
            peak = float(np.max(actions[:valley + 1, 2]))
            if peak - float(actions[valley, 2]) >= self.rise:
                candidate = actions[valley, :6].copy()
                width = float(np.min(actions[start:valley + 1, 6]))
        if self.reopening:
            # While the model reopens, preserve the immediate correction pose;
            # do not jump ahead to a future closing pose or reclose early.
            candidate = actions[0, :6].copy()
            width = float(actions[0, 6])
        if candidate is None:
            return False
        candidate[2] = min(candidate[2], self.correction_ceiling)
        changed = (
            np.linalg.norm(candidate[:3] - self.low_pose[:3]) > 0.001
            or (Rotation.from_rotvec(candidate[3:6]).inv()
                * Rotation.from_rotvec(self.low_pose[3:6])).magnitude() > 0.01
            or abs(width - self.base_width) > 0.0002)
        self.low_pose = candidate
        self.base_width = width
        if changed:
            self.target_stable_since = now
            self.command_since = None
            self.last_command = None
            self.reason = "fresh visual approach correction accepted"
        return changed

    def verify(self, now, epoch_now, actual_xyz, feedback_fresh, command,
               correction_done, contact_check, stable_sec):
        if not self.holding or self.failed or self.confirmed_epoch is not None:
            return
        if self.last_command is None or abs(command - self.last_command) > 1e-6:
            self.command_since = now
        self.last_command = command
        at_pose = (feedback_fresh and actual_xyz is not None and
                   np.linalg.norm(actual_xyz - self.low_pose[:3])
                   <= self.position_tolerance)
        # Require a fresh stable contact window after reaching the final jaw
        # command, rather than accepting an earlier transient obstruction.
        since = max(self.command_since, self.target_stable_since)
        contact, reason = contact_check(since)
        if self.reopening:
            reason = "model reopening for visual correction"
        elif not at_pose:
            reason = "waiting for actual TCP at model grasp point"
        elif not correction_done:
            reason = "waiting for bounded jaw tightening"
        elif now - self.command_since < stable_sec:
            reason = "waiting for final jaw command to settle"
        elif contact:
            self.confirmed_epoch = epoch_now
            self.frozen_command = command
            reason = "grip confirmed; waiting for post-contact observation"
        self.reason = reason
        if self.confirmed_epoch is None and now - self.started >= self.timeout:
            self.failed = True
            self.reason = "grip not confirmed before lift timeout; holding until restart"

    def accept_observation(self, stamp):
        if (self.holding and not self.failed and self.confirmed_epoch is not None
                and stamp is not None and math.isfinite(float(stamp))
                and float(stamp) > self.confirmed_epoch):
            self.holding = False
            self.released = True
            self.reason = "fresh post-grasp plan accepted; model lift resumed"
            return True
        return False
