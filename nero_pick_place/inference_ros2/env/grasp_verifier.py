"""Read-only contact evidence from gripper telemetry; no ROS dependencies."""

from collections import deque
import math

import numpy as np


def closing_target_index(actions, start, reopen_threshold):
    """Find the tightest predicted opening before the first reopening.

    80 mm is only the transition trigger, not a holding-width setpoint.
    """
    end = start + 1
    while end < len(actions) and actions[end, 6] < reopen_threshold:
        end += 1
    return start + int(np.argmin(actions[start:end, 6]))


class GraspVerifier:
    def __init__(self, config):
        self.threshold = config.get("contact_effort_min")
        self.calibrated = config.get("effort_calibrated", False) is True
        self.effort_required = config.get("effort_required", True) is True
        self.width_error_min = float(
            config.get("contact_width_error_min_m", 0.0))
        self.stable_sec = float(config.get("stable_sec", 0.3))
        self.width_tolerance = float(config.get("stable_width_tolerance_m", 0.001))
        self.min_width = float(config.get("empty_closed_width_m", 0.0)) + float(
            config.get("empty_width_margin_m", 0.003))
        self.max_age = float(config.get("max_feedback_age_sec", 0.25))
        self.max_gap = float(config.get("max_feedback_gap_sec", 0.1))
        if min(self.stable_sec, self.width_tolerance, self.max_age, self.max_gap) <= 0:
            raise ValueError("grasp verification timing/tolerance must be positive")
        if self.width_error_min < 0:
            raise ValueError("contact width error threshold must be non-negative")
        if self.effort_required and self.calibrated and (
            self.threshold is None or not math.isfinite(float(self.threshold))
            or float(self.threshold) <= 0
        ):
            raise ValueError("calibrated contact_effort_min must be positive")
        self.samples = deque(maxlen=1000)
        self.last_stamp = None

    def update(self, now, stamp, width, effort, valid):
        # Duplicate/backwards source timestamps cannot establish persistence.
        if not all(math.isfinite(x) for x in (now, stamp, width, effort)):
            self.samples.clear()
            return
        if stamp <= 0 or (self.last_stamp is not None and stamp <= self.last_stamp):
            return
        self.last_stamp = stamp
        if not valid:
            self.samples.clear()
            return
        self.samples.append((now, stamp, width, abs(effort)))
        while len(self.samples) > 1 and self.samples[1][0] < now - self.stable_sec:
            self.samples.popleft()

    def _stable_samples(self, now, epoch_now, closing_started, open_width):
        if self.effort_required and (
            not self.calibrated or self.threshold is None
        ):
            return None, "contact threshold uncalibrated"
        if not self.samples:
            return None, "no valid motor telemetry"
        last = self.samples[-1]
        if (now - last[0] > self.max_age or epoch_now - last[1] > self.max_age
                or last[1] > epoch_now + 0.05):
            return None, "stale motor telemetry"
        samples = [s for s in self.samples if s[0] >= closing_started]
        if len(samples) < 2 or samples[-1][0] - samples[0][0] < self.stable_sec:
            return None, "waiting for stable contact"
        if any(b[0] - a[0] > self.max_gap or b[1] - a[1] > self.max_gap
               for a, b in zip(samples, samples[1:])):
            return None, "motor telemetry gap"
        widths = [s[2] for s in samples]
        if min(widths) <= self.min_width:
            return None, "empty/full-close opening"
        if max(widths) - min(widths) > self.width_tolerance:
            return None, "gripper still moving"
        if last[2] > open_width - 0.002:
            return None, "no measurable closure from open width"
        return samples, "stable gripper telemetry"

    def check(
        self,
        now,
        epoch_now,
        closing_started,
        open_width,
        command_width=None,
    ):
        samples, reason = self._stable_samples(
            now, epoch_now, closing_started, open_width)
        if samples is None:
            return False, reason
        if command_width is not None and self.width_error_min > 0:
            if not math.isfinite(float(command_width)):
                return False, "invalid gripper command width"
            width_error = samples[-1][2] - float(command_width)
            if width_error < self.width_error_min:
                return False, "jaw obstruction below contact threshold"
        if self.effort_required and (
            min(s[3] for s in samples) < float(self.threshold)
        ):
            return False, "contact current below threshold"
        if self.effort_required:
            return True, "stable width and sustained calibrated motor current"
        return True, "stable calibrated jaw obstruction"

    def correction_ready(
        self,
        now,
        epoch_now,
        closing_started,
        open_width,
        model_width,
        target_tolerance,
        command_width=None,
        ignore_contact=False,
    ):
        """Allow bounded tightening only after a stable, no-contact model stop."""
        samples, reason = self._stable_samples(
            now, epoch_now, closing_started, open_width)
        if samples is None:
            return False, reason
        if samples[-1][2] > model_width + target_tolerance:
            return False, "model gripper target not reached"
        if (
            not ignore_contact
            and command_width is not None
            and self.width_error_min > 0
            and samples[-1][2] - float(command_width) >= self.width_error_min
        ):
            return False, "jaw obstruction already detected"
        if not ignore_contact and self.effort_required and (
            min(s[3] for s in samples) >= float(self.threshold)
        ):
            return False, "contact already detected"
        return True, "stable model target without contact"
