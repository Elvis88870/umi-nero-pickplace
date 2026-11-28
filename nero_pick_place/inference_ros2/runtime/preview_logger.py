"""Optional Rerun logging for policy preview."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class PreviewLogger:
    def __init__(self, save_path: Optional[Path], disabled: bool):
        self.disabled = disabled
        self.rr = None
        if disabled:
            return
        import rerun as rr

        self.rr = rr
        rr.init("nero_ros_umi_preview", spawn=save_path is None)
        if save_path is not None:
            save_path = save_path.expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(save_path))

    def log(
        self,
        index: int,
        obs: dict,
        actions: np.ndarray,
        action_timestamps: np.ndarray,
        inference_ms: float,
    ) -> None:
        if self.disabled:
            return
        rr = self.rr
        rr.set_time_sequence("inference", index)
        rr.log("camera/camera0_fisheye", rr.Image(obs["camera0_rgb"][-1]))
        if "camera1_rgb" in obs:
            rr.log("camera/camera1_d405", rr.Image(obs["camera1_rgb"][-1]))
        rr.log("state/tcp_position", rr.Points3D(
            obs["robot0_eef_pos"][-1:], colors=[[255, 255, 255]]))
        rr.log("action/predicted_path", rr.LineStrips3D(
            [actions[:, :3]], colors=[[240, 80, 60]]))
        rr.log("action/predicted_steps", rr.Points3D(
            actions[:, :3],
            colors=np.tile([[240, 80, 60]], (actions.shape[0], 1))))
        rr.log(
            "action/predicted_gripper",
            rr.TextDocument(str([float(v) for v in actions[:, 6]])))
        rr.log("metrics/inference_ms", rr.Scalar(float(inference_ms)))
        rr.log(
            "meta/timestamp_contract",
            rr.TextLog(
                "Execution timestamps are observation-relative."),
            static=True)
        rr.log(
            "action/timestamps_from_observation",
            rr.TextLog(str(np.asarray(action_timestamps).tolist())))
