#!/usr/bin/env python3
"""Convert offline evaluation NPZ files to Rerun 0.22.1 recordings.

Run in a separate environment with numpy and rerun-sdk==0.22.1 installed.
No checkpoint, torch, Zarr or GPU is needed by this exporter.
"""

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageDraw


def gripper_chart(offsets, target_mm, predicted_mm, time_max, width_max):
    """Render both curves on one fixed, labelled pair of physical axes."""
    image = Image.new("RGB", (720, 360), (25, 28, 32))
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 65, 695, 35, 310

    def point(t, width):
        return (left + float(t) / time_max * (right-left),
                bottom - float(width) / width_max * (bottom-top))

    for value in np.linspace(0, width_max, 6):
        y = point(0, value)[1]
        draw.line([(left, y), (right, y)], fill=(65, 65, 65))
        draw.text((10, y-5), f"{value:.0f}", fill="white")
    for value in np.linspace(0, time_max, 5):
        x = point(value, 0)[0]
        draw.line([(x, top), (x, bottom)], fill=(65, 65, 65))
        draw.text((x-10, bottom+8), f"{value:.2f}", fill="white")
    draw.text((left, 10), "Gripper width (mm)", fill="white")
    draw.text((480, 10), "Target", fill=(40, 210, 70))
    draw.text((570, 10), "Prediction", fill=(230, 60, 60))
    draw.text((270, 340), "Future time (seconds)", fill="white")
    for values, color in ((target_mm, (40, 210, 70)), (predicted_mm, (230, 60, 60))):
        draw.line([point(t, width) for t, width in zip(offsets, values)], fill=color, width=3)
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.evaluation_dir.glob("episode_*.npz"))
    if not files:
        parser.error("No episode_*.npz files found; run eval_checkpoint.py first")
    # One range for all episodes, expanded if a value exceeds nominal limits.
    time_max, width_max = 1.6, 110.0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            time_max = max(time_max, float(data["future_time_offsets"].max()))
            width_max = max(width_max, float(data["prediction"][..., 9].max())*1000,
                            float(data["ground_truth"][..., 9].max())*1000)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            episode = int(data["episode"])
            recording = rr.new_recording(f"pika_offline_episode_{episode:03d}")
            # Attach the sink BEFORE logging to preserve every frame.
            rgb_keys = [key for key in data.files if key.startswith("camera") and key.endswith("_rgb")]
            camera_views = [rrb.Spatial2DView(origin=f"camera/{key}", name=key) for key in rgb_keys]
            if not camera_views:
                camera_views = [rrb.Spatial2DView(origin="camera", name="RGB observation")]
            blueprint = rrb.Blueprint(rrb.Vertical(
                rrb.Horizontal(
                    *camera_views,
                    rrb.Spatial3DView(origin="world", name="TCP: green target / red prediction"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="gripper_horizon_mm", name="Gripper horizon (seconds, mm)"),
                    rrb.TimeSeriesView(origin="errors/position_mm", name="Position error (mm)"),
                    rrb.TimeSeriesView(origin="errors/rotation_deg", name="Rotation error (degrees)"),
                    rrb.TimeSeriesView(origin="errors/gripper_mm", name="Gripper error (mm)"),
                ), row_shares=[3, 1],
            ))
            recording.save(str(path.with_suffix(".rrd")), default_blueprint=blueprint)
            rr.log("legend", rr.TextDocument(
                "Green: demonstration. Red: checkpoint prediction. "
                "TCP paths are in this episode's world frame, in metres. "
                "Errors are computed over the 16-step prediction horizon. "
                "This is offline teacher-forced evaluation, not task success."
            ), static=True, recording=recording)
            for sample in range(len(data["timestamp"])):
                recording.set_time_sequence("sample", sample)
                recording.set_time_seconds("episode_time", float(data["timestamp"][sample]))
                if rgb_keys:
                    for key in rgb_keys:
                        rr.log(f"camera/{key}/rgb", rr.Image(data[key][sample]), recording=recording)
                else:
                    rr.log("camera/rgb", rr.Image(data["images"][sample]), recording=recording)
                predicted = data["predicted_world"][sample]
                target = data["ground_truth_world"][sample]
                for name, poses, color in (("ground_truth", target, [40, 210, 70]),
                                          ("prediction", predicted, [230, 60, 60])):
                    rr.log(f"world/{name}/path", rr.LineStrips3D(
                        [poses[:, :3, 3]], colors=[color]), recording=recording)
                    # Local RGB axes at the first and final predicted TCP poses.
                    for label, pose in (("first", poses[0]), ("last", poses[-1])):
                        rr.log(f"world/{name}/{label}_axes", rr.Arrows3D(
                            origins=np.repeat(pose[None, :3, 3], 3, axis=0),
                            vectors=pose[:3, :3].T * 0.03,
                            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        ), recording=recording)
                for metric in ("position_mm", "rotation_deg", "gripper_mm"):
                    rr.log(f"errors/{metric}", rr.Scalar(float(data[metric][sample].mean())), recording=recording)
                rr.log("timing/inference_ms", rr.Scalar(float(data["inference_ms"][sample])), recording=recording)
                # Horizon is a separate explicit curve rather than future frames
                # on the observation timeline (which would mix replanning windows).
                offsets = data["future_time_offsets"][sample]
                pred_width = data["prediction"][sample, :, 9] * 1000
                gt_width = data["ground_truth"][sample, :, 9] * 1000
                rr.log("gripper_horizon_mm/chart", rr.Image(gripper_chart(
                    offsets, gt_width, pred_width, time_max, width_max)), recording=recording)
            recording.flush()
            print(path.with_suffix(".rrd"))


if __name__ == "__main__":
    main()
