# NERO ROS2 UMI inference

Thor-side inference for the NERO continuous-IK ROS2 chain. This package only
loads a trained UMI checkpoint, builds policy observations from ROS2, and sends
safe action plans to the existing downstream controller. It does not modify
training code, checkpoints, datasets, upstream UMI, or NERO driver packages.

## Production structure

```text
inference_ros2/
├── eval_nero_ros.py
├── runtime/
│   ├── policy_runtime.py
│   ├── control_loop.py
│   └── preview_logger.py
├── env/
│   ├── nero_ros_umi_env.py
│   ├── ros_observation_buffer.py
│   └── ros_action_scheduler.py
├── tools/
│   ├── eval_nero_ros_isolated.py
│   └── cpu_affinity_sampler.py
├── config/
│   └── deploy.yaml
├── tests/
└── experiments/
    └── d405_worker/
```

Responsibilities are intentionally narrow:

```text
eval_nero_ros.py
  CLI parsing, config loading, checkpoint/env construction

PolicyRuntime
  checkpoint EMA weights, normalizer, original UMI obs/action transforms

UmiControlLoop
  original UMI 10 Hz cadence, 6-step replanning, observation-relative timestamps

NeroRosUmiEnv
  ROS2 observation and action adapter

RosObservationBuffer
  timestamp synchronization, pose SLERP, gripper interpolation

RosActionScheduler
  only authority for expiration, action limiting, safety checks, interpolation,
  plan replacement, and ROS2 publishing
```

The active checkpoint contract is enforced at startup. The current policy uses:

```text
camera0_rgb = right fisheye RGB
history horizon = 2
action = 10-D UMI action, horizon 16
observation/action pose representation = relative
EMA weights = state_dicts.ema_model
```

`camera1_rgb` remains supported only when the checkpoint explicitly requires it.
The current single-fisheye checkpoint does not.

## Runtime data path

```text
right fisheye CompressedImage + TCP + gripper
  -> RosObservationBuffer
  -> get_real_umi_obs_dict
  -> checkpoint EMA policy
  -> get_real_umi_action (10-D to 7-D)
  -> UmiControlLoop timestamps
  -> NeroRosUmiEnv.exec_actions
  -> RosActionScheduler
  -> /right/delta_pose
  -> right_continuous_ik_qp_node
  -> /right/control/joint_states
```

The fisheye MJPEG payload is decoded once. Images are not interpolated, copied
from older timestamps, or black-filled. Pose translation is linearly
interpolated, orientation uses quaternion SLERP, and gripper values use bounded
linear interpolation.

Deployment camera preprocessing is deterministic. After strict EMA loading,
`PolicyRuntime` replaces this instance's training `RandomCrop` with `CenterCrop`
of the same size and replaces `ColorJitter` with identity. The current checkpoint
therefore uses 212×212 center cropping followed by the original 224×224 resize,
without random color changes. RGB conversion, normalizer and resize settings
are retained; training code, checkpoint weights and training augmentation are
unchanged. Unknown transforms require an explicit deterministic conversion
instead of silently leaving randomness active. Startup prints
`[model] camera0_rgb deterministic preprocessing: ...`.

Verification on a captured live observation: different RNG seeds produced zero
change in the transformed pixels and encoder features. With diffusion noise
held fixed, predicted actions also had zero change. Diffusion sampling itself
remains stochastic during normal inference; this is distinct from image
augmentation.

## Modes

`--preview` is the default. It loads the checkpoint, reads observations, runs
inference, and optionally records Rerun data, but creates no robot command
publishers.

`--control` is refused unless every `hardware_verification` and safety field in
`config/deploy.yaml` is explicitly configured. Before any control publisher is
created, the runtime also checks the live TCP frame and gripper feedback range.

## Preview command

```bash
source /opt/ros/jazzy/setup.bash
source /home/thor/agx_arm_dual_ws/install/setup.bash
source /home/thor/miniconda3/etc/profile.d/conda.sh
conda activate umi-thor312

cd /home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2

python eval_nero_ros.py \
  --config config/deploy.yaml \
  --checkpoint ../checkpoints/epoch=0060-val_loss=0.0217.ckpt \
  --frequency 10 \
  --steps-per-inference 6 \
  --num-inference-steps 16 \
  --max-cycles 1 \
  --preview \
  --no-rerun \
  --print-action-diagnostics
```

No such command should be run concurrently with the historical D405 worker
experiment.

## Control command

Only after the arm, right feedback topics, exclusive QP path, workspace, and
emergency-stop procedure have been checked independently:

```bash
python eval_nero_ros.py \
  --config config/deploy.yaml \
  --checkpoint ../checkpoints/epoch=0060-val_loss=0.0217.ckpt \
  --frequency 10 \
  --steps-per-inference 6 \
  --num-inference-steps 16 \
  --max-cycles 1 \
  --control-action-limit 2 \
  --control \
  --no-rerun
```

Action timestamps are always based on the observation timestamp:

```text
observation_timestamp + step * 0.1 seconds
```

The deployment config currently sets
`safety.translation_step_check_enabled: false` and
`safety.rotation_step_check_enabled: false`, with both step limiters disabled.
Original model position and rotation waypoints pass
through without step rejection or path slowing. Gripper, timestamp,
finite-value and any enabled workspace checks still apply. Downstream QP
velocity/joint limits still apply, but do not limit jumps in the target pose.
The startup summary explicitly reports both step checks as disabled.

To restore translation checking, set
`safety.translation_step_check_enabled: true`. The optional
`action.translation_step_limiter` can then slow modest overshoots. With a
35 mm output bound and input ratio
1.5, input steps up to 52.5 mm are slowed along the source polyline. Rotation
uses SLERP and gripper width shares the same path progress. Output count and
timestamps remain unchanged; the untraversed tail is left for subsequent
replanning. Steps larger than 52.5 mm, workspace, rotation and gripper
violations still reject and clear the plan. A `[schedule] translation limited`
line reports input and output step sizes. Setting `enabled: false` restores
strict rejection at 35 mm.

Current deployment uses `action.grasp_sync.model_driven_pose: true`. TCP pose
and the base jaw opening follow the model's scheduled approach actions together.
No jaw width selects a Cartesian grasp waypoint; there is no 67 mm pose latch.
With `grasp_sync.lift_guard.enabled: true`, the model's lowest approach pose
since closing started is retained. When the predicted target rises more than
2 mm above it, that model pose becomes the initial waiting target. Before grip
confirmation, a fresh accepted forecast captured after waiting began may update
that target: a descending approach uses the first closing segment's approach
valley, while immediate jaw reopening follows the current correction pose and
opening. This permits visual XY/orientation/downward corrections and reopening
instead of freezing the first failed grasp point. A lift-only forecast does not
retarget the approach. Local upward correction is allowed up to 10 mm above the
initial grasp point (`max_upward_correction_m`); this ceiling never ratchets
upward with new forecasts. This is a correction allowance, not a fixed lift
height. Each meaningful target correction restarts contact settling; the
overall five-second waiting timeout does not reset. After contact confirmation,
both target and jaw command freeze until a post-contact plan is accepted.
Actual TCP
must be within 4 mm, the full bounded jaw correction must be commanded and
unchanged for at least 0.3 seconds, and fresh stable telemetry must provide
contact evidence. Then a new plan captured after confirmation may resume the
model's lift. No fixed lift height or manually selected grasp height is added.
Missing/stale TCP or jaw feedback, empty jaws and absent contact cannot release
this guard. A five-second failure latches the hold until inspection and restart.
Opening corrections during the wait cannot bypass contact confirmation. Failed
or confirmed grasps cannot be retargeted by another unconfirmed forecast.
`[lift_guard] visual correction: xyz=... width=...` reports target updates.
Gripper-only compensation ramps at up to 5 mm/s to at most 3 mm below the
model opening, with a 60 mm compensation floor for the current object. A
model command already below that floor passes through unchanged. Every command
is computed from the current model opening; there is no tightest-width latch.
Small predicted reopenings are allowed immediately, and compensation stops
increasing while the model is reopening. The existing 80 mm threshold resets
the compensation to zero for full opening; it does not block smaller reopening.
Contact evidence gates the initial lift when this guard is enabled; the
separate `[gripper]` contact message remains diagnostic. There is no fixed lift height
or feedback-progress path follower. The recovery tool is unaffected.

If a replacement prediction fails a safety check in this mode, publication
continues at the last actually published TCP and compensated jaw target. The
rejected plan and the old unexecuted future tail are not executed. The active
heartbeat remains true so QP does not restart against a model target far ahead
of measured feedback. A valid replacement resumes normal execution. Jaw
compensation is not reapplied to the held, already compensated command. If a
rejection does release control (for example, before any target was published),
the control loop re-primes from measured feedback before the next plan.

The optional `action.rotation_step_limiter.enabled: true` together with
`safety.rotation_step_check_enabled: true` handles modest angular overshoots
without rejecting their entire action block. Input steps up to 0.12 rad are
resampled along the model polyline to output steps at most 0.08 rad, using
SLERP for orientation and the same source-path fraction for XYZ and jaw width.
Output timestamps and count stay unchanged, so the remaining tail is left to
subsequent replanning. Steps above 0.12 rad, invalid jaw ranges and workspace
violations still reject. Logs report `[schedule] rotation limited` with input
and output maxima. This is angular path slowing, not measured-TCP progression
or a fixed grasp waypoint. Both are currently disabled at operator request.
It does not by itself compensate unmeasured robot
tracking lag; hardware timing and width calibration remain necessary.

The stationary grasp behavior documented below is legacy and applies only when
`model_driven_pose: false`. Its waypoint, settle, verification and timeout
parameters are inactive in the current deployment.

Legacy grasp targets are latched only after a future plan is accepted. The grasp
settling interval persists across accepted replans, including a newly
predicted open-gripper action; replacement and timer publication share a lock.

`grasp_sync` now uses the minimum model width within the first closing segment
before reopening, instead of holding the first width below 80 mm. Jaw closure
is ramped at `close_rate_mps`; current/force settings are not increased. The TCP
hold pose is interpolated where that segment reaches `pose_width_m` (67 mm for
the current object after 70 mm grasped too high): later than the
sometimes-premature 80 mm crossing, but
before the minimum-width action that may already be lifting. If an accepted
replan first exposes that intermediate point before physical closing starts,
the pose is completed once; tighter predictions update only the final jaw width
and cannot drag the grasp pose into the lift phase.

For the current single object, the model remains responsible for the grasp pose
and base closing width. After the model width is reached, stable fresh telemetry
with no contact evidence may start a bounded correction: at most
`max_extra_close_m: 0.003`, at `tighten_rate_mps: 0.005`, and never below the
manually tested `tighten_floor_m: 0.060`. Stale, moving or invalid telemetry
cannot start this correction. For the current object,
`always_tighten_before_verification: true` prevents sensor offset at the
model's approximately 63 mm base width from being mistaken for a secure grip:
the bounded correction must actually command 60 mm before contact can authorize
lift. Once contact is confirmed, that 60 mm command is not loosened during
transport; a tighter model command still passes through, and a model
open/release command releases normally.

Contact confirmation does not resume an action plan inferred during the
artificial low-pose grasp wait. With `require_fresh_post_grasp_plan: true`, the
scheduler holds the grasp TCP and 60 mm command until it receives a plan whose
source observation timestamp is newer than the contact timestamp. Control is
then returned entirely to the model; no fixed vertical-lift primitive is
inserted. The acceptance log reports `predicted_max_lift` for diagnosing whether
the learned post-grasp behavior has been recovered.

Lift requires fresh `/right/gripper/data` telemetry with stable opening. For
this current object, sustained motor current is not used as the gate because a
confirmed hold decayed from about 350 to 73. Instead, the calibrated difference
between measured and commanded opening is used: empty was about 0.65 mm and
held was 1.2--1.35 mm, so `contact_width_error_min_m` is 0.9 mm. `effort`
remains diagnostic raw motor Current, not force in Newtons. These values must be
recalibrated for another object, gripper or operating condition. Width at the
measured empty-close end, moving jaws, stale/invalid data and duplicate source
timestamps cannot authorize lift. After
`close_timeout_sec`, the failure latches for the session and holds the measured
jaw width and grasp TCP; inspect and restart rather than continuing closure
indefinitely.

The model-close phase and the object-specific bounded-tightening phase have
independent `close_timeout_sec` windows. Starting the 60 mm correction can
therefore never fail immediately because earlier model-plan rejection consumed
the first phase's wall time. If a replacement plan fails a safety check while
grasp synchronization is active, the scheduler keeps the previous accepted
stationary grasp hold. It does not release the GUI handoff or re-prime the
controller with a looser measured jaw width.

`contact confirmed` means evidence of jaw contact, not proof of a secure grasp:
a mechanical jam can resemble contact. Verify object-following/slip during a
separately supervised small lift. This code does not implement that lift test.

They are not based on inference completion time. Expired actions are removed
before the per-plan action limit and geometry validation are applied.

For each control session, the first model plan is preceded by a short
controller priming phase. The scheduler publishes the measured TCP pose and
gripper width at the configured rate for `controller_prime_hold_sec`. This lets
the downstream QP initialize from the real measured state before receiving a
future model target that may have advanced during inference. Priming publishes
only current feedback values; it does not interpolate images or relax QP safety
limits.

## Recovery pose

`tools/move_to_recovery_pose.py` restores the confirmed seven-joint task-start
configuration from `config/deploy.yaml`, rather than solving only a Cartesian
TCP target that could produce a different elbow posture. The recorded TCP pose
is checked after the joint move. The script uses `/right/teleop_active` to mute
the GUI during recovery and refuses execution while inference is publishing
`/right/delta_pose`.

Dry run (feedback comparison only):

```bash
python tools/move_to_recovery_pose.py --config config/deploy.yaml
```

Execute only after inference has stopped and the arm area is clear:

```bash
python tools/move_to_recovery_pose.py --config config/deploy.yaml --execute
```

## Verification

Static import check and unit tests:

```bash
cd /home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2
python -m py_compile \
  eval_nero_ros.py runtime/*.py env/*.py tools/*.py \
  experiments/d405_worker/*.py
python -m unittest tests.test_action_scheduler tests.test_observation_buffer
```

Configuration-only check:

```bash
python eval_nero_ros.py --config config/deploy.yaml --check-config
```

Historical D405/shared-memory and isolated-decoder code is under
`experiments/d405_worker/`; see its README for preview-only usage.
