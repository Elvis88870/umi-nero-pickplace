# Pika/NERO Pick-and-Place 交接文档

更新时间：2026-11-23  
工作区：`/home/thor/umi-nero-pickplace/nero_pick_place`

本文档用于交接当前训练、数据、checkpoint、推理和 NERO 真机适配状态。

## 1. 当前最重要的状态

当前有一个可运行的 100 episode 单鱼眼 checkpoint，但还没有完成稳定的真机抓取验证。

必须优先注意：

1. 新采集批次 `pika_120_20261123_003536` 的 `episode0` 是空的，只有目录和
   `instructions.json/statistic.txt`，所有相机、Pose 和夹爪数据均为 0 帧。
2. 当前训练配置文件继承的 `camera1_rgb` 与手里的单鱼眼 Zarr 不一致。
3. 真机抓取已经出现过“到达罐子附近、夹爪闭合，但未夹稳或抬升打滑”。
4. Cartesian workspace 当前处于关闭状态，真机测试必须人工值守。

## 2. 当前数据集

当前 checkpoint 对应的是单鱼眼 Zarr：

```text
/home/thor/data/pika_100_20261117_234735/pika_dataset_fisheye_0_99.zarr.zip
```

已知信息：

```text
episode 数: 100
存储帧数: 14368
存储频率: 30 Hz
图像 key: data/camera0_rgb
图像尺寸: 224 x 224
源相机: camera/color/pikaFisheyeCamera_r
压缩: imagecodecs_jpegxl
大小: 约 313 MB
```

同批数据的其他目录：

```text
原始数据:
  /home/thor/data/pika_100_20261117_234735/raw

清洗后逐 episode 数据:
  /home/thor/data/pika_100_20261117_234735/clean

双相机 Zarr:
  /home/thor/data/pika_100_20261117_234735/pika_dataset_0_99.zarr.zip

Zarr manifest:
  /home/thor/data/pika_100_20261117_234735/pika_dataset_fisheye_0_99.zarr.zip.manifest.json

整批 clean 元数据:
  /home/thor/data/pika_100_20261117_234735/clean/audit_manifest.json
```

远程采集端：

```text
agilex@192.168.1.114
```

## 3. 当前 120 条重采状态

已创建批次名：

```text
pika_120_20261123_003536
```

远程路径：

```text
/home/agilex/data/pika_120_20261123_003536
/home/agilex/data/pika_120_20261123_003536_clean
```

本地路径：

```text
/home/thor/data/pika_120_20261123_003536
```

当前 `episode0` 不可用。检查结果：

```text
color/pikaDepthCamera_r:    0 frames
color/pikaFisheyeCamera_r:  0 frames
depth/pikaDepthCamera_r:    0 frames
pose/pika_r:                0 frames
gripper/pika_r:             0 frames
```

`pika-os.service` 当时显示 active，但录制没有收到任何 topic 数据。下一步
必须在背包端确认每个 topic 实际持续发布，再重新采集这条测试 episode。

不要直接继续录剩余 119 条，直到新建 episode0 通过：

```text
ACTION=audit
ACTION=clean
ACTION=audit-clean
```

确认主题持续发布后，用新的批次名重新执行试录：

```bash
cd /home/thor/agx_arm_dual_ws

RUN=pika_120_$(date +%Y%m%d_%H%M%S)

BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/$RUN \
CLEAN_ROOT=/home/agilex/data/${RUN}_clean \
LOCAL_ROOT=/home/thor/data/$RUN \
START_EPISODE=0 \
EPISODE_COUNT=1 \
CAMERA_PREFIXES=pikaDepthCamera,pikaFisheyeCamera \
TASK="pick and place" \
ON_EXISTING=error \
SKIP_SERVICE_CHECK=0 \
ACTION=capture \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

试录通过后，再以 `START_EPISODE=1 EPISODE_COUNT=119` 采集剩余数据。

## 4. 当前 checkpoint

当前评估和推理主 checkpoint：

```text
/home/thor/umi-nero-pickplace/nero_pick_place/checkpoints/epoch=0060-val_loss=0.0217.ckpt
```

该文件约 2.6 GB。

另有一个：

```text
/home/thor/umi-nero-pickplace/nero_pick_place/checkpoints/latest.ckpt
```

约 4.2 GB，不是当前 `epoch=0060` checkpoint 的等价软链接。不要覆盖。

当前 checkpoint 的主要契约：

```text
policy: DiffusionUnetTimmPolicy
weights: state_dicts.ema_model
action: 10D
action horizon: 16
action down_sample_steps: 3
observation horizon: 2
observation interval: 0.1 s
pose representation: relative
```

## 5. 视觉编码器

当前 checkpoint 使用：

```text
TimmObsEncoder
model_name: vit_base_patch16_clip_224.openai
pretrained: true
frozen: false
global pooling: CLS token
GroupNorm: true
ImageNet norm: true
```

训练时从 CLIP ViT-B/16 预训练权重初始化，整个视觉编码器参与微调。

部署时 `PolicyRuntime` 先设置 `pretrained=False`，避免重新下载基础权重，然后
严格加载 checkpoint 中的 EMA 权重。因此部署使用的是训练后的 ViT 权重，
不是随机初始化，也不是未微调的原始 CLIP。

## 6. 当前训练配置

文件：

```text
config/pika_nero_pick_place_train.yaml
```

关键设置：

```yaml
dataset_frequency: 30.0
obs_down_sample_steps: 3
low_dim_obs_horizon: 2
img_obs_horizon: 2
action_horizon: 16
obs_pose_repr: relative
action_pose_repr: relative
camera_obs_latency: 0.0
robot_obs_latency: 0.0
gripper_obs_latency: 0.0
```

重要风险：

`diffusion_policy/config/task/umi.yaml` 当前默认包含 `camera1_rgb`，而当前
checkpoint 和 fisheye Zarr 只有 `camera0_rgb`。重新训练前必须确认最终 Hydra
配置只包含 `camera0_rgb`，否则 dataset/shape_meta 会不匹配。

`scripts/validate_dataset.py` 当前也要求 `camera1_rgb` 和 `d405_valid`，
与现有单鱼眼 Zarr 不一致。重新训练前必须同步修正验证器。

## 7. 训练命令

训练服务器应设置：

```bash
export UMI_ROOT=/root/umi-nero-pickplace
export NERO_PICK_PLACE_ROOT=/root/umi-nero-pickplace/nero_pick_place
export NERO_PICK_PLACE_DATASET_PATH=/path/to/pika_dataset_fisheye_0_119.zarr.zip
```

先解析配置：

```bash
bash scripts/check_config.sh
```

结构验证：

```bash
bash scripts/validate_dataset.sh
```

单步训练冒烟：

```bash
bash scripts/train_smoke.sh
```

正式训练：

```bash
bash scripts/train.sh
```

双 GPU：

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train_multi_gpu.sh
```

## 8. 离线评估

脚本：

```text
scripts/eval_checkpoint.py
scripts/render_eval_rerun.py
```

评估命令：

```bash
python scripts/eval_checkpoint.py \
  checkpoints/epoch=0060-val_loss=0.0217.ckpt \
  --dataset /home/thor/data/pika_100_20261117_234735/pika_dataset_fisheye_0_99.zarr.zip \
  --samples-per-episode 10
```

输出包括：

- TCP position error，mm
- SO(3) rotation error，degree
- gripper width error，mm
- hold-current-pose baseline
- 每个 horizon step 的误差

该评估是 teacher-forced offline evaluation，不证明闭环抓取成功率。

## 9. NERO 推理代码位置

生产推理目录：

```text
/home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2
```

结构：

```text
eval_nero_ros.py
runtime/
  policy_runtime.py
  control_loop.py
  preview_logger.py
env/
  nero_ros_umi_env.py
  ros_observation_buffer.py
  ros_action_scheduler.py
tools/
  move_to_recovery_pose.py
  gripper_latency_probe.py
  eval_nero_ros_isolated.py
  cpu_affinity_sampler.py
experiments/d405_worker/
```

职责简述：

```text
eval_nero_ros.py
  CLI、配置、checkpoint/env 装配

PolicyRuntime
  checkpoint、EMA、normalizer、原版 UMI obs/action transform

UmiControlLoop
  10 Hz 控制、每 6 步重推理、observation-relative 时间戳

NeroRosUmiEnv
  ROS2 observation/action adapter

RosObservationBuffer
  鱼眼、TCP、gripper 时间同步和插值

RosActionScheduler
  过期过滤、限幅、计划替换、QP priming、grasp_sync、50 Hz 发布
```

当前生产输入只有：

```text
camera0_rgb = /right/camera/fisheye/color/image_raw/compressed
TCP         = /right/feedback/tcp_pose
gripper     = /right/gripper/joint_state
```

D405/worker 仅是实验，不在当前生产路径。

## 10. 当前推理时间语义

```text
模型输出 action horizon = 16
action interval = 0.1 s
重规划间隔 = 0.6 s
scheduler publish rate = 50 Hz
QP control rate = 100 Hz
```

当前部署配置：

```yaml
inference_latency_sec: 0.0
gripper_latency_sec: 0.03
qp_latency_sec: 0.0
workspace_enabled: false
max_translation_step_m: 0.035
max_rotation_step_rad: 0.08
gripper_max_m: 0.098
```

推理 P50 约 225 ms。该耗时只作为诊断记录，不从 observation-relative 时间轴
再次扣除，否则会错误丢弃更多动作。

## 11. 当前真机硬件结构

### 机械臂

```text
NERO 右臂
namespaces: /right
CAN: can_right
反馈: /right/feedback/*
关节命令: /right/control/joint_states
```

控制链：

```text
RosActionScheduler
  -> /right/delta_pose
  -> right_continuous_ik_qp_node
  -> /right/control/joint_states
  -> right agx_arm_ctrl_single
  -> NERO
```

推荐初始 QP 限速：

```text
max_linear_velocity: 0.08 m/s
max_angular_velocity: 0.30 rad/s
max_joint_velocity: 0.80 rad/s
```

这些参数在 launch 命令中传入，不在训练 checkpoint 内。

### 右鱼眼

```text
设备:
  /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.2.1:1.0-video-index0

实际设备:
  /dev/video14

topic:
  /right/camera/fisheye/color/image_raw/compressed

frame_id:
  right_camera_fisheye_link
```

左鱼眼是 `4.1.1.1`，不要混用。

### 右夹爪

```text
设备:
  /dev/serial/by-path/platform-a80aa10000.usb-usb-0:4.2.2.4:1.0-port0

实际设备:
  /dev/ttyUSB1

反馈:
  /right/gripper/joint_state

原始状态:
  /right/gripper/data

命令:
  /right/gripper/joint_states_ctrl
```

左夹爪是 `4.1.1.4`。

## 12. GUI 与 QP 切换

右侧部署已配置 `teleop_handoff:=true`：

```text
GUI
  -> /right/control/slider/joint_states

QP
  -> /right/control/joint_states
  -> /right/teleop_active

joint_command_mux
  QP active 时屏蔽 GUI
  QP inactive 或心跳 0.5 秒超时后
    先让 GUI 同步实际反馈
    再由 GUI 接管
```

因此 GUI 可以一直打开，推理运行时不会和 QP 同时向驱动发命令。

## 13. 当前抓取同步机制

`RosActionScheduler` 新增 `grasp_sync`：

```yaml
grasp_sync:
  enabled: true
  close_threshold_m: 0.080
  position_tolerance_m: 0.012
  settle_sec: 0.6
```

流程：

```text
检测夹爪 open -> close
锁存模型预测抓取 TCP
夹爪保持张开，等待实际 TCP 进入抓取点 12 mm
闭合夹爪
保持 0.6 s
再执行后续抬升动作
```

日志应出现：

```text
[grasp_sync] locked grasp target
[grasp_sync] pose reached; closing and settling for 0.60s
[grasp_sync] settle complete; resuming plan
```

该机制尚未完成稳定抓取验收。

## 14. 已观察到的真机现象

### 14.1 抓取点过早

一次调试记录中，夹爪第一次闭合时 TCP 约为：

```text
z = 0.1247 m
```

而预测抓取高度约为：

```text
z = 0.08–0.10 m
```

说明 TCP 跟随滞后会令夹爪在空中提前闭合。`grasp_sync` 就是针对该问题。

### 14.2 夹不稳/打滑

后续测试中出现过：

```text
到达罐子附近
夹爪闭合到约 0.0635 m
执行抬升时罐子滑脱
```

人工检查确认并不总是夹实。仍需联调：

- 抓取高度
- 夹爪停留时间
- QP 跟随速度
- 夹爪闭合力/电流阈值
- 抬升阶段速度

### 14.3 空场景仍执行轨迹

训练集只有成功的 pick-and-place，没有 no-object/no-op 样本。把物体拿走后，
模型仍会输出近似抓取轨迹。部署端目前没有物体存在性门控。

## 15. 坐标系问题

训练元数据：

```text
coordinate_frame = libsurvive_world
```

训练初始 TCP 中位数约为：

```text
[-0.391, 0.910, 0.014]
```

NERO 初始 TCP 约为：

```text
[-0.585, -0.004, 0.138]
```

模型学习的是相对动作，因此绝对原点不同本身不是致命问题。但仍需正式确认
Pika tracker 工具轴与 NERO `right_tcp_link` 工具轴之间的固定变换：

```text
T_nero_tcp_from_pika_tracker
```

目前没有该外参的正式标定和残差报告。单次任务成功不能证明任意初始姿态下
动作方向都正确。

## 16. 当前未提交改动

训练仓库中当前存在未提交修改：

```text
diffusion_policy/common/sampler.py
diffusion_policy/config/task/umi.yaml
diffusion_policy/dataset/umi_dataset.py
nero_pick_place/README.md
nero_pick_place/config/pika_nero_pick_place_train.yaml
nero_pick_place/scripts/validate_dataset.py
nero_pick_place/inference_ros2/
nero_pick_place/scripts/eval_checkpoint.py
nero_pick_place/scripts/render_eval_rerun.py
tests/test_sampler_valid_masks.py
```

其中 sampler/dataset 修改加入了 valid-mask 过滤和 normalizer 保护。

## 17. 下一步优先级

### P0：修复 120 条采集

1. 在背包端确认 `/camera_r/color/image_raw` 等 topic 有帧。
2. 确认 data capture 进程没有立即退出。
3. 重录一个新的测试 episode。
4. 只有 audit/clean/audit-clean 都通过后才批量采集。

### P0：统一单相机训练契约

重新训练前必须解决：

```text
umi.yaml 仍包含 camera1_rgb
validate_dataset.py 仍要求 camera1_rgb/d405_valid
当前 Zarr 只有 camera0_rgb
```

否则新训练无法读取当前单鱼眼数据。

### P1：提高数据多样性

新 120 条不要全部固定物体位置和轨迹。至少变化：

- 物体左右位置
- 物体远近
- 物体朝向
- 起始 TCP
- 光照和背景
- 接近和抓取方式

如果希望模型学会“没有物体时不动”，需要额外采集 no-object/no-op episode。

### P1：真机抓取联调

1. 验证 `grasp_sync` 日志顺序。
2. 测量夹爪闭合后的 `/right/gripper/data.effort`。
3. 判断是否夹实。
4. 再增加抬升速度或放大连续动作数。
5. 完成一次稳定 pick-lift-place。

### P2：坐标/工具外参标定

记录：

```text
/right/feedback/tcp_pose
Pika tracker pose
```

至少三个非共线姿态，求解固定 tracker-to-TCP 变换并报告位置/旋转残差。

## 18. 安全规则

当前 `workspace_enabled=false`，只有：

- 单步平移限制
- 单步旋转限制
- QP 速度/关节限制
- 夹爪范围
- 急停

重要约束：

1. 打开控制门前必须确认急停。
2. 不要同时保留 GUI 直连和 QP 作为两个实际控制源。
3. 推理结束或 QP 关闭后，检查：

```bash
ros2 topic echo /right/control/source --once
```

4. 正式无人值守运行前必须重新开启 Cartesian workspace。

## 19. 常用运行命令

训练数据采集测试命令见本工作区父目录的 Pika pipeline 文档：

```text
/home/thor/agx_arm_dual_ws/src/data_tools/PIKA_DATASET_PIPELINE.md
```

推理预览：

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

真机命令必须由操作者亲自执行，并保持急停可触达。
