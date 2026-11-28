# Pika 数采、训练数据处理与 NERO 推理操作说明

本文档记录当前工作空间已经实际跑通的操作流程：

```text
Pika 背包数采：/home/thor/agx_arm_dual_ws
训练与推理：  /home/thor/umi-nero-pickplace
背包 SSH：    agilex@192.168.1.114
```

最重要的边界：

```text
数采话题（Pika 背包） /camera_r/... /camera_fisheye_r/... /pika_pose_r /gripper_r/data
推理话题（Thor/NERO） /right/camera/... /right/feedback/... /right/gripper/...
```

这两套话题不能混用。数采不会控制 NERO；推理也不直接读取背包的数采目录。

## 1. 当前数据与模型状态

已经完成的 120 条数据由四组组成：

```text
episode0～29      dataset_ml
episode30～59     dataset_mr
episode60～89     dataset_bl
episode90～119    dataset_br
```

最终 Zarr：

```text
/home/thor/data/pika_ml_mr_bl_br_120_motion_v1_20261127/pika_dataset_0_119.zarr.zip
```

内容：

```text
120 episodes
15,847 synchronized steps
camera0_rgb = 右腕鱼眼，224×224
camera1_rgb = 右腕 D405 RGB，224×224
pose/gripper/timestamp 和各模态 valid mask
SHA-256 = a5152853ec99614363068e1e25ee932d8adb46d8c0e8451689676a79010281d5
```

本版本从原始数据恢复起始接近、转腕和对齐过程，替代旧的固定热身裁剪版本。
有效训练窗口从 5,889 增至 9,046，120/120 审计通过（12 条 TRAINING_READY、
108 条 SALVAGEABLE，局部无效帧仍有 mask）。详细结果和原始归档来源见
[重新清洗报告](/home/thor/data/pika_ml_mr_bl_br_120_motion_v1_20261127/REPORT.md)。
旧的 `/home/thor/data/pika_ml_mr_bl_br_120/` 目录完整保留用于对照。

现有推理 checkpoint：

```text
/home/thor/umi-nero-pickplace/nero_pick_place/checkpoints/epoch=0060-val_loss=0.0217.ckpt
```

它是旧的单鱼眼 checkpoint，只要求 `camera0_rgb`。新 120 条 Zarr 是双相机数据；要使用这批数据提高模型，必须先训练出新的 checkpoint。

## 2. 数采前检查

### 2.1 背包设备和服务

背包开机后登录：

```bash
ssh agilex@192.168.1.114
```

检查服务：

```bash
systemctl is-active pika-os.service http.service
```

正常应均为 `active`。检查 RealSense：

```bash
rs-enumerate-devices -s
```

右侧 D405 必须被 Linux 枚举，当前配置的右侧序列号为：

```text
260522273086
```

如果看不到右侧 D405，重启 ROS 服务没有意义。先检查 USB、Hub、数据线和供电，必要时让背包完整断电后重新上电。设备出现后才执行：

```bash
sudo systemctl restart pika-os.service
```

### 2.2 检查右侧四路数据

在背包执行：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/pika_ros/install/setup.bash
export ROS_DOMAIN_ID=25
export ROS_LOCALHOST_ONLY=0
```

依次检查，每条观察几秒后按 `Ctrl+C`：

```bash
ros2 topic hz /camera_r/color/image_raw
ros2 topic hz /camera_fisheye_r/color/image_raw
ros2 topic hz /pika_pose_r
ros2 topic hz /gripper_r/data
```

要求：

- D405 和鱼眼持续出图，不能为 0；通常接近 30 Hz。
- `/pika_pose_r` 必须持续更新，不能只有一个样本或冻结时间戳。
- `/gripper_r/data` 必须持续更新，通常约 100～125 Hz。

单独检查 Pose 时间戳：

```bash
ros2 topic echo /pika_pose_r --field header.stamp
```

`sec/nanosec` 应连续变化。若同一时间戳不断重复，不要开始采集。

只有 D405、鱼眼、Pose 和夹爪都持续更新时才能开始。网页画面、服务显示 `active` 或 ROS 节点存在，都不能代替实际 topic 检查。

## 3. 采集数据

所有采集命令在 Thor 执行：

```bash
cd /home/thor/agx_arm_dual_ws
```

采集一组新数据，例如 30 条：

```bash
BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/dataset_new \
START_EPISODE=0 \
EPISODE_COUNT=30 \
TASK="pick and place" \
ACTION=capture \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

每条 episode 的操作节奏：

1. 将夹爪放到任务起始位置。
2. 等待程序自动预热，此时不保存动作数据。看到“预热完成，请按 Enter 开始正式录制。”后按 Enter。
3. 看到“正式录制已开始”后，从转腕、对准开始完整示教；按自然速度完成 pick and place。
4. 放置完成后保持约 1～2 秒。
5. 再按 Enter 停止。

预热已经在正式录制之前完成，不需要按 Enter 后再专门等待 1 秒。新 episode
带有 `capture_ready.json`，清洗自动保留正式录制起点，不再额外裁去开头热身段。
保留结尾是为了满足未来动作窗口；两帧历史仍由清洗程序验证。当前默认清洗
对旧 raw 也不再固定裁去开头 1 秒，仅裁首尾可确认静止且超出缓冲的部分。
已有 clean/Zarr 不会自动更新，重处理需输出到新目录。

成功输出应包含所有模态且不是 0：

```text
pikaDepthCamera > 0
pikaFisheyeCamera > 0
pose > 0
gripper > 0
episodeN complete
```

若所有 frame 都是 0，说明录制期间采集进程没有收到 ROS 数据。先恢复设备/topic，不要清洗空 episode。

## 4. 清洗与审计

当前短动作任务使用：

```text
MIN_TRAINING_WINDOWS=20
BOUNDARY_POLICY=motion-buffer
MOTION_HEAD_BUFFER_S=0.5
MOTION_TAIL_BUFFER_S=1.6
```

默认 30 对当前自然速度的 pick and place 偏保守。20 是当前 120 条数据实际采用并全部通过的门槛。

速度判定采用 0.1 秒窗口：平移 15 mm/s、旋转 6°/s、夹爪 5 mm/s 任一满足，
持续约 0.15 秒确认运动。静止阈值更低（8 mm/s、3°/s、2 mm/s），需同时满足。
短转腕、缓慢接近、不确定或无效片段保守保留，内部动作不压缩。
默认开头保留 0.5 秒、结尾 1.6 秒以适配当前 1.5 秒预测窗口；可设置
`MOTION_TAIL_BUFFER_S=0.5` 首尾对称。原始缓冲不足时保留可用数据，不补造帧。
`BOUNDARY_POLICY=keep-all` 完整保留共同区间；`legacy-warmup` 显式重现旧热身裁剪。

### 4.1 全量清洗

```bash
cd /home/thor/agx_arm_dual_ws

BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/dataset_new \
CLEAN_ROOT=/home/agilex/data/dataset_new_clean \
START_EPISODE=0 \
END_EPISODE=29 \
OVERWRITE=1 \
MIN_TRAINING_WINDOWS=20 \
TASK="pick and place" \
ACTION=clean \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

`END_EPISODE` 是闭区间，30 条应写 `0～29`。

质量等级：

```text
TRAINING_READY          可训练，质量最好
SALVAGEABLE             有少量无效槽位，但有效窗口足够，可训练
RECORD_AGAIN_REQUIRED   需要重录
```

`CLEANED` 只表示程序生成了输出，不表示该 episode 可以训练。最终看 `quality=` 和 `TRAINABLE PASS RATE`。

### 4.2 审计 clean 数据

```bash
BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/dataset_new \
CLEAN_ROOT=/home/agilex/data/dataset_new_clean \
START_EPISODE=0 \
END_EPISODE=29 \
TASK="pick and place" \
ACTION=audit-clean \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

目标：

```text
Summary: 30/30 passed
AUDIT PASS RATE: 30/30 (100.00%)
```

鱼眼 `camera_info_uncalibrated` 是当前图像训练允许的警告；它不代表 episode 失败，但数据不能直接用于严格鱼眼几何、去畸变或三维重建。SciPy 的 NumPy 版本 warning 也不是质量失败原因。

## 5. 重录失败 episode

### 5.1 单条重录

`ON_EXISTING=archive` 会先归档旧 episode，再录制同一编号：

```bash
cd /home/thor/agx_arm_dual_ws

BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/dataset_new \
START_EPISODE=7 \
EPISODE_COUNT=1 \
ON_EXISTING=archive \
TASK="pick and place" \
ACTION=capture \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

### 5.2 一次重录多个分散编号

`EPISODES=` 不适用于 `ACTION=capture`。分散编号使用原采集命令逐条调用：

```bash
cd /home/thor/agx_arm_dual_ws

for ep in 2 7 14 24; do
  BACKPACK_HOST=agilex@192.168.1.114 \
  DATASET_ROOT=/home/agilex/data/dataset_new \
  START_EPISODE="$ep" \
  EPISODE_COUNT=1 \
  ON_EXISTING=archive \
  TASK="pick and place" \
  ACTION=capture \
  bash src/data_tools/scripts/pika_dataset_from_thor.sh
done
```

重录后只清洗对应条目：

```bash
BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/dataset_new \
CLEAN_ROOT=/home/agilex/data/dataset_new_clean \
EPISODES="episode2 episode7 episode14 episode24" \
OVERWRITE=1 \
MIN_TRAINING_WINDOWS=20 \
TASK="pick and place" \
ACTION=clean \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

然后再次全量 `ACTION=audit-clean`，不能只看重录条目的清洗汇总。

## 6. 常见数采故障

### 6.1 `pose=0`

```bash
ros2 topic info /pika_pose_r
ros2 topic hz /pika_pose_r
ros2 topic echo /pika_pose_r --field header.stamp
```

发布端为 0、没有消息或时间戳冻结时都不能继续采集。Sense 插拔后还要确认 Tracker 身份和定位绑定正常。

### 6.2 相机全为 0

```bash
rs-enumerate-devices -s
lsusb -t
```

只有设备在 USB 层出现后，才考虑重启 `pika-os.service`。

### 6.3 `valid_windows` 太少

通常由 episode 太短、结尾未来动作不足或中间数据间断造成。新录制使用
预热后按 Enter 的流程，清洗不会再次为热身裁掉动作开头。旧数据仍可能存在
这个问题。应检查逐帧有效性和报告原因；仅增加静止时间不能修复断流或时间戳冻结。

### 6.4 `streams_ready_late`

说明某路数据在 episode 开始后较晚才稳定。新录制会在进入正式录制前等待
各路时间戳连续推进，等待按键时断流会撤销就绪；预热超时会列出等待的话题。

## 7. 生成和验证训练 Zarr

### 7.1 单一数据集

clean 审计 100% 后执行：

```bash
cd /home/thor/agx_arm_dual_ws

BACKPACK_HOST=agilex@192.168.1.114 \
DATASET_ROOT=/home/agilex/data/dataset_new \
CLEAN_ROOT=/home/agilex/data/dataset_new_clean \
LOCAL_ROOT=/home/thor/data/dataset_new \
START_EPISODE=0 \
END_EPISODE=29 \
TASK="pick and place" \
ACTION=convert \
bash src/data_tools/scripts/pika_dataset_from_thor.sh
```

`ACTION=convert` 只同步、转 HDF5 和生成 Zarr，不会重新清洗或审计。

### 7.2 当前四组 120 条

四组原来都叫 `episode0～29`，当前重编号为：

```text
ml + 0
mr + 30
bl + 60
br + 90
```

保留运动的合并结果已经生成：

```text
/home/thor/data/pika_ml_mr_bl_br_120_motion_v1_20261127/pika_dataset_0_119.zarr.zip
```

### 7.3 验证 Zarr

```bash
cd /home/thor/umi-nero-pickplace

export UMI_ROOT=/home/thor/umi-nero-pickplace
export NERO_PICK_PLACE_DATASET_PATH=/home/thor/data/pika_ml_mr_bl_br_120_motion_v1_20261127/pika_dataset_0_119.zarr.zip

/home/thor/miniconda3/envs/data-tools/bin/python \
  nero_pick_place/scripts/validate_dataset.py \
  "$NERO_PICK_PLACE_DATASET_PATH"
```

当前结果应为：

```text
episodes=120 steps=15847
VALIDATION PASSED
```

## 8. 从 Zarr 到 checkpoint

Zarr 不能直接控制机械臂，必须先训练 checkpoint。当前 120 条 Zarr 包含两路图像，当前训练配置也包含 `camera0_rgb` 和 `camera1_rgb`。

训练前执行：

```bash
cd /home/thor/umi-nero-pickplace/nero_pick_place

export UMI_ROOT=/home/thor/umi-nero-pickplace
export NERO_PICK_PLACE_DATASET_PATH=/home/thor/data/pika_ml_mr_bl_br_120_motion_v1_20261127/pika_dataset_0_119.zarr.zip

bash scripts/validate_dataset.sh
bash scripts/check_config.sh
bash scripts/train_debug.sh
```

三个检查均通过后，再正式训练：

```bash
bash scripts/train.sh
```

选择其他 GPU：

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/train.sh
```

双 GPU：

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train_multi_gpu.sh
```

正式推理前，应先对新 checkpoint 做离线验证，例如：

```bash
cd /home/thor/umi-nero-pickplace

CUDA_VISIBLE_DEVICES=0 python \
  nero_pick_place/scripts/eval_checkpoint.py \
  /path/to/new-checkpoint.ckpt \
  --dataset /home/thor/data/pika_ml_mr_bl_br_120_motion_v1_20261127/pika_dataset_0_119.zarr.zip \
  --max-episodes 1 \
  --samples-per-episode 1
```

离线检查通过后，再进入真机 preview。重复评估时需指定一个尚不存在的 `--output-dir`。

## 9. 推理前必须区分 checkpoint 契约

### 9.1 当前旧 checkpoint

```text
epoch=0060-val_loss=0.0217.ckpt
camera0_rgb = 右鱼眼
不需要 D405
```

当前 `deploy.yaml` 的 `include_d405: false` 与它匹配，可以按下面的 preview 流程运行。

### 9.2 由新 120 条训练出的 checkpoint

新训练配置包含 `camera1_rgb`，因此新 checkpoint 会要求：

```text
camera0_rgb = 右鱼眼
camera1_rgb = 右 D405 RGB
```

运行时会从 checkpoint 自动识别 `camera1_rgb` 并打开 D405 输入。但当前 `deploy.yaml` 的 D405 默认源是 shared memory，对应 worker 仍在 `experiments/d405_worker/`，尚未完成生产控制链验收。

新双相机 checkpoint 上真机前必须先完成：

1. 确定生产 D405 来源是 ROS Image 还是独立 shared-memory worker。
2. 若用 ROS Image，将 `d405_source` 改为 `ros_image`，并确认 `d405_rgb_topic` 正确。
3. 实测 D405 与鱼眼时间偏移，填写 `d405_to_fisheye_offset_ms`。
4. 在 `--preview` 下确认两路相机、TCP 和夹爪能持续组成同步 observation。
5. preview 稳定后才能考虑 `--control`。

在这五项完成前，不要用新双相机 checkpoint 控制真机。

## 10. NERO 推理硬件启动

以下命令在 Thor 执行，使用 ROS 2 Jazzy，不是背包的 Humble 环境。

### 10.1 检查 CAN 和设备

```bash
cd /home/thor/agx_arm_dual_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

bash src/agx_arm_dual_tools/scripts/devices/can_activate_dual.sh --check
ip -brief link show type can
```

若 CAN 未就绪：

```bash
bash src/agx_arm_dual_tools/scripts/devices/can_activate_dual.sh
bash src/agx_arm_dual_tools/scripts/devices/can_activate_dual.sh --check
```

### 10.2 启动右侧部署栈

安全默认是机械臂不自动使能、控制门关闭：

```bash
ros2 launch agx_arm_dual_bringup start_right_deploy.launch.py \
  auto_enable:=false \
  control_enabled:=false \
  speed_percent:=10 \
  use_camera:=true \
  use_gripper:=true \
  rviz:=true \
  gui:=true \
  teleop_handoff:=true
```

这个 launch 启动右臂、右夹爪、右鱼眼、RViz/GUI 和控制源仲裁；不启动 QP 和策略推理。

### 10.3 检查推理输入

另开终端并 source：

```bash
source /opt/ros/jazzy/setup.bash
source /home/thor/agx_arm_dual_ws/install/setup.bash
```

检查：

```bash
ros2 topic hz /right/camera/fisheye/color/image_raw/compressed
ros2 topic echo /right/feedback/tcp_pose --once
ros2 topic echo /right/gripper/joint_state --once
```

使用新双相机 checkpoint 时还必须检查 D405 生产话题。

## 11. 推理软件检查和 preview

进入推理环境：

```bash
source /opt/ros/jazzy/setup.bash
source /home/thor/agx_arm_dual_ws/install/setup.bash
source /home/thor/miniconda3/etc/profile.d/conda.sh
conda activate umi-thor312

cd /home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2
```

配置检查：

```bash
python eval_nero_ros.py --config config/deploy.yaml --check-config
```

当前旧单鱼眼 checkpoint 的单周期 preview：

```bash
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

`--preview` 不创建机械臂命令发布器，只读取 observation 并运行模型。必须先成功运行 preview，再启动真机控制。

不要同时运行历史 `experiments/d405_worker/` 实验和生产推理入口，除非正在专门完成新双相机链路验收。

## 12. 真机控制

真机控制存在运动风险，操作者必须在设备旁亲自执行，保持硬件急停可触达。

当前已知安全限制：

```text
workspace_enabled=false
translation_step_check_enabled=false（按操作者要求关闭平移步长拒绝与减速）
rotation_step_check_enabled=false（按操作者要求关闭旋转步长拒绝与减速）
仍保留 QP 速度/关节和夹爪范围限制
当前抓取稳定性尚未完成正式验收
```

正式无人值守前必须重新启用并标定 Cartesian workspace。

### 12.1 启动右侧 QP

另开终端：

```bash
source /opt/ros/jazzy/setup.bash
source /home/thor/agx_arm_dual_ws/install/setup.bash

ros2 launch pika_remote_agx_arm right_velocity_qp_only.launch.py \
  max_linear_velocity:=0.08 \
  max_angular_velocity:=0.30 \
  max_joint_velocity:=0.80
```

### 12.2 使能右臂和控制门

只有在姿态安全、GUI 已用真实反馈初始化、QP 和急停均已确认后执行：

```bash
ros2 service call /right/enable_agx_arm std_srvs/srv/SetBool '{data: true}'
ros2 service call /right/control_enable std_srvs/srv/SetBool '{data: true}'
```

### 12.3 第一次控制只执行短前缀

```bash
cd /home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2

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

第一次测试保持：

```text
max-cycles=1
control-action-limit=2
speed_percent=10
```

确认方向、TCP frame、夹爪方向和停止逻辑全部正确后，才逐步放宽。

## 13. 停止、接管与急停

推理图像预处理已改为确定性：当前 checkpoint 的 95% 随机裁剪替换为
212×212 固定中心裁剪，保留原有 224×224 缩放，关闭 ColorJitter。仅修改
部署时加载的模型实例，不修改训练增强、normalizer 或 checkpoint 权重。
启动打印 `[model] camera0_rgb deterministic preprocessing`。已用真实采样
观测确认：不同随机种子下预处理图像、视觉特征均无变化；固定扩散噪声时
动作预测差异为 0。扩散采样随机性仍正常保留。

当前运行模式为 `action.grasp_sync.model_driven_pose=true`：模型同步决定
TCP 接近位姿和基础夹爪开度。程序不再用 67 mm 等开度选择抓取位置。
当前 `lift_guard.enabled=true`：闭合开始后记录模型最低接近点；模型目标
转为上升超过 2 mm 时，进入夹紧等待。未确认夹稳期间，等待开始后采集的新
观测生成的有效模型计划仍可更新接近点：有下降接近段时采用该段谷底位姿；
要求重新张开时执行当前纠偏位姿和开度。因此不会冻结在第一次夹偏的位置。
只有上升搬运的计划不更新抓取点。允许相对最初抓取点向上最多 10 mm 的
局部纠偏（`max_upward_correction_m`），连续预测不能累加抬高该上限。
目标明显修正后重新计算稳定接触时间，但总的 5 秒等待超时不重置。
日志 `[lift_guard] visual correction` 输出新的 xyz 和开度。
只有实际 TCP 距最新接近点不超过
4 mm、最终有界收紧命令稳定至少 0.3 秒，且新鲜有效遥测提供稳定接触证据，
才允许用确认后新观测重新生成的计划恢复上升。此机制不规定抓取绝对高度
或上抬高度。反馈过时、空夹或未确认接触不能放行；5 秒超时锁定保持，检查
后重启。夹爪单独保留最多 3 mm、
补偿速度 5 mm/s 的收紧，补偿目标不低于 60 mm；模型本身小于 60 mm 的
命令原样通过。补偿始终基于当前预测，不锁存最紧开度；模型小幅张开时立即
跟随，且暂停增加补偿。80 mm 阈值仅用于完全张开时清零补偿，不阻止模型
在该阈值以下张开纠偏，也不参与位姿选择。等待期间张开纠偏不能跳过接触
确认；一旦确认夹稳，先固定该位姿和夹持命令，收到确认后的新观测计划
才恢复模型抬升。超时失败后不会继续自动追踪新的目标。
没有固定上抬动作，也没有按 TCP 到达进度执行路径的功能。回位脚本不变。

当前模式遇到旋转步长等安全拒绝时，会持续发布最后一次实际下发的 TCP 和
已补偿夹爪目标，并保持控制心跳；不会执行被拒绝的预测或旧计划尚未执行的
尾部，也不会重复叠加夹爪补偿。有效新计划到来后恢复模型轨迹。这样避免
控制权中断后 QP 被迫在领先实际位置数厘米的新目标上重新启动，从而报
`STARTUP_TCP_MODEL_MISMATCH` 并锁定。若控制确实已释放，下一轮先用真实
反馈重新 priming。

当前已按操作者要求关闭推理层平移和旋转步长检查及两种减速器，模型原始
位姿直接通过。QP 速度、关节和夹爪范围限制保持不变。以下轻微旋转超限
减速机制仅在重新启用相应检查与减速器时适用：输入相邻角度差不超过 0.12 rad 时，沿模型路径
重新取样，使输出不超过 0.08 rad；XYZ、旋转和夹爪采用同一段路径进度。
时间戳和点数不变，未走完的尾部交给后续重规划。日志会显示
`[schedule] rotation limited`。当前关闭模式下不会出现该减速日志或旋转步长拒绝。
大于 0.12 rad 等异常仍拒绝。此功能没有恢复之前已回退的 TCP 到达进度方案，
也不取代硬件延迟及实际跟踪速度的标定。

以下为旧的静止抓取模式说明，仅在 `model_driven_pose=false` 时适用；
当前模式下，这些抓取位姿选择、等待、接触放行和超时设置均不生效：

抓取接触确认的当前状态：闭合目标取首次闭合段的最小预测开度，
而不是首次低于 80 mm 的开度；停留闭合的 TCP 位姿取该闭合段插值到
67 mm 的位置。80 mm 位姿有时过早，70 mm 实测夹持位置偏高，而最小开度
（约 63 mm）的位姿可能已经进入抬升阶段。67 mm 用于沿模型接近轨迹稍微
向下，同时保留模型原本的闭合时序。单纯等待 0.6 秒不再表示抓取成功。
程序订阅 `/right/gripper/data`，要求新鲜、有效的电机遥测和稳定开度。
机械卡阻也可能产生类似证据，仍需人工观察物体是否跟随抬升、有无打滑。

当前单一物体采用“模型基础开度 + 有界反馈补偿”：先到达模型预测开度；
若遥测稳定、新鲜且仍无接触证据，才以 5 mm/s 继续收紧，最多额外 3 mm，
并且不低于已经人工验证的 60 mm 命令。当前物体必须实际完成 60 mm 下发
以后才允许确认接触，不能在模型约 63 mm 的基础开度处因传感器偏差提前
放行。接触确认后继续保持 60 mm 下发开度，避免运输阶段意外松开；模型
明确预测张开时仍正常释放。

接触确认后不会继续执行低位夹紧等待期间生成的旧动作。调度器保持抓取 TCP
和 60 mm 夹爪命令，直到收到一份观测时间戳晚于接触确认时间的新模型计划，
随后完整交还模型执行抬升与 place，不插入固定 80 mm 等手工抬升动作。日志
中的 `predicted_max_lift` 会报告该新计划相对当前 TCP 预测的最大上抬量。

当前物体夹住后 `|effort|` 会从约 350 衰减到约 73，因此持续电流不再作为
放行条件。60 mm 命令下，空载“实际开度－命令开度”约 0.65 mm，夹持约
1.2--1.35 mm；`deploy.yaml` 使用 0.9 mm 的稳定阻挡差，并结合开度稳定、
时间戳新鲜和空载终点开度共同判断。它只适用于当前物体与当前夹爪条件，
不是通用接触标定。5 秒内未确认则锁定为失败、保持 TCP 和测得的夹爪开度；
检查后重新启动推理。

模型基础闭合和 60 mm 有界收紧分别拥有独立的 5 秒确认窗口，不能在刚开始
60 mm 补偿的同一周期因上一阶段耗时而立刻失败。抓取同步已经开始后，如果
新模型计划因旋转等安全检查被拒绝，调度器继续保持上一份已接受的静止抓取
计划，不释放 GUI 控制，也不重新 priming 为较松的实测夹爪开度。

接触阈值测量（只在停止策略和回位脚本后进行）：

```bash
ros2 topic echo /right/gripper/data
```

记录空载闭合过程和停在任务目标开度时，以及人工确认夹住物体时各数秒的
`distance/effort/enable/error`。夹持样本应重复几次。不要把机械闭合到底
产生的高电流当成物体接触。若空载和夹持电流范围重叠，不能靠该阈值分类，
需要增加视觉或接触传感信息。

独立张开命令使用自动填充的当前 header 时间，避免驱动忽略旧时间戳：

```bash
ros2 topic pub --once /right/gripper/joint_states_ctrl \
  sensor_msgs/msg/JointState \
  "{header: auto, name: ['right_gripper_joint'], position: [0.0967]}"
```

手动闭合时，从实际开度开始小幅逐步减小 `position`，确认物体位置和夹持
状态；不增加驱动电流/力参数。机械臂保持不动，手避开夹指运动范围。

任务回位使用已确认的 7 关节初始构型，并用同期记录的 TCP 位姿交叉验证，
避免仅恢复 TCP 时得到不同的肘部姿态。必须先停止推理；先做只读检查：

```bash
cd /home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2
python tools/move_to_recovery_pose.py --config config/deploy.yaml
```

确认机械臂周围安全后才执行：

```bash
python tools/move_to_recovery_pose.py --config config/deploy.yaml --execute
```

脚本检测到 `/right/delta_pose` 仍有推理发布者时会拒绝运动。回位期间 GUI
由现有互斥信号暂停，结束后自动重新同步到真实关节反馈。

正常停止顺序：

1. 先在推理终端按 `Ctrl+C`，停止持续发布源。
2. 关闭驱动控制门：

   ```bash
   ros2 service call /right/control_enable std_srvs/srv/SetBool '{data: false}'
   ```

3. 检查控制源：

   ```bash
   ros2 topic echo /right/control/source --once
   ```

4. 不再使用机械臂时失能：

   ```bash
   ros2 service call /right/enable_agx_arm std_srvs/srv/SetBool '{data: false}'
   ```

异常运动时先使用现场硬件急停，同时停止推理进程。软件停止服务为：

```bash
ros2 service call /right/emergency_stop std_srvs/srv/Empty '{}'
```

软件 `emergency_stop` 发送当前位置保持目标，不会切断供电，也不会自动阻止之后的新控制命令，不能代替硬件急停。

## 14. 每次操作的最短检查表

### 数采

- [ ] 背包能 SSH。
- [ ] `pika-os.service` active。
- [ ] 右侧 D405 已被 `rs-enumerate-devices` 枚举。
- [ ] D405、鱼眼、Pose、夹爪四路均持续更新。
- [ ] Pose 时间戳不冻结。
- [ ] 每条开始后等 1 秒，结束后等 1～2 秒。
- [ ] clean 使用 `MIN_TRAINING_WINDOWS=20`。
- [ ] `audit-clean` 达到 100%。
- [ ] Zarr 通过 `validate_dataset.py`。

### 推理

- [ ] checkpoint 的相机契约已确认：单鱼眼或双相机。
- [ ] 双相机 checkpoint 的 D405 生产链和时间偏移已经验收。
- [ ] CAN、右臂、夹爪、鱼眼和反馈话题正常。
- [ ] `--check-config` 通过。
- [ ] `--preview --max-cycles 1` 通过。
- [ ] QP 低速启动，急停可触达。
- [ ] 首次控制使用 `--max-cycles 1 --control-action-limit 2`。
- [ ] 结束时先停推理，再关 `control_enable`。

## 15. 相关文档

```text
数采 pipeline： /home/thor/agx_arm_dual_ws/src/data_tools/PIKA_DATASET_PIPELINE.md
AGX 运行手册：  /home/thor/agx_arm_dual_ws/docs/HANDOVER_RUNBOOK.md
项目交接：      /home/thor/umi-nero-pickplace/nero_pick_place/HANDOFF.md
推理说明：      /home/thor/umi-nero-pickplace/nero_pick_place/inference_ros2/README.md
```
