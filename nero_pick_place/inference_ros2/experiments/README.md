# Inference experiments

This directory contains historical and diagnostic experiments. Nothing here is
part of the production inference path.

Current production path:

```text
eval_nero_ros.py
  -> runtime/control_loop.py
  -> env/nero_ros_umi_env.py
  -> env/ros_observation_buffer.py
```

Experiments are preview-only. They must not publish robot control and must not
run concurrently with production inference because they may attach to the same
ROS2 observations, shared-memory rings, or camera devices.
