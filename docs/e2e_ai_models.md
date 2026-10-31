# End-to-End (E2E) AI Driving Models in CARLA

This guide details CARLA's closed-loop simulation architecture and tooling for modern **End-to-End (E2E) Autonomous Driving AI Models**, including **QwenDrive 1.0**, **AutoE2E**, **Alpamayo 2 Super**, and **VAD**.

---

## 1. Paradigm Shift: Classical Stacks vs. E2E AI Models

### Classical Autonomous Driving Stacks (e.g., Autoware Universe)
As demonstrated in [`PythonAPI/examples/ros2/test_carla_autoware_ros2.py`](../PythonAPI/examples/ros2/test_carla_autoware_ros2.py), classical self-driving stacks rely on modular, decoupled pipelines:
1. **Sensors**: Raw LiDAR point clouds, camera images, IMU, GNSS.
2. **Localization**: NDT scan-matching against pre-built HD pointcloud maps, fused via EKF with wheel odometry and IMU.
3. **Perception**: Rule-based ground filtering, Euclidean clustering, or 3D bounding box neural networks.
4. **Planning**: Costmap generation, lanelet2 routing, lattice or optimization-based trajectory generation.
5. **Control**: Low-level MPC or Pure Pursuit controllers publishing `/control/command/control_cmd`.

While modular and inspectable, classical stacks suffer from error compounding across modules, struggle with out-of-distribution long-tail scenarios, and require extensive manual HD-map pre-authoring.

### End-to-End (E2E) AI Driving Models
Modern E2E models unify perception, prediction, and motion planning into a single learned representation:
- Multi-camera visual inputs are lifted into Bird's-Eye-View (BEV) representations or processed by Vision-Language-Action (VLA) foundation models.
- Temporal history buffers (past ego-poses, speeds, accelerations) capture vehicle dynamics.
- Natural language instructions or discrete route intents guide high-level decision making.
- The model directly outputs future trajectory waypoints $[(x_i, y_i, t_i)]$ or actuation tokens (acceleration, steering curvature).

---

## 2. Supported E2E Model Families

### A. QwenDrive 1.0 (Alibaba / Qwen Foundation Model)
- **Backbone**: Vision-Language-Action foundation model built upon Qwen3.5-4B / Qwen2-VL.
- **Perception Probe**: External Bird's-Eye-View (BEV) perception probe for 3D bounding box detection, semantic occupancy, and lane boundaries.
- **Planning Expert**: Flow Matching trajectory generator producing continuous future trajectories (typically 50 waypoints over 5.0 seconds at 10 Hz).
- **Explainability**: Chain-of-Thought (CoT) natural language reasoning describing scene hazards, traffic light states, and planned maneuvers.
- **Input Sensor Rig**: Multi-modal surround ring with forward wide-angle and telephoto cameras.

### B. AutoE2E (Autoware Foundation)
- **Design Philosophy**: Compact (~80M parameters), latency-critical, vision-first model designed for real-time automotive hardware (10 Hz).
- **Architecture**: Latent-space BEV reasoning without language overhead; combines 6 surround camera streams with odometry history into a Gated Recurrent Unit (GRU) motion planner.
- **Actuation**: Direct trajectory waypoints or $(a, \kappa)$ steering tire angle and longitudinal acceleration commands interfacing with Autoware ROS 2 control topics.
- **Input Sensor Rig**: nuScenes 6-camera surround rig (1600x900 or downsampled 800x450).

### C. Alpamayo 2 Super (NVIDIA)
- **Architecture**: Paired Vision-Language Model (VLM) + Diffusion Expert.
- **Inputs**: 7 PhysicalAI-AV camera ring (wide, cross, rear, telephoto) + 1.6 seconds of ego-motion history (16 poses at 10 Hz).
- **Outputs**: 64 predicted waypoints (6.4 seconds at 10 Hz) defined in vehicle rear-axle coordinates + Chain-of-Causation (CoC) natural language reasoning.

### D. VAD (Vectorized Autonomous Driving)
- **Architecture**: Multi-task end-to-end vectorized planning model (ICCV 2023).
- **Features**: Vectorized map queries and dynamic agent queries driving end-to-end ego trajectory planning.
- **Input Sensor Rig**: 6 surround cameras (nuScenes configuration).

---

## 3. Simulator Architecture & Closed-Loop Design

Running E2E models in closed-loop simulations requires solving five fundamental challenges:

### 1. Deterministic Lockstep Simulation Advancement
Foundation models (especially multi-billion-parameter VLMs like QwenDrive and Alpamayo) can require from 100 ms to over 1 second per inference cycle. If the simulator advances in real wall-clock time, the vehicle would crash during model inference.
CARLA solves this via **lockstep synchronous simulation**:
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 Hz fixed physics step
world.apply_settings(settings)
```
The simulation advances only when the client calls `world.tick()`. While the E2E model computes the next trajectory, world physics is paused. When inference completes, CARLA advances physics by the exact $\Delta t$, ensuring 100% deterministic, repeatable closed-loop driving.

### 2. Standardized Coordinate Transformations (`e2e_coordinates.py`)
CARLA uses a **left-handed coordinate system** (+X forward, +Y right, +Z up, yaw +right).
Most E2E models, ROS 2, and robotics standards (ISO 8855) use a **right-handed FLU coordinate system** (+X forward, +Y left, +Z up, yaw +left).
The CARLA E2E module provides automatic, bidirectionally verified transformations:
```python
x_carla =  x_iso
y_carla = -y_iso
z_carla =  z_iso
yaw_carla = -yaw_iso
```
It also seamlessly translates between vehicle center coordinates and rear-axle origins (`base_link`).

### 3. Universal Trajectory Tracking (`e2e_tracker.py`)
To convert arbitrary model waypoints into vehicle actuation:
- **Lateral Controller**: Pure Pursuit / Stanley tracking with lookahead scaling ($L_d = \text{clip}(v \cdot t_{preview}, L_{min}, L_{max})$) and steering rate limiting.
- **Longitudinal Controller**: Speed error PID with anti-windup clamping, feedforward acceleration, and automatic stop-hold braking when the vehicle halts.

### 4. Decoupled Model Server Protocol (`e2e_protocol.py`)
To prevent large machine-learning dependencies (CUDA 12, PyTorch 2.5+, vLLM, transformers) from polluting or breaking the CARLA Python client environment:
- The model runs in a dedicated Python 3.12 / container process (`e2e_model_server.py`).
- A high-throughput binary TCP protocol transfers camera frames (contiguous NumPy arrays) and JSON metadata with minimal overhead.

### 5. Native ROS 2 Integration (`e2e_ros2_bridge.py`)
For ROS 2 based E2E models (e.g. AutoE2E or Autoware TRT-VAD):
- CycloneDDS unicast bridging enables communication across macOS/Linux hosts and Docker containers.
- Standard ROS 2 camera topics (`/sensing/camera/*/image_raw`) and actuation topics (`/control/command/control_cmd`, `/planning/trajectory`) are supported natively.

---

## 4. Usage Guide

### Offline Dry-Run Verification
You can verify the entire closed-loop control and trajectory execution loop without needing a running CARLA server:

```bash
# Verify QwenDrive 1.0
python3 PythonAPI/examples/av_stacks/e2e/test_e2e_closed_loop.py --model-type qwendrive_1.0 --dry-run

# Verify AutoE2E
python3 PythonAPI/examples/av_stacks/e2e/test_e2e_closed_loop.py --model-type autoe2e --dry-run

# Verify Alpamayo 2 Super
python3 PythonAPI/examples/av_stacks/e2e/test_e2e_closed_loop.py --model-type alpamayo_2_super --dry-run

# Verify VAD
python3 PythonAPI/examples/av_stacks/e2e/test_e2e_closed_loop.py --model-type vad --dry-run
```

### Live Closed-Loop Simulation
1. Start CARLA:
```bash
./CarlaUnreal.sh -quality-level=Epic
```

2. Run closed-loop driving evaluation (cruising, sharp curves, or lane changing):
```bash
# Evaluate on a sharp curve (automatically selects a curved road segment >35 deg)
./carla-e2e --model-type qwendrive_1.0 --curve --ticks 200

# Evaluate lane change right maneuver
./carla-e2e --model-type qwendrive_1.0 --command lane_change_right --spawn-point 1 --ticks 200

# Evaluate AutoE2E on a specific spawn point
./carla-e2e --model-type autoe2e --curve --speed 8.0 --ticks 200

# Full Python script invocation
python3 PythonAPI/examples/av_stacks/e2e/test_e2e_closed_loop.py \
    --host 127.0.0.1 \
    --port 2000 \
    --model-type qwendrive_1.0 \
    --curve \
    --speed 8.0 \
    --ticks 200 \
    --telemetry /tmp/qwendrive_telemetry.jsonl
```

### Running Decoupled Model Server
On a GPU server:
```bash
python3 PythonAPI/examples/av_stacks/e2e/e2e_model_server.py \
    --model-type qwendrive_1.0 \
    --host 0.0.0.0 \
    --port 8090
```

On the CARLA simulation client:
```bash
python3 PythonAPI/examples/av_stacks/e2e/test_e2e_closed_loop.py \
    --model-type qwendrive_1.0 \
    --use-remote \
    --remote-host <GPU_SERVER_IP> \
    --remote-port 8090
```

---

## 5. Interactive PyGame HUD Dashboard & Simulator Spectator Follow

When closed-loop evaluation runs, two real-time visual streams are provided:

### 1. Dedicated Multi-Camera PyGame Dashboard
A 1600x900 cockpit dashboard window displaying:
- **Multi-Camera Sensor Wall**: Live streams of all model cameras with FOV and role badges (e.g., nuScenes 6-camera surround grid or PhysicalAI-AV 7-camera ring).
- **Model Input Context**: Selected AI architecture, high-level navigation intent (e.g. `LANE_FOLLOW`, `TURN_LEFT`), digital speedometers (current vs target speed), and natural language task prompts.
- **Predicted 2D Trajectory Corridor**: Real-time top-down corridor plotting the model's future waypoints, curvature arc, cross-track error ($e_{ct}$), and heading error ($\psi_{err}$).
- **Output Actuation Gauges**: Real-time visual meters for Throttle (green), Brake (red), and Steering Angle (cyan bi-directional bar).
- **Explainable Driving Console**: Live word-wrapped Chain-of-Thought (CoT) / Chain-of-Causation (CoC) natural language reasoning.

### 2. Main Simulator Window Spectator Follow Camera
The CARLA spectator camera in the main simulator window (`CarlaUnreal`) automatically follows the ego vehicle dynamically in real time:
- `--spectator-view chase` (default): Smooth elevated chase camera positioned behind and above the car ($D=8.5$m, $H=4.2$m, pitch $-16^\circ$), tracking its position and heading through all maneuvers.
- `--spectator-view topdown`: Birds-eye top-down camera positioned $18$m directly above the vehicle looking down.
- `--no-spectator`: Disables spectator camera updates.
- `--no-display`: Disables the Pygame HUD window (useful for headless servers or CI/CD).

