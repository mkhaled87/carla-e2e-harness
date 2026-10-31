# Carla E2E Harness

> **Universal Closed-Loop Evaluation, Benchmarking & Artifact Recording Harness for End-to-End Autonomous Driving Foundation Models in CARLA.**

Compatible with **CARLA 0.9.14, 0.9.15, 0.9.16, and 0.10.0+ (UE4 and UE5 / UE 5.8)**.

---

## Overview

**Carla E2E Harness** is a completely standalone, client-side evaluation framework for modern **End-to-End (E2E) AI Autonomous Driving Models** and **Vision-Language-Action (VLA)** foundation architectures.

Unlike legacy autonomous driving stacks (e.g. Autoware, Apollo) that require complex multi-node ROS 2 networks and pre-built HD pointcloud maps, E2E driving models take raw multi-camera video streams and vehicle kinematics history directly into a neural network to predict future driving trajectories or control actuations.

This harness is **isolated from the CARLA simulator repository**:
* **Zero Unreal Engine / C++ build dependencies**: Operates entirely over CARLA's standard Python client RPC protocol (TCP port 2000).
* **Multi-Version Simulator Compatibility**: Tested and verified across CARLA 0.9.14, 0.9.15, 0.9.16, and 0.10.0 (UE4 & UE 5.8).
* **Remote & Distributed Execution**: Run CARLA on one workstation (or cloud GPU instance) and the E2E model on another, or evaluate completely offline in dry-run mode.

---

## Supported E2E Model Families

| Model Family | Organization | Architecture & Capabilities | Sensor Suite |
|---|---|---|---|
| **AutoE2E** | Autoware Foundation | Compact (~80M param) latent BEV model (ResNet-50 + BEVFormer V2 + GRU motion planner). Supports local PyTorch execution on Apple Silicon (`mps`), CUDA, and ONNX Runtime. | 6-camera surround ring (nuScenes configuration) |
| **QwenDrive 1.0** | Alibaba / Qwen | 4B Vision-Language-Action (VLA) foundation model (Qwen2-VL) with Flow Matching trajectory generator and Chain-of-Thought (CoT) natural language driving reasoning. | Multi-modal surround ring + telephoto context |
| **Alpamayo 2 Super** | NVIDIA | PhysicalAI-AV 7-camera VLA model featuring paired VLM + diffusion expert and Chain-of-Causation (CoC) natural language reasoning. | 7-camera ring (wide, cross, rear, telephoto) |
| **VAD** | OpenDriveLab | Vectorized Autonomous Driving (ICCV 2023) multi-task perception and vectorized trajectory generation. | 6-camera surround ring |
| **Simulated Backend** | Harness Fallback | Closed-loop lane-geometry synthesis via OpenDRIVE map topology; enables instant testing without downloading neural network weights. | All camera rigs supported |

---

## Key Features

### 1. Deterministic Lockstep Simulation
Deep learning inference for foundation models (especially multi-billion-parameter VLMs) requires from 100 ms to over 1 second per step. Running in real wall-clock time causes the vehicle to crash during inference. 
Carla E2E Harness runs in **synchronous lockstep mode**: CARLA's physics clock pauses while the neural network executes its forward pass, then advances by the exact simulation time step ($\Delta t = 0.05$s, 20 Hz). This guarantees 100% deterministic, repeatable closed-loop evaluation regardless of GPU speed.

### 2. Universal Trajectory Tracking (`e2e_tracker.py`)
Converts model trajectories into vehicle control commands:
* **Lateral Controller**: Pure Pursuit / Stanley tracking with dynamic lookahead scaling ($L_d = \text{clip}(v \cdot t_{\text{preview}}, L_{\min}, L_{\max})$) and steering rate limiting.
* **Longitudinal Controller**: Anti-windup PID controller with feedforward acceleration and automatic stop-hold braking when the vehicle halts.

### 3. Bidirectional Coordinate System Handling (`e2e_coordinates.py`)
Automatically translates between:
* **CARLA Left-Handed Frame**: $+X$ forward, $+Y$ right, $+Z$ up, clockwise yaw (+right).
* **ISO 8855 / ROS 2 FLU Right-Handed Frame**: $+X$ forward, $+Y$ left, $+Z$ up, counter-clockwise yaw (+left).
* Translates between vehicle center-of-mass and rear-axle origins (`base_link`).

### 4. Interactive Cockpit Dashboard (`e2e_display.py`)
A dedicated 1600x900 Pygame cockpit displaying:
* Live 6-view surround camera video wall.
* Target speed, steering angle, throttle, and brake effort gauges.
* 2D top-down bird's-eye-view trajectory corridor.
* Explainable Chain-of-Thought (CoT) natural language driving reasoning.
* Bottom-left status displaying **step index**, **actual simulation time in seconds**, and **elapsed wall-clock time** (`Step: 020 | Time: 1.00s | Wall: 14.2s`).

### 5. Simulation Artifact Recording (`--record <dir>`)
When passed `--record /path/to/artifacts/`, the harness records and packages the entire drive:
* **`spectator_drive.mp4`**: High-definition 720p H.264 video captured from an onboard spectator chase camera, encoded at the **exact simulation framerate** ($1/\Delta t$ FPS, e.g. 20 FPS) via `ffmpeg` / OpenCV so the resulting video plays smoothly at real-world driving speed.
* **`cameras/`**: Raw JPEG frame sequences for every onboard camera (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, etc.).
* **`vehicle_pose_telemetry.jsonl`**: High-rate 6-DOF vehicle pose ($x, y, z$, roll, pitch, yaw), linear velocity, linear acceleration, angular velocity ($\omega_x, \omega_y, \omega_z$ in deg/s and rad/s), control actuations, and model predictions for every tick.
* **`sensors/`**: Infraction event logs for collisions and lane invasions.
* **`metadata.json`**: Machine-readable configuration and summary metrics.

---

## Installation

### Prerequisites
* Python 3.10, 3.11, or 3.12
* `ffmpeg` (recommended for H.264 video encoding: `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu)

### Install Carla E2E Harness
```bash
git clone https://github.com/carla-simulator/carla-e2e-harness.git  # or your local path
cd carla-e2e-harness
pip install -e .
```

### Install the CARLA Client Library
Install the `carla` client package matching your CARLA server version:
```bash
# For CARLA 0.9.16:
pip install carla==0.9.16

# For CARLA 0.9.14 or 0.9.15:
pip install carla==0.9.14

# For CARLA 0.10.0 (Mac/Linux source builds):
pip install /path/to/carla/PythonAPI/carla/dist/carla-*.whl
```

---

## Quick Start Guide

### 1. Offline Dry-Run Test (No CARLA Server Required)
Verify your installation and model adapters without launching CARLA:
```bash
carla-e2e --dry-run --ticks 50
```

### 2. Live CARLA Closed-Loop Driving
Start your CARLA simulator server (e.g. `./CarlaUnreal.sh` or `./CarlaUE4.sh`), then run:
```bash
# Connect to local CARLA server on port 2000:
carla-e2e --host 127.0.0.1 --port 2000 --model-type autoe2e --speed 8.0
```

### 3. Evaluate Real Model Weights (AutoE2E on Apple Silicon / CUDA)
```bash
carla-e2e --host 127.0.0.1 --port 2000 \
          --model-type autoe2e \
          --model-path /path/to/AI_MODELS/AutoE2E/models/nuplan-epoch-5/checkpoint.pt \
          --ticks 200 \
          --speed 8.0
```

### 4. Record Simulation Video & Artifacts
Record the drive, camera feeds, and telemetry to a directory:
```bash
carla-e2e --host 127.0.0.1 --port 2000 \
          --model-type autoe2e \
          --model-path /path/to/checkpoint.pt \
          --ticks 300 \
          --record ./my_simulation_run/
```
Once the simulation completes, open the rendered video:
```bash
open ./my_simulation_run/spectator_drive.mp4
```

### 5. Evaluate on Curved Road Segments or Maneuvers
```bash
# Spawn ego at a sharp curve (>35 degree turn):
carla-e2e --curve --speed 8.0

# Request lane change maneuvers:
carla-e2e --command lane_change_left --ticks 100
carla-e2e --command lane_change_right --ticks 100
```

### 6. Remote Decoupled GPU Inference Server (`carla-e2e-server`)
If running a multi-billion-parameter foundation model (like QwenDrive 1.0 or Alpamayo 2 Super) on a dedicated GPU cluster:
```bash
# On the GPU machine (e.g. 192.168.1.50):
carla-e2e-server --host 0.0.0.0 --port 8090 --model-type qwendrive_1.0

# On the CARLA client machine:
carla-e2e --use-remote --remote-host 192.168.1.50 --remote-port 8090
```

---

## CLI Reference Options

```text
usage: carla-e2e [-h] [--host HOST] [--port PORT]
                 [--model-type {qwendrive_1.0,autoe2e,alpamayo_2_super,vad,mock_e2e,custom}]
                 [--model-path MODEL_PATH] [--speed SPEED] [--ticks TICKS]
                 [--dt DT] [--replan-ticks REPLAN_TICKS]
                 [--command {lane_follow,turn_left,turn_right,straight,lane_change_left,lane_change_right,stop}]
                 [--use-remote] [--remote-host REMOTE_HOST] [--remote-port REMOTE_PORT]
                 [--telemetry TELEMETRY] [--no-display] [--no-spectator]
                 [--spectator-view {chase,topdown}] [--curve]
                 [--spawn-point SPAWN_POINT] [--respawn]
                 [--record ARTIFACT_DIR] [--dry-run]

Options:
  --host HOST           CARLA server host (default: 127.0.0.1)
  --port PORT           CARLA server port (default: 2000)
  --model-type TYPE     E2E model family (default: qwendrive_1.0)
  --model-path PATH     Path to model weights (.pt, .pth, .onnx, or HuggingFace ID)
  --speed SPEED         Target cruising speed in m/s (default: 7.0)
  --ticks TICKS         Simulation steps to execute (default: 50)
  --dt DT               Simulation delta seconds (default: 0.05, 20 Hz)
  --replan-ticks N      Inference replan interval (default: 2 ticks = 10 Hz replan)
  --command CMD         High-level navigation intent (default: lane_follow)
  --use-remote          Connect to remote binary TCP model server
  --remote-host HOST    Remote model server host (default: 127.0.0.1)
  --remote-port PORT    Remote model server port (default: 8090)
  --telemetry PATH      Telemetry JSONL output path
  --no-display          Disable the Pygame HUD cockpit window
  --no-spectator        Disable following ego vehicle in main simulator window
  --spectator-view VIEW Chase camera (behind/above) or topdown (default: chase)
  --curve               Spawn ego vehicle on a curved road section (>35 deg bend)
  --spawn-point ID      Explicit spawn point index from CARLA map
  --respawn             Force destroying existing ego vehicle and respawning anew
  --record DIR          Record full simulation artifact bundle (spectator video,
                        camera streams, vehicle pose, and sensors)
  --dry-run             Run offline kinematic dry-run test without CARLA server
```

---

## Directory Structure

```
carla-e2e-harness/
├── adapters/                  # Model adapters for each E2E model architecture
│   ├── base_adapter.py        # Base adapter interface
│   ├── autoe2e_adapter.py     # Autoware AutoE2E (PyTorch / ONNX)
│   ├── qwendrive_adapter.py   # Alibaba QwenDrive 1.0 (VLA / Flow Matching)
│   ├── alpamayo_adapter.py    # NVIDIA Alpamayo 2 Super (PhysicalAI-AV)
│   ├── vad_adapter.py         # Vectorized Autonomous Driving (VAD)
│   └── mock_adapter.py        # Simulated test adapter
├── tests/                     # Comprehensive unit and integration test suite
│   ├── test_e2e_adapters.py
│   ├── test_e2e_artifact_recorder.py
│   ├── test_e2e_closed_loop_mock.py
│   ├── test_e2e_coordinates.py
│   ├── test_e2e_protocol.py
│   ├── test_e2e_rigs.py
│   ├── test_e2e_schema.py
│   └── test_e2e_tracker.py
├── docs/                      # Technical documentation and architecture guides
│   └── e2e_ai_models.md       # Full mathematical & architectural specification
├── e2e_artifact_recorder.py   # Video compilation and artifact packaging
├── e2e_closed_loop_runner.py  # Closed-loop orchestrator
├── e2e_coordinates.py         # CARLA LHD <-> ISO FLU coordinate transformations
├── e2e_display.py             # 1600x900 Pygame interactive cockpit dashboard
├── e2e_model_server.py        # Standalone binary TCP model inference server
├── e2e_protocol.py            # High-throughput zero-copy binary communication
├── e2e_rigs.py                # Sensor rig specifications (nuScenes, Alpamayo, Qwen)
├── e2e_ros2_bridge.py         # CycloneDDS ROS 2 bridge for Autoware integration
├── e2e_schema.py              # Strongly-typed dataclasses and schemas
├── e2e_tracker.py             # Pure Pursuit / Stanley & PID tracking controller
├── test_e2e_closed_loop.py    # CLI entrypoint for closed-loop evaluation
├── pyproject.toml             # Standard PEP 517/621 package specification
└── setup.py                   # Setup script with console entrypoints
```

---

## License

This harness is licensed under the [MIT License](LICENSE).
Carla is an open-source autonomous driving simulator developed by the CARLA team.
