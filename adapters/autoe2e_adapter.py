"""Adapter for AutoE2E (Autoware Foundation E2E compact BEV model).

Characteristics:
- Compact ~80M parameter vision-first model designed for 10 Hz real-time automotive deployment.
- ResNet-50 + BEVFormer V2 temporal cross-attention + GRU motion planner.
- Ingests 6 surround cameras + ego odometry history.
- Computes BEV representations + GRU trajectory planner + direct acceleration and steering commands.
- Fully supports local PyTorch weights (.pt) and ONNX runtime engines (.onnx).
"""

from __future__ import annotations

import glob
import inspect
import math
import os
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from adapters.base_adapter import (
    BaseE2EAdapter,
    plan_trajectory_with_lane_geometry,
    resolve_torch_device,
)
from e2e_schema import (
    CoordinateFrame,
    E2EObservation,
    E2EPrediction,
    HighLevelCommand,
    ModelType,
)


def integrate_controls_to_trajectory(
    accel: np.ndarray,
    curvature: np.ndarray,
    v0: float,
    dt: float = 0.1,
    min_projection_speed: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate acceleration and curvature into ego-centric XY positions (ISO FLU: +X fwd, +Y left).

    Args:
        accel: (T,) predicted longitudinal acceleration (m/s^2).
        curvature: (T,) predicted path curvature (1/m, +left).
        v0: Initial speed (m/s).
        dt: Timestep (s).
        min_projection_speed: Minimum forward speed used for spatial trajectory projection
            to ensure downstream path trackers have an adequate forward lookahead horizon
            even from a standstill (m/s).

    Returns:
        positions: (T, 2) array of [x, y] in metres.
        headings: (T,) yaw angles relative to initial pose (rad).
        speeds: (T,) longitudinal speeds (m/s).
    """
    T = len(accel)
    positions = np.zeros((T, 2), dtype=np.float32)
    headings = np.zeros(T, dtype=np.float32)
    speeds = np.zeros(T, dtype=np.float32)

    # Use a forward projection speed floor for trajectory geometry when stopped/crawling
    v_proj = max(float(v0), min_projection_speed)
    v_actual = float(v0)
    theta = 0.0
    x, y = 0.0, 0.0

    for t in range(T):
        a_t = float(accel[t])
        c_t = float(curvature[t])
        v_actual = max(0.0, v_actual + a_t * dt)
        v_proj = max(0.5, v_proj + a_t * dt)
        theta = theta + c_t * v_proj * dt
        x = x + v_proj * math.cos(theta) * dt
        y = y + v_proj * math.sin(theta) * dt
        positions[t] = [x, y]
        headings[t] = theta
        speeds[t] = v_actual

    return positions, headings, speeds


def _resolve_autoe2e_model_path(search_path: str | None) -> str | None:
    """Find valid AutoE2E PyTorch (.pt) or ONNX (.onnx) model checkpoint."""
    if search_path:
        if os.path.isfile(search_path):
            return search_path
        if os.path.isdir(search_path):
            # Check for standard model names
            for name in ["checkpoint.pt", "model.onnx", "autoe2e_80m.onnx"]:
                p = os.path.join(search_path, name)
                if os.path.isfile(p):
                    return p
            # Search subdirectories for .pt or .onnx
            pts = glob.glob(os.path.join(search_path, "**", "*.pt"), recursive=True)
            if pts:
                # Prefer nuplan-epoch-5 or kitscenes-epoch-5 if present
                for pt in pts:
                    if "nuplan" in pt or "epoch-5" in pt:
                        return pt
                return pts[0]
            onnxs = glob.glob(os.path.join(search_path, "**", "*.onnx"), recursive=True)
            if onnxs:
                return onnxs[0]

    # Check environment variable
    env_path = os.environ.get("AUTOE2E_MODEL_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check standard local directories in priority order
    default_candidates = [
        "/Users/ra94giv/Workspace/AI_MODELS/AutoE2E/models/nuplan-epoch-5/checkpoint.pt",
        "/Users/ra94giv/Workspace/AI_MODELS/AutoE2E/models/kitscenes-epoch-5/checkpoint.pt",
        "/Users/ra94giv/Workspace/AI_MODELS/AutoE2E/checkpoint.pt",
    ]
    for cand in default_candidates:
        if os.path.isfile(cand):
            return cand

    return None


def _ensure_autoe2e_source_imported() -> bool:
    """Ensure auto_e2e python modules are in sys.path and can be imported."""
    try:
        from model_components.auto_e2e import AutoE2E
        return True
    except ImportError:
        pass

    # Try standard paths where repo was saved
    search_dirs = [
        "/Users/ra94giv/Workspace/AI_MODELS/AutoE2E/src/Model",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../AI_MODELS/AutoE2E/src/Model")),
        "/tmp/auto_e2e_repo/Model",
        os.environ.get("AUTOE2E_SOURCE_DIR", ""),
    ]
    for d in search_dirs:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
            try:
                from model_components.auto_e2e import AutoE2E
                return True
            except ImportError:
                continue
    return False


class AutoE2EAdapter(BaseE2EAdapter):
    """Adapter for the Autoware Foundation AutoE2E model."""

    def __init__(
        self,
        model_path: str | None = None,
        target_fps: float = 10.0,
        planning_horizon_s: float = 3.0,
        device: str = "auto",
        wheelbase: float = 2.85,
    ) -> None:
        super().__init__(model_type=ModelType.AUTO_E2E, model_id="autoe2e_80m")
        self.model_path = model_path
        self.target_fps = target_fps
        self.planning_horizon = planning_horizon_s
        self.num_points = int(round(planning_horizon_s * target_fps))
        self.dt = 1.0 / target_fps
        self.device = resolve_torch_device(device)
        self.wheelbase = wheelbase

        self.pytorch_model: Any = None
        self.onnx_session: Any = None
        self.use_simulated_backend = True

        # Rolling egomotion history (64 steps x 4 signals: speed, accel, yaw_rate, curvature)
        self.history_len = 64
        self.egomotion_history: list[list[float]] = []
        self.last_speed = 0.0
        self.last_timestamp_us: int | None = None

        # Camera canonical names for the 6-view surround rig
        self.canonical_camera_slots = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

    def load(self, force_simulated: bool = False, **kwargs: Any) -> None:
        """Load real PyTorch/ONNX checkpoint if available, or fall back to simulated backend."""
        if force_simulated:
            self.use_simulated_backend = True
            self.is_loaded = True
            print(
                "[AutoE2E NOTICE] Running with SIMULATED BACKEND (forced simulated mode).\n"
                "  No neural network model is loaded. Trajectories are synthesized via OpenDRIVE road lane geometry.\n"
                "  To execute real weights: pass --model-path <path/to/checkpoint.pt>."
            )
            return

        resolved_path = _resolve_autoe2e_model_path(self.model_path)

        if not resolved_path:
            self.use_simulated_backend = True
            self.is_loaded = True
            print(
                f"[AutoE2E NOTICE] No model weights found at model_path={self.model_path}.\n"
                f"  Checked AI_MODELS default paths without finding a checkpoint.\n"
                f"  No neural network model is active. Running with SIMULATED BACKEND.\n"
                f"  Waypoint trajectories and controls are synthesized via OpenDRIVE road lane geometry.\n"
                f"  To execute real weights: specify --model-path <path/to/checkpoint.pt>."
            )
            return

        self.model_path = resolved_path

        # 1. Try PyTorch Checkpoint (.pt)
        if resolved_path.endswith(".pt") or resolved_path.endswith(".pth"):
            try:
                import torch
                if not _ensure_autoe2e_source_imported():
                    raise ImportError("Could not find or import AutoE2E architecture from auto_e2e source repository.")

                from model_components.auto_e2e import AutoE2E

                print(f"[AutoE2E] Loading PyTorch checkpoint from {resolved_path} onto device '{self.device}'...")
                data = torch.load(resolved_path, map_location="cpu", weights_only=False)

                if isinstance(data, dict) and "model_state_dict" in data and "config" in data:
                    config = data["config"]
                    state_dict = data["model_state_dict"]
                elif isinstance(data, dict) and "state_dict" in data:
                    state_dict = data["state_dict"]
                    config = data.get("config", {})
                else:
                    state_dict = data
                    config = {}

                valid_keys = set(inspect.signature(AutoE2E.__init__).parameters) - {"self"}
                model_kwargs = {k: v for k, v in config.items() if k in valid_keys}
                model_kwargs["is_pretrained"] = False

                self.pytorch_model = AutoE2E(**model_kwargs)
                self.pytorch_model.load_state_dict(state_dict, strict=False)
                self.pytorch_model.to(self.device)
                self.pytorch_model.eval()

                param_count = sum(p.numel() for p in self.pytorch_model.parameters()) / 1e6
                self.use_simulated_backend = False
                self.is_loaded = True
                print(
                    f"[AutoE2E] Real PyTorch model loaded successfully!\n"
                    f"  - Parameters: {param_count:.1f}M\n"
                    f"  - Architecture: ResNet-50 + BEVFormer V2 + GRU\n"
                    f"  - Device: {self.device}\n"
                    f"  - Checkpoint: {resolved_path}"
                )
                return
            except Exception as e:
                print(f"[AutoE2E WARNING] Failed to load PyTorch checkpoint ({e}). Trying ONNX fallback...")

        # 2. Try ONNX Runtime Engine (.onnx)
        try:
            import onnxruntime as ort
            print(f"[AutoE2E] Loading ONNX model from {resolved_path}...")
            self.onnx_session = ort.InferenceSession(resolved_path)
            self.use_simulated_backend = False
            self.is_loaded = True
            print(f"[AutoE2E] ONNX model loaded successfully from {resolved_path}.")
            return
        except Exception as e:
            self.use_simulated_backend = True
            self.is_loaded = True
            print(
                f"[AutoE2E NOTICE] Failed to load ONNX model ({e}).\n"
                f"  No actual AI model is active. Running with SIMULATED BACKEND.\n"
                f"  Waypoint trajectories are synthesized via OpenDRIVE road lane geometry."
            )

    def infer(self, observation: E2EObservation) -> E2EPrediction:
        """Execute AutoE2E BEV perception and GRU trajectory generation."""
        t0 = time.perf_counter()

        if not self.use_simulated_backend and self.pytorch_model is not None:
            try:
                return self._infer_pytorch(observation, t0)
            except Exception as exc:
                print(f"[AutoE2E ERROR] PyTorch inference failed ({exc}), falling back to simulated backend.")
                return self._infer_simulated(observation, t0)
        elif not self.use_simulated_backend and self.onnx_session is not None:
            try:
                return self._infer_onnx(observation, t0)
            except Exception as exc:
                print(f"[AutoE2E ERROR] ONNX inference failed ({exc}), falling back to simulated backend.")
                return self._infer_simulated(observation, t0)
        else:
            return self._infer_simulated(observation, t0)

    def _infer_pytorch(self, observation: E2EObservation, t0: float) -> E2EPrediction:
        """Run real forward pass through AutoE2E PyTorch neural network."""
        import torch
        import torchvision.transforms.functional as TF

        current_speed = observation.ego_state.speed_mps
        ts_us = observation.timestamp_us
        dt = (ts_us - self.last_timestamp_us) * 1e-6 if self.last_timestamp_us is not None else self.dt
        if dt <= 0.0 or dt > 1.0:
            dt = self.dt

        # Compute instantaneous acceleration and curvature in ISO FLU (rad/s, rad/m)
        accel_inst = (current_speed - self.last_speed) / dt
        # CARLA wz is in degrees/s (+right). Convert to ISO FLU rad/s (+left):
        yaw_rate_iso = -math.radians(observation.ego_state.wz)
        curvature_inst = float(np.clip(yaw_rate_iso / max(current_speed, 0.5), -0.5, 0.5))

        self.last_speed = current_speed
        self.last_timestamp_us = ts_us

        # Maintain 64-step rolling egomotion history in ISO FLU
        ego_entry = [float(current_speed), float(accel_inst), float(yaw_rate_iso), float(curvature_inst)]
        if not self.egomotion_history:
            self.egomotion_history = [ego_entry for _ in range(self.history_len)]
        else:
            self.egomotion_history.append(ego_entry)
            if len(self.egomotion_history) > self.history_len:
                self.egomotion_history.pop(0)

        flat_egomotion = []
        for step_data in self.egomotion_history:
            flat_egomotion.extend(step_data)
        egomotion_tensor = torch.tensor([flat_egomotion], dtype=torch.float32, device=self.device)

        # Ingest 6 surround camera images and resize to (512, 512)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        cam_tensors = []
        for slot in self.canonical_camera_slots:
            img_np = self._find_camera_image(observation.camera_images, slot)
            # Permute (H, W, 3) -> (3, H, W)
            t = torch.from_numpy(img_np).permute(2, 0, 1).float().to(self.device)
            if t.max() > 1.0:
                t = t / 255.0
            t_resized = TF.resize(t, [512, 512])
            cam_tensors.append(t_resized)

        # Batch: [1, 6, 3, 512, 512]
        camera_tiles = torch.stack(cam_tensors, dim=0).unsqueeze(0)
        camera_tiles = (camera_tiles - mean.unsqueeze(1)) / std.unsqueeze(1)

        # Standard context tensors
        map_context = torch.zeros(1, 14, 256, 256, device=self.device)
        route_mask = torch.zeros(1, 2, 256, 256, device=self.device)
        visual_history = torch.zeros(1, 896, device=self.device)

        # Forward pass (explicit validity gates off for zero-filled map/route tensors)
        with torch.no_grad():
            out = self.pytorch_model(
                camera_tiles,
                map_context,
                visual_history,
                egomotion_tensor,
                route_mask=route_mask,
                map_valid=torch.zeros(1, dtype=torch.bool, device=self.device),
                route_valid=torch.zeros(1, dtype=torch.bool, device=self.device),
                mode="eval",
            )
            if str(self.device) == "mps":
                torch.mps.synchronize()

        # Unpack 64-step (longitudinal accel, curvature)
        pred_accel = out[0, 0::2].cpu().numpy()
        pred_curv = out[0, 1::2].cpu().numpy()

        # Integrate into ISO FLU coordinates (+X fwd, +Y left) with minimum projection horizon
        positions_iso, headings_iso, speeds_profile = integrate_controls_to_trajectory(
            pred_accel, pred_curv, v0=current_speed, dt=self.dt, min_projection_speed=2.5
        )

        # Convert to CARLA Left-Handed coordinates (+X fwd, +Y right, +Z up)
        waypoints_carla = np.column_stack([
            positions_iso[:, 0],
            -positions_iso[:, 1],
            np.zeros(len(positions_iso), dtype=np.float32),
        ])

        # Slice to requested horizon points
        num_pts = min(self.num_points, len(waypoints_carla))
        waypoints_carla = waypoints_carla[:num_pts]
        speeds_profile = speeds_profile[:num_pts]
        timestamps = np.arange(1, num_pts + 1, dtype=np.float32) * self.dt

        # Control actuation: immediate acceleration & curvature
        target_accel = float(pred_accel[0])
        target_curv_iso = float(pred_curv[0])
        desired_cruise_speed = observation.target_speed_mps or 8.0

        if current_speed < 1.0:
            # Standstill launch: apply initial departure speed unless braking requested
            if target_accel > -1.0:
                effective_speed = min(3.0, desired_cruise_speed)
            else:
                effective_speed = 0.0
        else:
            effective_speed = float(np.clip(current_speed + target_accel * 1.5, 0.0, desired_cruise_speed))

        target_speeds_profile = np.full(num_pts, effective_speed, dtype=np.float32)

        # In CARLA: positive steer is right turn (negative ISO curvature)
        target_steer_angle = -math.atan(target_curv_iso * self.wheelbase)

        latency = time.perf_counter() - t0

        return E2EPrediction(
            waypoints=waypoints_carla.astype(np.float32),
            frame=CoordinateFrame.CARLA_LHD,
            timestamps=timestamps,
            target_speeds=target_speeds_profile,
            direct_control={
                "target_speed_mps": effective_speed,
                "target_acceleration": target_accel,
                "target_curvature": -target_curv_iso,
                "steering_tire_angle": target_steer_angle,
            },
            reasoning=(
                f"AutoE2E Real PyTorch Forward: {latency*1000:.1f}ms latency | "
                f"cmd={observation.high_level_command.value} | "
                f"accel={target_accel:.2f} m/s^2 | "
                f"kappa={target_curv_iso:.4f} 1/m | "
                f"device={self.device}"
            ),
            perception_outputs={
                "real_weights_executed": True,
                "backend": "pytorch_native",
                "device": str(self.device),
                "model_path": self.model_path,
            },
            inference_seconds=latency,
            confidence=0.99,
            model_version=f"{self.model_id} (REAL PyTorch)",
        )

    def _find_camera_image(self, images: dict[str, np.ndarray], slot_name: str) -> np.ndarray:
        """Find matching camera image from observation dictionary by slot name without fuzzy scrambling."""
        slot_clean = slot_name.upper().replace("-", "_").strip()

        # 1. Exact match
        for key, val in images.items():
            if slot_clean == key.upper().replace("-", "_").strip():
                return val

        # 2. Stripped prefix match (e.g. CAM_ or SENSOR_)
        slot_bare = slot_clean.replace("CAM_", "").replace("SENSOR_", "")
        for key, val in images.items():
            k_bare = key.upper().replace("-", "_").replace("CAM_", "").replace("SENSOR_", "").strip()
            if slot_bare == k_bare:
                return val

        # 3. Canonical aliases
        aliases = {
            "CAM_FRONT": ["FRONT", "FORWARD", "CAM_FORWARD", "FRONT_WIDE"],
            "CAM_FRONT_LEFT": ["FRONT_LEFT", "FL", "CAM_FL", "FORWARD_LEFT"],
            "CAM_FRONT_RIGHT": ["FRONT_RIGHT", "FR", "CAM_FR", "FORWARD_RIGHT"],
            "CAM_BACK_LEFT": ["BACK_LEFT", "REAR_LEFT", "RL", "CAM_RL"],
            "CAM_BACK": ["BACK", "REAR", "CAM_REAR", "REAR_TELE"],
            "CAM_BACK_RIGHT": ["BACK_RIGHT", "REAR_RIGHT", "RR", "CAM_RR"],
        }
        candidates = aliases.get(slot_clean, [])
        for key, val in images.items():
            k_clean = key.upper().replace("-", "_").strip()
            if k_clean in candidates:
                return val

        # 4. Fallback: first available or zero canvas
        for val in images.values():
            return val
        return np.zeros((450, 800, 3), dtype=np.uint8)

    def _infer_simulated(self, observation: E2EObservation, t0: float) -> E2EPrediction:
        """Fallback simulated trajectory synthesis using OpenDRIVE lane geometry."""
        current_speed = observation.ego_state.speed_mps
        target_speed = observation.target_speed_mps or 9.0
        cmd = observation.high_level_command

        waypoints, timestamps, target_speeds, curvature, effective_speed = plan_trajectory_with_lane_geometry(
            observation=observation,
            num_points=self.num_points,
            dt=self.dt,
            target_speed=target_speed,
            is_iso_flu=False,
            lane_change_length_m=22.0,
        )
        speed_err = effective_speed - current_speed
        target_accel = float(np.clip(0.8 * speed_err, -2.5, 1.5))
        target_steer_angle = math.atan(curvature * self.wheelbase)

        latency = time.perf_counter() - t0

        return E2EPrediction(
            waypoints=waypoints,
            frame=CoordinateFrame.CARLA_LHD,
            timestamps=timestamps,
            target_speeds=target_speeds,
            direct_control={
                "target_speed_mps": effective_speed,
                "target_acceleration": target_accel,
                "target_curvature": curvature,
                "steering_tire_angle": target_steer_angle,
            },
            reasoning=f"[SIMULATED BACKEND - NO NEURAL NETWORK LOADED] AutoE2E Latent BEV: cmd={cmd.value}, a={target_accel:.2f} m/s^2, kappa={curvature:.4f}",
            perception_outputs={
                "backend": "simulated",
                "real_weights_executed": False,
                "bev_feature_grid_shape": [128, 128, 64],
            },
            inference_seconds=latency,
            confidence=0.98,
            model_version="AutoE2E-80M (SIMULATED BACKEND)",
        )

    def _infer_onnx(self, observation: E2EObservation, t0: float) -> E2EPrediction:
        """Run real ONNX runtime session forward pass."""
        images = []
        for name in self.canonical_camera_slots:
            img = self._find_camera_image(observation.camera_images, name).astype(np.float32) / 255.0
            images.append(np.transpose(img, (2, 0, 1)))

        images_tensor = np.expand_dims(np.stack(images, axis=0), axis=0)  # [1, 6, 3, H, W]

        ego_tensor = np.array([[
            observation.ego_state.vx,
            observation.ego_state.vy,
            observation.ego_state.vz,
            observation.ego_state.wx,
            observation.ego_state.wy,
            observation.ego_state.wz,
        ]], dtype=np.float32)

        input_feed = {
            self.onnx_session.get_inputs()[0].name: images_tensor,
            self.onnx_session.get_inputs()[1].name: ego_tensor,
        }
        outputs = self.onnx_session.run(None, input_feed)
        waypoints = outputs[0][0]
        if waypoints.shape[1] == 2:
            waypoints = np.column_stack([waypoints, np.zeros(len(waypoints), dtype=np.float32)])

        timestamps = np.arange(1, self.num_points + 1, dtype=np.float32) * self.dt
        latency = time.perf_counter() - t0

        return E2EPrediction(
            waypoints=waypoints.astype(np.float32),
            frame=CoordinateFrame.CARLA_LHD,
            timestamps=timestamps,
            target_speeds=np.full(self.num_points, observation.target_speed_mps or 9.0, dtype=np.float32),
            direct_control={
                "target_speed_mps": observation.target_speed_mps or 9.0,
                "target_acceleration": float(outputs[1][0][0]) if len(outputs) > 1 else 0.5,
                "target_curvature": float(2.0 * waypoints[min(5, len(waypoints)-1), 1] / max(waypoints[min(5, len(waypoints)-1), 0]**2, 1e-2)),
                "steering_tire_angle": float(outputs[1][0][1]) if len(outputs) > 1 and len(outputs[1][0]) > 1 else 0.0,
            },
            reasoning=f"AutoE2E Real ONNX Inference: {latency*1000:.1f}ms latency, cmd={observation.high_level_command.value}",
            perception_outputs={"real_weights_executed": True, "backend": "onnx"},
            inference_seconds=latency,
            confidence=0.99,
            model_version=f"{self.model_id} (ONNX)",
        )
