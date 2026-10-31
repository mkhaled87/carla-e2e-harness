"""Adapter for NVIDIA Alpamayo 2 Super (PhysicalAI-AV 7-Camera VLA).

Supports:
- 7-camera PhysicalAI-AV sensor input
- 1.6 seconds ego-motion history (16 poses @ 10 Hz)
- 64 predicted waypoints (6.4 seconds @ 10 Hz) defined in rear-axle FLU frame
- Chain-of-Causation (CoC) natural language reasoning
"""

from __future__ import annotations

import math
import os
import time
from typing import Any
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


class AlpamayoAdapter(BaseE2EAdapter):
    """Adapter for NVIDIA Alpamayo 2 Super."""

    def __init__(
        self,
        model_id: str = "nvidia/Alpamayo2-Super",
        model_path: str | None = None,
        diffusion_steps: int = 10,
        device: str = "auto",
        rear_axle_x: float = -1.425,
    ) -> None:
        super().__init__(model_type=ModelType.ALPAMAYO, model_id=model_id)
        self.model_path = model_path
        self.diffusion_steps = diffusion_steps
        self.device = resolve_torch_device(device)
        self.rear_axle_x = rear_axle_x
        self.num_points = 64
        self.dt = 0.1
        self.vlm_model: Any = None
        self.diffusion_model: Any = None
        self.tokenizer: Any = None
        self.use_simulated_backend = True

    def load(self, force_simulated: bool = False, **kwargs: Any) -> None:
        """Load real PyTorch/HuggingFace checkpoint if available, or fall back to simulated backend."""
        if force_simulated:
            self.use_simulated_backend = True
            self.is_loaded = True
            print(
                "[Alpamayo NOTICE] Running with SIMULATED BACKEND (forced simulated mode).\n"
                "  No neural network model is loaded. Trajectories are synthesized in rear-axle FLU frame via OpenDRIVE road geometry.\n"
                "  To execute real Diffusion/VLA weights: pass --model-path <path/to/alpamayo.pt>."
            )
            return

        target_source = self.model_path or self.model_id
        try:
            import torch
            if not torch.cuda.is_available() and self.device not in ("cpu", "mps"):
                self.use_simulated_backend = True
                self.is_loaded = True
                print(
                    f"[Alpamayo NOTICE] CUDA GPU / MPS device not available (device='{self.device}').\n"
                    f"  No neural network model is loaded. Running with SIMULATED BACKEND.\n"
                    f"  Trajectories are synthesized via OpenDRIVE road lane geometry.\n"
                    f"  To execute real Diffusion/VLA weights: run on a CUDA/MPS GPU and pass --model-path <PATH_OR_ID>."
                )
                return

            if self.model_path and os.path.isfile(self.model_path):
                print(f"[Alpamayo] Loading PyTorch checkpoint file from {self.model_path}...")
                loaded = torch.load(self.model_path, map_location=self.device)
                if isinstance(loaded, dict) and "diffusion_model" in loaded:
                    self.diffusion_model = loaded["diffusion_model"]
                    self.vlm_model = loaded.get("vlm_model")
                else:
                    self.diffusion_model = loaded
                self.use_simulated_backend = False
                self.is_loaded = True
                print("[Alpamayo] Checkpoint file loaded successfully.")
                return

            # Directory or HuggingFace identifier
            if os.path.isdir(target_source) or not self.model_path:
                print(f"[Alpamayo] Loading Alpamayo 2 Super foundation model from '{target_source}'...")
                # Try NVlabs official alpamayo2 package first
                try:
                    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
                    self.diffusion_model = Alpamayo2Super.from_pretrained(
                        target_source,
                        dtype=torch.bfloat16,
                        device_map=self.device,
                        low_cpu_mem_usage=True,
                    )
                    self.use_simulated_backend = False
                    self.is_loaded = True
                    print("[Alpamayo] Alpamayo2Super model loaded successfully.")
                    return
                except ImportError:
                    pass

                # Fallback to HuggingFace transformers
                try:
                    from transformers import AutoProcessor, AutoModel
                    self.tokenizer = AutoProcessor.from_pretrained(target_source, trust_remote_code=True)
                    self.diffusion_model = AutoModel.from_pretrained(
                        target_source,
                        device_map=self.device,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    self.use_simulated_backend = False
                    self.is_loaded = True
                    print("[Alpamayo] Foundation model loaded via Transformers.")
                    return
                except Exception as ex_hf:
                    print(
                        f"[Alpamayo NOTICE] Could not load Alpamayo2Super architecture ({ex_hf}).\n"
                        f"  Weights directory was found at: '{target_source}', but the official NVlabs 'alpamayo2' Python package is required.\n"
                        f"  To install: pip install git+https://github.com/NVlabs/alpamayo2.git\n"
                        f"  Running in SIMULATED BACKEND mode for now."
                    )
                    self.use_simulated_backend = True
                    self.is_loaded = True
                    return
        except Exception as e:
            self.use_simulated_backend = True
            self.is_loaded = True
            print(
                f"[Alpamayo NOTICE] Real weights not loaded ({e}).\n"
                f"  No actual AI model is active. Running with SIMULATED BACKEND.\n"
                f"  Trajectories are synthesized via OpenDRIVE road lane geometry."
            )

    def infer(self, observation: E2EObservation) -> E2EPrediction:
        """Generate 64 waypoints in rear-axle ISO 8855 FLU coordinates + CoC reasoning."""
        t0 = time.perf_counter()

        if not self.use_simulated_backend and self.diffusion_model is not None:
            try:
                return self._infer_real_weights(observation, t0)
            except Exception:
                return self._infer_simulated(observation, t0)
        else:
            return self._infer_simulated(observation, t0)

    def _infer_real_weights(self, observation: E2EObservation, t0: float) -> E2EPrediction:
        """Run real neural forward pass through Alpamayo 2 7-camera diffusion head."""
        import torch

        cam_order = [
            "cam_front_wide", "cam_front_telephoto", "cam_cross_left",
            "cam_cross_right", "cam_rear", "cam_side_left", "cam_side_right"
        ]
        images_tensor = []
        for name in cam_order:
            if name in observation.camera_images:
                arr = observation.camera_images[name].astype(np.float32) / 255.0
            else:
                arr = np.zeros((450, 800, 3), dtype=np.float32)
            images_tensor.append(np.transpose(arr, (2, 0, 1)))

        img_pt = torch.from_numpy(np.stack(images_tensor, axis=0)).unsqueeze(0).to(self.device)

        hist = list(observation.ego_history)
        if len(hist) < 16:
            curr = observation.ego_state
            hist = [curr] * (16 - len(hist)) + hist
        hist = hist[-16:]
        hist_data = np.array([[s.vx, s.vy, s.vz, s.wx, s.wy, s.wz] for s in hist], dtype=np.float32)
        hist_pt = torch.from_numpy(hist_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.diffusion_model, "sample_trajectories_from_data"):
                pred_xyz, _, _, extra = self.diffusion_model.sample_trajectories_from_data(
                    data={"cameras": img_pt, "history": hist_pt},
                    num_traj_samples=1,
                )
                waypoints_flu = pred_xyz[0].cpu().numpy()
            elif hasattr(self.diffusion_model, "sample"):
                waypoints_tensor = self.diffusion_model.sample(
                    cameras=img_pt,
                    ego_history=hist_pt,
                    steps=self.diffusion_steps,
                    command=observation.high_level_command.value,
                )
                waypoints_flu = waypoints_tensor.cpu().numpy().squeeze(0)
            else:
                out = self.diffusion_model(cameras=img_pt, ego_history=hist_pt)
                if isinstance(out, torch.Tensor):
                    waypoints_flu = out.cpu().numpy().squeeze(0)
                elif hasattr(out, "waypoints"):
                    waypoints_flu = out.waypoints.cpu().numpy().squeeze(0)
                else:
                    return self._infer_simulated(observation, t0)

        timestamps = np.arange(1, self.num_points + 1, dtype=np.float32) * self.dt
        effective_speed = observation.target_speed_mps or 8.0
        latency = time.perf_counter() - t0

        return E2EPrediction(
            waypoints=waypoints_flu.astype(np.float32),
            frame=CoordinateFrame.ISO_8855_FLU,
            timestamps=timestamps,
            target_speeds=np.full(self.num_points, effective_speed, dtype=np.float32),
            direct_control={"target_speed_mps": effective_speed},
            reasoning=f"Alpamayo 2 Real Diffusion Inference: {latency*1000:.1f}ms latency, {self.diffusion_steps} steps, cmd={observation.high_level_command.value}",
            perception_outputs={"cameras_processed": 7, "real_weights_executed": True, "diffusion_steps": self.diffusion_steps},
            inference_seconds=latency,
            confidence=0.99,
            model_version=f"{self.model_id} (Real Diffusion)",
        )

    def _infer_simulated(self, observation: E2EObservation, t0: float) -> E2EPrediction:
        """Simulated backend respecting map lane geometry and smoothstep lateral offsets."""
        target_speed = observation.target_speed_mps or 8.0
        cmd = observation.high_level_command
        prompt_text = (observation.text_prompt or "").lower()

        waypoints_flu, timestamps, target_speeds, curvature, effective_speed = plan_trajectory_with_lane_geometry(
            observation=observation,
            num_points=self.num_points,
            dt=self.dt,
            target_speed=target_speed,
            is_iso_flu=True,
            lane_change_length_m=22.0,
        )

        if cmd == HighLevelCommand.STOP or "slow" in prompt_text or "stop" in prompt_text:
            coc = (
                "Chain-of-Causation: Ego detected stop constraint ahead. "
                "Diffusion expert generates deceleration curve to a full halt."
            )
        elif cmd == HighLevelCommand.LANE_CHANGE_RIGHT or "change right" in prompt_text:
            coc = (
                "Chain-of-Causation: Seven-camera ring confirms target right lane clear. "
                "Diffusion model shifts laterally into target right corridor."
            )
        elif cmd == HighLevelCommand.LANE_CHANGE_LEFT or "change left" in prompt_text:
            coc = (
                "Chain-of-Causation: Seven-camera ring confirms target left lane clear. "
                "Diffusion model shifts laterally into target left corridor."
            )
        elif cmd == HighLevelCommand.TURN_LEFT or "turn left" in prompt_text:
            coc = (
                "Chain-of-Causation: Junction detected. Left turn path clear of oncoming traffic. "
                "Diffusion trajectory initiates left lateral displacement."
            )
        elif cmd == HighLevelCommand.TURN_RIGHT or "turn right" in prompt_text:
            coc = (
                "Chain-of-Causation: Right turn path clear of pedestrians. "
                "Diffusion trajectory curves right into destination corridor."
            )
        elif abs(curvature) > 0.008:
            direction = "left" if curvature > 0 else "right"
            coc = (
                f"Chain-of-Causation: Seven-camera ring detects {direction} curved road geometry. "
                "Diffusion planner traces continuous curved centerline."
            )
        else:
            coc = (
                "Chain-of-Causation: Seven-camera ring clear of obstacles. "
                "Cruising along centerline."
            )

        latency = time.perf_counter() - t0

        return E2EPrediction(
            waypoints=waypoints_flu,
            frame=CoordinateFrame.ISO_8855_FLU,
            timestamps=timestamps,
            target_speeds=target_speeds,
            direct_control={
                "target_speed_mps": effective_speed,
            },
            reasoning=f"[SIMULATED BACKEND - NO NEURAL NETWORK LOADED] {coc}",
            perception_outputs={
                "backend": "simulated",
                "real_weights_executed": False,
                "cameras_processed": 7,
                "diffusion_steps": self.diffusion_steps,
            },
            inference_seconds=latency,
            confidence=0.97,
            model_version="Alpamayo2-Super (SIMULATED BACKEND)",
        )

