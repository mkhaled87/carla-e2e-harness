"""Base interface for all End-to-End autonomous driving model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any
import numpy as np

from e2e_schema import (
    CoordinateFrame,
    E2EObservation,
    E2EPrediction,
    HighLevelCommand,
    ModelType,
)


class BaseE2EAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_type: ModelType, model_id: str | None = None) -> None:
        self.model_type = model_type
        self.model_id = model_id or model_type.value
        self.is_loaded = False

    @abstractmethod
    def load(self, **kwargs: Any) -> None:
        """Load model weights, construct neural networks, or connect to backend."""
        pass

    @abstractmethod
    def infer(self, observation: E2EObservation) -> E2EPrediction:
        """Execute model forward pass or trajectory generation."""
        pass

    def shutdown(self) -> None:
        """Release GPU memory and resources."""
        self.is_loaded = False


def resolve_torch_device(requested_device: str = "auto") -> str:
    """Resolve 'auto' device string to best available compute target (cuda, mps, cpu)."""
    if requested_device != "auto":
        return requested_device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def plan_trajectory_with_lane_geometry(
    observation: E2EObservation,
    num_points: int,
    dt: float,
    target_speed: float,
    is_iso_flu: bool = False,
    lane_change_length_m: float = 22.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthesizes high-fidelity future waypoints respecting map road curvature and lane shifts.

    Returns:
        waypoints: (num_points, 3) in target coordinate frame (CARLA LHD or ISO FLU)
        timestamps: (num_points,) in seconds
        target_speeds: (num_points,) in m/s
        target_curvature: scalar curvature estimate at immediate lookahead
    """
    current_speed = observation.ego_state.speed_mps
    cmd = observation.high_level_command
    prompt_text = (observation.text_prompt or "").lower()

    # Determine effective target speed
    if "slow" in prompt_text or "caution" in prompt_text or cmd == HighLevelCommand.STOP:
        effective_speed = 0.0
    elif cmd in (HighLevelCommand.TURN_LEFT, HighLevelCommand.TURN_RIGHT) or "turn" in prompt_text:
        effective_speed = min(target_speed, 5.0)
    else:
        effective_speed = target_speed

    # Determine lateral offset for lane change (+Y is right in CARLA LHD, +Y is left in ISO FLU)
    if "target_lane_offset" in observation.custom_metadata:
        lateral_offset = float(observation.custom_metadata["target_lane_offset"])
    elif cmd == HighLevelCommand.LANE_CHANGE_RIGHT or "lane change right" in prompt_text or "change right" in prompt_text:
        lateral_offset = 3.5  # shift right
    elif cmd == HighLevelCommand.LANE_CHANGE_LEFT or "lane change left" in prompt_text or "change left" in prompt_text:
        lateral_offset = -3.5 # shift left
    else:
        lateral_offset = 0.0

    if is_iso_flu:
        # In ISO FLU, +Y is left, -Y is right
        lateral_offset = -lateral_offset

    timestamps = np.arange(1, num_points + 1, dtype=np.float32) * dt
    waypoints = np.zeros((num_points, 3), dtype=np.float32)
    speeds = np.zeros(num_points, dtype=np.float32)
    cum_dist = np.zeros(num_points, dtype=np.float32)

    sp = current_speed
    for i in range(num_points):
        sp += (effective_speed - sp) * 0.15
        speeds[i] = sp
        cum_dist[i] = (cum_dist[i - 1] if i > 0 else 0.0) + sp * dt

    cl_data = observation.custom_metadata.get("local_lane_centerline")
    if cl_data is not None and len(cl_data) >= 2:
        cl = np.asarray(cl_data, dtype=np.float32)[:, :2]
        cl_full = np.vstack([[0.0, 0.0], cl])
        diffs = np.diff(cl_full, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        s_cl = np.concatenate([[0.0], np.cumsum(seg_lens)])
        tangents = diffs / np.maximum(seg_lens[:, None], 1e-4)

        for i, d in enumerate(cum_dist):
            if d <= s_cl[-1]:
                idx = int(np.searchsorted(s_cl, d))
                if idx == 0:
                    px, py = cl_full[0]
                    tx, ty = tangents[0]
                else:
                    ratio = (d - s_cl[idx - 1]) / max(s_cl[idx] - s_cl[idx - 1], 1e-4)
                    px = cl_full[idx - 1, 0] + ratio * (cl_full[idx, 0] - cl_full[idx - 1, 0])
                    py = cl_full[idx - 1, 1] + ratio * (cl_full[idx, 1] - cl_full[idx - 1, 1])
                    tx, ty = tangents[min(idx - 1, len(tangents) - 1)]
            else:
                extra = d - s_cl[-1]
                tx, ty = tangents[-1]
                px = cl_full[-1, 0] + extra * tx
                py = cl_full[-1, 1] + extra * ty

            # Right normal vector in CARLA LHD: (tx, ty) -> (-ty, tx)
            nx, ny = -ty, tx

            u = min(1.0, max(0.0, d / lane_change_length_m))
            s_u = 3.0 * (u ** 2) - 2.0 * (u ** 3)
            lat = (lateral_offset if not is_iso_flu else -lateral_offset) * s_u

            waypoints[i, 0] = px + nx * lat
            waypoints[i, 1] = py + ny * lat
            waypoints[i, 2] = 0.0

        if is_iso_flu:
            # Convert CARLA LHD (+Y right) to ISO FLU (+Y left)
            waypoints[:, 1] = -waypoints[:, 1]
    else:
        # Kinematic model fallback for tests or off-map execution
        steer_rate = 0.0
        if cmd == HighLevelCommand.TURN_LEFT or "turn left" in prompt_text:
            steer_rate = 0.12 if is_iso_flu else -0.12
        elif cmd == HighLevelCommand.TURN_RIGHT or "turn right" in prompt_text:
            steer_rate = -0.12 if is_iso_flu else 0.12

        x, y, yaw = 0.0, 0.0, 0.0
        for i, d in enumerate(cum_dist):
            yaw += steer_rate * dt
            x += speeds[i] * math.cos(yaw) * dt
            y += speeds[i] * math.sin(yaw) * dt

            u = min(1.0, max(0.0, d / lane_change_length_m))
            s_u = 3.0 * (u ** 2) - 2.0 * (u ** 3)
            lat = lateral_offset * s_u

            waypoints[i, 0] = x
            waypoints[i, 1] = y + lat
            waypoints[i, 2] = 0.0

    lookahead_idx = min(8, num_points - 1)
    d_la = math.hypot(waypoints[lookahead_idx, 0], waypoints[lookahead_idx, 1])
    target_curvature = float(2.0 * waypoints[lookahead_idx, 1] / max(d_la ** 2, 0.1))

    return waypoints, timestamps, speeds, target_curvature, effective_speed


