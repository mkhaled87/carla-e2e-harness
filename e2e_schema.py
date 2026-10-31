"""Canonical data schema and types for End-to-End (E2E) Autonomous Driving models in CARLA.

Provides unified abstractions across models including:
- QwenDrive 1.0 (Vision-Language-Action, Flow Matching trajectory, CoT reasoning)
- AutoE2E (Latent BEV reasoning, GRU trajectory planning, ROS 2 control)
- Alpamayo 2 Super (PhysicalAI-AV 7-camera VLA, Diffusion trajectory, CoC reasoning)
- VAD (Vectorized Autonomous Driving, multi-task perception & planning)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence
import numpy as np


class CoordinateFrame(str, Enum):
    """Supported coordinate frame conventions."""
    CARLA_LHD = "carla_lhd"        # Left-handed: +X forward, +Y right, +Z up, yaw turns right
    ISO_8855_FLU = "iso_8855_flu"  # Right-handed: +X forward, +Y left, +Z up, yaw turns left
    SENSOR_OPTICAL = "optical"     # Camera optical: +X right, +Y down, +Z forward


class HighLevelCommand(str, Enum):
    """High-level route / navigation intent."""
    LANE_FOLLOW = "lane_follow"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STRAIGHT = "straight"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    STOP = "stop"


class ModelType(str, Enum):
    """Supported E2E model families."""
    QWEN_DRIVE = "qwendrive_1.0"
    AUTO_E2E = "autoe2e"
    ALPAMAYO = "alpamayo_2_super"
    VAD = "vad"
    MOCK = "mock_e2e"
    CUSTOM = "custom"


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fov: float
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "rectilinear"
    distortion_coeffs: tuple[float, ...] = ()

    @classmethod
    def from_fov(cls, width: int, height: int, fov: float) -> CameraIntrinsics:
        """Derive pinhole camera intrinsics from width, height, and horizontal FOV."""
        fov_rad = math.radians(fov)
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx  # Square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(
            width=width,
            height=height,
            fov=fov,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera mounting pose relative to vehicle origin (base_link or vehicle center)."""
    x: float
    y: float
    z: float
    roll: float = 0.0   # degrees
    pitch: float = 0.0  # degrees
    yaw: float = 0.0    # degrees
    frame: CoordinateFrame = CoordinateFrame.CARLA_LHD


@dataclass(frozen=True)
class CameraSpec:
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    role: str = "perception"


@dataclass(frozen=True)
class EgoKinematicState:
    timestamp_us: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float
    speed_mps: float
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    wx: float = 0.0  # deg/s or rad/s
    wy: float = 0.0
    wz: float = 0.0
    frame: CoordinateFrame = CoordinateFrame.CARLA_LHD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryWaypoint:
    """A predicted or target future waypoint in ego-centric coordinates."""
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    speed_mps: float = 0.0
    timestamp_rel_s: float = 0.0
    acceleration: float = 0.0
    curvature: float = 0.0
    frame: CoordinateFrame = CoordinateFrame.CARLA_LHD

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw, self.speed_mps, self.timestamp_rel_s], dtype=np.float32)


@dataclass
class E2EObservation:
    """Unified observation bundle supplied to an E2E model at an inference step."""
    model_type: ModelType
    frame_id: int
    timestamp_us: int
    camera_images: dict[str, np.ndarray]  # name -> RGB array (H, W, 3) uint8
    camera_specs: dict[str, CameraSpec]
    ego_state: EgoKinematicState
    ego_history: list[EgoKinematicState] = field(default_factory=list)
    high_level_command: HighLevelCommand = HighLevelCommand.LANE_FOLLOW
    text_prompt: str = ""
    target_speed_mps: float | None = None
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate shapes and required fields."""
        if not self.camera_images:
            raise ValueError("E2EObservation must contain at least one camera image")
        for name, img in self.camera_images.items():
            if not isinstance(img, np.ndarray):
                raise TypeError(f"Camera image {name} must be a numpy ndarray")
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Camera image {name} must have shape (H, W, 3), got {img.shape}")
            if name not in self.camera_specs:
                raise KeyError(f"Camera {name} image provided but missing CameraSpec")


@dataclass
class E2EPrediction:
    """Unified output from an E2E autonomous driving model."""
    waypoints: np.ndarray  # Shape (N, 2) [x, y], (N, 3) [x, y, z], or (N, >=3)
    frame: CoordinateFrame = CoordinateFrame.CARLA_LHD
    timestamps: np.ndarray | None = None  # Shape (N,) in seconds relative to current time
    target_speeds: np.ndarray | None = None  # Shape (N,) in m/s
    direct_control: dict[str, float] | None = None  # Optional throttle, steer, brake, accel, curvature
    reasoning: str = ""  # Natural language Chain-of-Thought / explanation
    perception_outputs: dict[str, Any] = field(default_factory=dict)  # Bounding boxes, occupancy, etc.
    inference_seconds: float = 0.0
    confidence: float = 1.0
    model_version: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.waypoints, np.ndarray):
            self.waypoints = np.asarray(self.waypoints, dtype=np.float32)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] < 2:
            raise ValueError(f"Waypoints must be shape (N, >=2), got {self.waypoints.shape}")
        if self.timestamps is None:
            # Default 10 Hz waypoints (0.1s step)
            n_points = self.waypoints.shape[0]
            self.timestamps = (np.arange(1, n_points + 1, dtype=np.float32) * 0.1)

    @property
    def num_points(self) -> int:
        return self.waypoints.shape[0]

    def get_xy(self) -> np.ndarray:
        return self.waypoints[:, :2]

