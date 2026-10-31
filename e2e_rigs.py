"""Standard sensor rig specifications and CARLA actor spawner for E2E models.

Defines calibrated rigs for:
- AutoE2E & VAD (nuScenes 6-surround camera rig)
- Alpamayo 2 Super (PhysicalAI-AV 7-camera ring)
- QwenDrive 1.0 (Multi-modal surround + telephoto context rig)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from e2e_schema import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraSpec,
    CoordinateFrame,
    ModelType,
)


# ==============================================================================
# RIG SPECIFICATIONS
# ==============================================================================

# nuScenes 6-camera rig (Standard for VAD and AutoE2E)
# Values relative to vehicle center in CARLA frame (+X fwd, +Y right, +Z up)
NUSCENES_CAMERAS = [
    # (name, fov_deg, (x, y, z, roll, pitch, yaw))
    ("CAM_FRONT",       70.0,  ( 0.80,  0.00, 1.60, 0.0, 0.0,    0.0)),
    ("CAM_BACK",       110.0,  (-2.00,  0.00, 1.60, 0.0, 0.0,  180.0)),
    ("CAM_FRONT_LEFT",  70.0,  ( 0.27, -0.55, 1.60, 0.0, 0.0,  -55.0)),
    ("CAM_BACK_LEFT",   70.0,  (-0.32, -0.55, 1.60, 0.0, 0.0, -110.0)),
    ("CAM_FRONT_RIGHT", 70.0,  ( 0.27,  0.55, 1.60, 0.0, 0.0,   55.0)),
    ("CAM_BACK_RIGHT",  70.0,  (-0.32,  0.55, 1.60, 0.0, 0.0,  110.0)),
]

# PhysicalAI-AV 7-camera rig (NVIDIA Alpamayo 2 Super)
ALPAMAYO_CAMERAS = [
    ("front_cross_left",   120.0, ( 1.52, -0.50, 1.60, 0.0, 0.0,  -55.0)),
    ("front_wide",         120.0, ( 1.65,  0.00, 1.60, 0.0, 0.0,    0.0)),
    ("front_cross_right",  120.0, ( 1.52,  0.50, 1.60, 0.0, 0.0,   55.0)),
    ("rear_left",           70.0, (-0.35, -0.55, 1.55, 0.0, 0.0, -135.0)),
    ("rear_tele",           30.0, (-1.80,  0.00, 1.55, 0.0, 0.0,  180.0)),
    ("rear_right",          70.0, (-0.35,  0.55, 1.55, 0.0, 0.0,  135.0)),
    ("front_tele",          30.0, ( 1.65,  0.00, 1.65, 0.0, 0.0,    0.0)),
]

# QwenDrive 1.0 Foundation Model Multi-Modal Rig
QWENDRIVE_CAMERAS = [
    ("CAM_FRONT_WIDE",     100.0, ( 1.60,  0.00, 1.60, 0.0, 0.0,    0.0)),
    ("CAM_FRONT_TELE",      35.0, ( 1.60,  0.00, 1.70, 0.0, 0.0,    0.0)),
    ("CAM_FRONT_LEFT",      80.0, ( 0.30, -0.60, 1.60, 0.0, 0.0,  -60.0)),
    ("CAM_FRONT_RIGHT",     80.0, ( 0.30,  0.60, 1.60, 0.0, 0.0,   60.0)),
    ("CAM_BACK_LEFT",       80.0, (-0.30, -0.60, 1.60, 0.0, 0.0, -120.0)),
    ("CAM_BACK_RIGHT",      80.0, (-0.30,  0.60, 1.60, 0.0, 0.0,  120.0)),
    ("CAM_BACK",           110.0, (-1.80,  0.00, 1.60, 0.0, 0.0,  180.0)),
]


def build_rig_specs(
    model_type: ModelType,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, CameraSpec]:
    """Build CameraSpec map for a given model family with resolution defaults."""
    if model_type in (ModelType.AUTO_E2E, ModelType.VAD):
        w = image_width or 800
        h = image_height or 450
        camera_defs = NUSCENES_CAMERAS
    elif model_type == ModelType.ALPAMAYO:
        w = image_width or 640
        h = image_height or 360
        camera_defs = ALPAMAYO_CAMERAS
    elif model_type == ModelType.QWEN_DRIVE:
        w = image_width or 960
        h = image_height or 540
        camera_defs = QWENDRIVE_CAMERAS
    else:  # MOCK or CUSTOM
        w = image_width or 640
        h = image_height or 360
        camera_defs = NUSCENES_CAMERAS[:3]  # Front + sides for mock

    specs: dict[str, CameraSpec] = {}
    for name, fov, (x, y, z, roll, pitch, yaw) in camera_defs:
        intrinsics = CameraIntrinsics.from_fov(w, h, fov)
        extrinsics = CameraExtrinsics(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            frame=CoordinateFrame.CARLA_LHD,
        )
        specs[name] = CameraSpec(
            name=name,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            role="e2e_input",
        )
    return specs


class CarlaE2ESensorRig:
    """Manages spawning, streaming, and cleanup of camera sensors in CARLA."""

    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        specs: dict[str, CameraSpec],
        sensor_tick_seconds: float = 0.05,
        enable_ros2: bool = False,
        enable_spectator_camera: bool = False,
        spectator_view_mode: str = "chase",
    ) -> None:
        self.world = world
        self.ego = ego_vehicle
        self.specs = specs
        self.sensor_tick = sensor_tick_seconds
        self.enable_ros2 = enable_ros2
        self.enable_spectator_camera = enable_spectator_camera
        self.spectator_view_mode = spectator_view_mode
        self.sensors: dict[str, Any] = {}
        self.spectator_sensor: Any = None
        self.latest_frames: dict[str, tuple[int, np.ndarray]] = {}
        self.latest_spectator_frame: tuple[int, np.ndarray] | None = None
        self._spawn_sensors()

    def _spawn_sensors(self) -> None:
        try:
            import carla
        except ImportError:
            carla = None

        bp_lib = self.world.get_blueprint_library()
        for name, spec in self.specs.items():
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(spec.intrinsics.width))
            bp.set_attribute("image_size_y", str(spec.intrinsics.height))
            bp.set_attribute("fov", str(spec.intrinsics.fov))
            bp.set_attribute("sensor_tick", str(self.sensor_tick))
            bp.set_attribute("role_name", name)

            if bp.has_attribute("ros_name"):
                bp.set_attribute("ros_name", name)
            if self.enable_ros2 and bp.has_attribute("ros_topic_name"):
                bp.set_attribute("ros_topic_name", f"/sensing/camera/{name}/image_raw")

            ext = spec.extrinsics
            if carla is not None:
                tf = carla.Transform(
                    carla.Location(x=ext.x, y=ext.y, z=ext.z),
                    carla.Rotation(roll=ext.roll, pitch=ext.pitch, yaw=ext.yaw),
                )
            else:
                tf = None
            sensor = self.world.spawn_actor(bp, tf, attach_to=self.ego)
            if self.enable_ros2 and hasattr(sensor, "enable_for_ros"):
                sensor.enable_for_ros()

            # Python queue listener for synchronous capture
            def _make_callback(cam_name: str):
                def _callback(image):
                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = array.reshape((image.height, image.width, 4))
                    # Convert BGRA to RGB
                    rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
                    self.latest_frames[cam_name] = (image.frame, rgb)
                return _callback

            sensor.listen(_make_callback(name))
            self.sensors[name] = sensor

        # Optional High-Definition Spectator Chase Camera for Artifact Video Recording
        if self.enable_spectator_camera and carla is not None:
            bp_spec = bp_lib.find("sensor.camera.rgb")
            bp_spec.set_attribute("image_size_x", "1280")
            bp_spec.set_attribute("image_size_y", "720")
            bp_spec.set_attribute("fov", "90")
            bp_spec.set_attribute("sensor_tick", str(self.sensor_tick))
            bp_spec.set_attribute("role_name", "spectator_chase")

            if self.spectator_view_mode == "topdown":
                tf_spec = carla.Transform(
                    carla.Location(x=-2.0, y=0.0, z=18.0),
                    carla.Rotation(pitch=-75.0, yaw=0.0, roll=0.0),
                )
            else:
                tf_spec = carla.Transform(
                    carla.Location(x=-8.5, y=0.0, z=4.2),
                    carla.Rotation(pitch=-16.0, yaw=0.0, roll=0.0),
                )

            sensor_spec = self.world.spawn_actor(bp_spec, tf_spec, attach_to=self.ego)

            def _callback_spec(image):
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
                self.latest_spectator_frame = (image.frame, rgb)

            sensor_spec.listen(_callback_spec)
            self.spectator_sensor = sensor_spec

    def get_latest_images(self) -> dict[str, np.ndarray]:
        """Fetch the latest available RGB images for all sensors."""
        images: dict[str, np.ndarray] = {}
        for name in self.specs:
            if name in self.latest_frames:
                images[name] = self.latest_frames[name][1]
        return images

    def get_spectator_image(self) -> np.ndarray | None:
        """Fetch latest rendered frame from the spectator chase camera."""
        if self.latest_spectator_frame is not None:
            return self.latest_spectator_frame[1]
        return None

    def destroy(self) -> None:
        """Clean up all spawned sensors."""
        for name, sensor in list(self.sensors.items()):
            try:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        self.sensors.clear()
        self.latest_frames.clear()

        if self.spectator_sensor is not None:
            try:
                if self.spectator_sensor.is_listening:
                    self.spectator_sensor.stop()
                self.spectator_sensor.destroy()
            except Exception:
                pass
            self.spectator_sensor = None
        self.latest_spectator_frame = None
