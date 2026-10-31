"""Closed-Loop Orchestrator and Driving Runner for E2E AI Models in CARLA.

Executes lockstep, deterministic closed-loop simulations:
1. Configures synchronous mode (fixed delta time).
2. Spawns/attaches ego vehicle and attaches the model-specific sensor rig.
3. Collects synchronized multi-camera frames and ego kinematics.
4. Queries local model adapter or remote model server (QwenDrive, AutoE2E, Alpamayo, VAD).
5. Tracks predicted trajectories using lateral Pure Pursuit and longitudinal PID.
6. Monitors driving metrics (speed, cross-track error, jerk, collisions, lane invasions).
7. Records JSONL telemetry for auditing and post-analysis.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any
import numpy as np

from adapters.base_adapter import BaseE2EAdapter
from adapters.mock_adapter import MockE2EAdapter
from adapters.qwendrive_adapter import QwenDriveAdapter
from adapters.autoe2e_adapter import AutoE2EAdapter
from adapters.alpamayo_adapter import AlpamayoAdapter
from adapters.vad_adapter import VADAdapter
from e2e_coordinates import normalize_trajectory_to_carla, world_to_local_2d
from e2e_display import E2EDashboardDisplay, NullDisplay
from e2e_model_server import pack_prediction, unpack_observation
from e2e_protocol import (
    connect_e2e_service,
    recv_protocol_message,
    send_protocol_message,
)
from e2e_rigs import CarlaE2ESensorRig, build_rig_specs
from e2e_schema import (
    CoordinateFrame,
    E2EObservation,
    E2EPrediction,
    EgoKinematicState,
    HighLevelCommand,
    ModelType,
)
from e2e_tracker import E2ETrajectoryTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_closed_loop")


class TelemetryLogger:
    """Line-buffered JSONL telemetry recorder."""

    def __init__(self, output_path: str | None) -> None:
        self.file = None
        if output_path:
            p = Path(output_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(p, "w", encoding="utf-8", buffering=1)
            logger.info(f"Recording telemetry to {p}")

    def log(self, event_type: str, data: dict[str, Any]) -> None:
        if self.file:
            entry = {"timestamp": time.time(), "event": event_type, **data}
            self.file.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None


class ClosedLoopE2ERunner:
    """Manages the full closed-loop simulation cycle."""

    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        model_type: ModelType,
        target_speed_mps: float = 8.0,
        fixed_delta_seconds: float = 0.05,
        replan_every_n_ticks: int = 2,
        use_remote_server: bool = False,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8090,
        telemetry_file: str | None = None,
        text_prompt: str = "Drive smoothly along the lane, respect traffic constraints and avoid obstacles.",
        enable_display: bool = True,
        spectator_follow: bool = True,
        spectator_view_mode: str = "chase",
        model_path: str | None = None,
        record_dir: str | Path | None = None,
        screenshot_path: str | Path | None = None,
        screenshot_tick: int = 5,
    ) -> None:
        self.world = world
        self.ego = ego_vehicle
        self.model_type = model_type
        self.model_path = model_path
        self.target_speed = target_speed_mps
        self.dt = fixed_delta_seconds
        self.replan_ticks = replan_every_n_ticks
        self.use_remote_server = use_remote_server
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.text_prompt = text_prompt
        self.enable_display = enable_display
        self.spectator_follow = spectator_follow
        self.spectator_view_mode = spectator_view_mode
        self.record_dir = Path(record_dir).expanduser().resolve() if record_dir else None
        self.screenshot_path = Path(screenshot_path).expanduser().resolve() if screenshot_path else None
        self.screenshot_tick = screenshot_tick
        self._screenshot_taken = False

        self.artifact_recorder: Any = None
        if self.record_dir:
            from e2e_artifact_recorder import E2EArtifactRecorder
            self.artifact_recorder = E2EArtifactRecorder(
                record_dir=self.record_dir,
                dt=self.dt,
                model_type=self.model_type.value,
                spectator_view_mode=self.spectator_view_mode,
            )
        self._last_collision_event: dict[str, Any] | None = None
        self._last_lane_invasion_event: dict[str, Any] | None = None
        self.start_wall_time = time.time()

        self.telemetry = TelemetryLogger(telemetry_file)
        self.tracker = E2ETrajectoryTracker()
        self.specs = build_rig_specs(model_type)
        self.rig: CarlaE2ESensorRig | None = None
        self.display: Any = None

        self.remote_sock: Any = None
        self.local_adapter: BaseE2EAdapter | None = None

        self.ego_history: deque[EgoKinematicState] = deque(maxlen=20)
        self.latest_prediction: E2EPrediction | None = None
        self.step_count = 0
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.collision_sensor: Any = None
        self.lane_sensor: Any = None

        self.lane_change_status = "idle"
        self.target_lane_id: int | None = None
        self.origin_lane_id: int | None = None
        self.lane_change_ticks = 0
        self.max_lane_change_ticks = 60

    @property
    def is_simulated_backend(self) -> bool:
        """Returns True if the underlying model adapter is running in simulated backend mode."""
        if self.local_adapter is not None:
            return bool(self.local_adapter.use_simulated_backend)
        return False

    def setup(self) -> None:
        """Initialize sensors, model adapters, and infaction monitors."""
        logger.info(f"Setting up E2E closed-loop runner for model: {self.model_type.value}")
        self.rig = CarlaE2ESensorRig(
            world=self.world,
            ego_vehicle=self.ego,
            specs=self.specs,
            sensor_tick_seconds=self.dt,
            enable_spectator_camera=bool(self.artifact_recorder is not None),
            spectator_view_mode=self.spectator_view_mode,
        )

        if self.use_remote_server:
            logger.info(f"Connecting to remote model server at {self.remote_host}:{self.remote_port}...")
            self.remote_sock = connect_e2e_service(self.remote_host, self.remote_port)
            send_protocol_message(self.remote_sock, "PING")
            pong = recv_protocol_message(self.remote_sock)
            logger.info(f"Connected to model server: {pong.metadata}")
        else:
            logger.info("Initializing local model adapter...")
            if self.model_type == ModelType.QWEN_DRIVE:
                self.local_adapter = QwenDriveAdapter(model_id=self.model_path or "Qwen/Qwen-Drive-1.0-4B")
            elif self.model_type == ModelType.AUTO_E2E:
                self.local_adapter = AutoE2EAdapter(model_path=self.model_path)
            elif self.model_type == ModelType.ALPAMAYO:
                self.local_adapter = AlpamayoAdapter(model_path=self.model_path)
            elif self.model_type == ModelType.VAD:
                self.local_adapter = VADAdapter(model_path=self.model_path)
            else:
                self.local_adapter = MockE2EAdapter(default_target_speed_mps=self.target_speed)
            self.local_adapter.load()
            if self.local_adapter.use_simulated_backend:
                logger.warning(
                    f"\n{'='*75}\n"
                    f" [NOTICE] RUNNING IN SIMULATED BACKEND MODE (NO NEURAL NETWORK ACTIVE)\n"
                    f" - Adapter: '{self.model_type.value}'\n"
                    f" - Model Path: {self.model_path or 'None (default)'}\n"
                    f" - Behavior: Trajectories are synthesized using CARLA OpenDRIVE map lane geometry.\n"
                    f" - To run real weights: specify --model-path <path_or_id> with PyTorch/ONNX.\n"
                    f"{'='*75}"
                )
            else:
                logger.info(
                    f"\n{'='*75}\n"
                    f" [NOTICE] RUNNING WITH REAL NEURAL NETWORK WEIGHTS\n"
                    f" - Adapter: '{self.model_type.value}'\n"
                    f" - Model Path: {self.model_path}\n"
                    f"{'='*75}"
                )

        # Infraction sensors
        try:
            import carla
        except ImportError:
            carla = None

        bp_lib = self.world.get_blueprint_library()
        col_bp = bp_lib.find("sensor.other.collision")
        tf = carla.Transform() if carla is not None else None
        self.collision_sensor = self.world.spawn_actor(col_bp, tf, attach_to=self.ego)
        if hasattr(self.collision_sensor, "listen"):
            self.collision_sensor.listen(self._on_collision)

        lane_bp = bp_lib.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_bp, tf, attach_to=self.ego)
        if hasattr(self.lane_sensor, "listen"):
            self.lane_sensor.listen(self._on_lane_invasion)

        # Pygame HUD Display
        if self.enable_display:
            try:
                self.display = E2EDashboardDisplay(
                    title=f"CARLA E2E AI Driving Cockpit — {self.model_type.value.upper()}"
                )
            except Exception as e:
                logger.warning(f"Could not open Pygame HUD display: {e}")
                self.display = NullDisplay()
        else:
            self.display = NullDisplay()

        # Warm up camera sensors in synchronous mode so cameras produce rendered frames and auto-exposure settles
        if self.world is not None:
            logger.info("Warming up camera sensors and auto-exposure (8 ticks)...")
            try:
                for _ in range(8):
                    self.world.tick()
                    time.sleep(0.01)
                logger.info("Camera sensors warmed up successfully.")
            except Exception as e:
                logger.warning(f"Sensor warmup notice: {e}")

    def _on_collision(self, event: Any) -> None:
        self.collision_count += 1
        logger.warning(f"COLLISION DETECTED with actor {event.other_actor.type_id}!")
        self.telemetry.log("collision", {"other_actor": event.other_actor.type_id})
        self._last_collision_event = {"other_actor": event.other_actor.type_id}

    def _on_lane_invasion(self, event: Any) -> None:
        self.lane_invasion_count += 1
        crossed_lanes = [str(x.type) for x in event.crossed_lane_markings]
        self.telemetry.log("lane_invasion", {"crossed_lanes": crossed_lanes})
        self._last_lane_invasion_event = {"crossed_lanes": crossed_lanes}

    def run_step(self, high_level_command: HighLevelCommand = HighLevelCommand.LANE_FOLLOW) -> dict[str, Any]:
        """Execute one simulation and control step."""
        try:
            import carla
        except ImportError:
            carla = None

        # 1. Advance simulation step
        self.world.tick()
        self.step_count += 1

        # 2. Extract current ego state
        tf = self.ego.get_transform()
        vel = self.ego.get_velocity()
        acc = self.ego.get_acceleration()
        ang_vel = self.ego.get_angular_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        ego_state = EgoKinematicState(
            timestamp_us=int(time.time() * 1e6),
            x=tf.location.x,
            y=tf.location.y,
            z=tf.location.z,
            roll=tf.rotation.roll,
            pitch=tf.rotation.pitch,
            yaw=tf.rotation.yaw,
            vx=vel.x,
            vy=vel.y,
            vz=vel.z,
            speed_mps=speed,
            ax=acc.x,
            ay=acc.y,
            az=acc.z,
            wx=ang_vel.x,
            wy=ang_vel.y,
            wz=ang_vel.z,
            frame=CoordinateFrame.CARLA_LHD,
        )
        self.ego_history.append(ego_state)

        # 3. Model Inference (every replan_ticks)
        if self.step_count % self.replan_ticks == 0 or self.latest_prediction is None:
            # Query local lane centerline from CARLA map ahead of ego vehicle
            local_lane_centerline = None
            target_lane_offset = 0.0
            if carla is not None and self.world is not None:
                try:
                    carla_map = self.world.get_map()
                    ego_tf = self.ego.get_transform()
                    current_wp = carla_map.get_waypoint(
                        ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    if current_wp is not None:
                        # Manage lane change state machine
                        if high_level_command == HighLevelCommand.LANE_CHANGE_RIGHT:
                            if self.lane_change_status != "in_progress":
                                r_lane = current_wp.get_right_lane()
                                if r_lane is not None and r_lane.lane_type == carla.LaneType.Driving and r_lane.lane_id != current_wp.lane_id:
                                    self.origin_lane_id = current_wp.lane_id
                                    self.target_lane_id = r_lane.lane_id
                                else:
                                    self.origin_lane_id = current_wp.lane_id
                                    self.target_lane_id = None
                                self.lane_change_status = "in_progress"
                                self.lane_change_ticks = 0

                            if self.target_lane_id is not None and current_wp.lane_id == self.target_lane_id:
                                self.lane_change_status = "completed"
                                target_lane_offset = 0.0
                            elif self.lane_change_ticks >= self.max_lane_change_ticks:
                                self.lane_change_status = "completed"
                                target_lane_offset = 0.0
                            else:
                                target_lane_offset = 3.5
                                self.lane_change_ticks += 1

                        elif high_level_command == HighLevelCommand.LANE_CHANGE_LEFT:
                            if self.lane_change_status != "in_progress":
                                l_lane = current_wp.get_left_lane()
                                if l_lane is not None and l_lane.lane_type == carla.LaneType.Driving and l_lane.lane_id != current_wp.lane_id:
                                    self.origin_lane_id = current_wp.lane_id
                                    self.target_lane_id = l_lane.lane_id
                                else:
                                    self.origin_lane_id = current_wp.lane_id
                                    self.target_lane_id = None
                                self.lane_change_status = "in_progress"
                                self.lane_change_ticks = 0

                            if self.target_lane_id is not None and current_wp.lane_id == self.target_lane_id:
                                self.lane_change_status = "completed"
                                target_lane_offset = 0.0
                            elif self.lane_change_ticks >= self.max_lane_change_ticks:
                                self.lane_change_status = "completed"
                                target_lane_offset = 0.0
                            else:
                                target_lane_offset = -3.5
                                self.lane_change_ticks += 1
                        else:
                            self.lane_change_status = "idle"
                            self.target_lane_id = None
                            self.origin_lane_id = None
                            self.lane_change_ticks = 0
                            target_lane_offset = 0.0

                        # Query ahead along current lane centerline for 50 meters
                        pts_world = []
                        wp = current_wp
                        step_m = 1.0
                        for _ in range(50):
                            next_wps = wp.next(step_m)
                            if not next_wps:
                                break
                            wp = next_wps[0]
                            pts_world.append([wp.transform.location.x, wp.transform.location.y, wp.transform.location.z])

                        if pts_world:
                            pts_local = world_to_local_2d(
                                np.array(pts_world, dtype=np.float32),
                                ego_tf.location.x,
                                ego_tf.location.y,
                                ego_tf.rotation.yaw,
                                is_carla_frame=True,
                            )
                            local_lane_centerline = pts_local.tolist()
                except Exception as e:
                    logger.debug(f"Could not extract local lane centerline from CARLA map: {e}")

            custom_metadata = {
                "local_lane_centerline": local_lane_centerline,
                "target_lane_offset": target_lane_offset,
                "lane_change_status": self.lane_change_status,
            }

            camera_images = self.rig.get_latest_images() if self.rig else {}
            # Fallback for dummy image if capture pending
            if not camera_images:
                for name, spec in self.specs.items():
                    camera_images[name] = np.zeros(
                        (spec.intrinsics.height, spec.intrinsics.width, 3), dtype=np.uint8
                    )

            observation = E2EObservation(
                model_type=self.model_type,
                frame_id=self.step_count,
                timestamp_us=ego_state.timestamp_us,
                camera_images=camera_images,
                camera_specs=self.specs,
                ego_state=ego_state,
                ego_history=list(self.ego_history),
                high_level_command=high_level_command,
                text_prompt=self.text_prompt,
                target_speed_mps=self.target_speed,
                custom_metadata=custom_metadata,
            )

            if self.use_remote_server and self.remote_sock:
                # Transmit over TCP
                msg_meta = {
                    "model_type": self.model_type.value,
                    "frame_id": self.step_count,
                    "timestamp_us": ego_state.timestamp_us,
                    "ego_state": ego_state.to_dict(),
                    "camera_specs": {
                        k: {"intrinsics": asdict(v.intrinsics), "extrinsics": asdict(v.extrinsics)}
                        for k, v in self.specs.items()
                    },
                    "high_level_command": high_level_command.value,
                    "text_prompt": self.text_prompt,
                    "target_speed_mps": self.target_speed,
                    "custom_metadata": custom_metadata,
                }
                msg_arrays = {f"cam_{k}": v for k, v in camera_images.items()}
                send_protocol_message(self.remote_sock, "INFER", metadata=msg_meta, arrays=msg_arrays)
                resp = recv_protocol_message(self.remote_sock)
                if resp.kind == "PREDICTION":
                    self.latest_prediction = E2EPrediction(
                        waypoints=resp.arrays["waypoints"],
                        frame=CoordinateFrame(resp.metadata.get("frame", CoordinateFrame.CARLA_LHD.value)),
                        timestamps=resp.arrays.get("timestamps"),
                        target_speeds=resp.arrays.get("target_speeds"),
                        direct_control=resp.metadata.get("direct_control"),
                        reasoning=resp.metadata.get("reasoning", ""),
                        perception_outputs=resp.metadata.get("perception_outputs", {}),
                        inference_seconds=resp.metadata.get("inference_seconds", 0.0),
                        confidence=resp.metadata.get("confidence", 1.0),
                        model_version=resp.metadata.get("model_version", "unknown"),
                    )
            elif self.local_adapter:
                self.latest_prediction = self.local_adapter.infer(observation)

        # 4. Trajectory Tracking Control
        if self.latest_prediction:
            cmd = self.tracker.compute_control(
                prediction=self.latest_prediction,
                current_speed_mps=speed,
                dt=self.dt,
            )
            if carla is not None:
                ctrl = carla.VehicleControl(
                    throttle=cmd.throttle,
                    steer=cmd.steer,
                    brake=cmd.brake,
                    hand_brake=False,
                    reverse=False,
                )
            else:
                from dataclasses import dataclass
                @dataclass
                class FallbackControl:
                    throttle: float
                    steer: float
                    brake: float
                    hand_brake: bool = False
                    reverse: bool = False
                ctrl = FallbackControl(throttle=cmd.throttle, steer=cmd.steer, brake=cmd.brake)
            self.ego.apply_control(ctrl)

            step_data = {
                "step": self.step_count,
                "speed_mps": speed,
                "target_speed_mps": cmd.target_speed_mps,
                "throttle": cmd.throttle,
                "steer": cmd.steer,
                "brake": cmd.brake,
                "cross_track_error": cmd.cross_track_error,
                "heading_error": cmd.heading_error_rad,
                "reasoning": self.latest_prediction.reasoning,
                "collisions": self.collision_count,
                "lane_invasions": self.lane_invasion_count,
            }
            self.telemetry.log("control_step", step_data)

            # Record simulation artifacts (spectator video frame, camera feeds, pose & sensor telemetry)
            if self.artifact_recorder is not None:
                spectator_img = self.rig.get_spectator_image() if self.rig else None
                rec_camera_images = self.rig.get_latest_images() if self.rig else {}
                if spectator_img is None:
                    spectator_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                self.artifact_recorder.record_step(
                    step_idx=self.step_count - 1,
                    ego_state=ego_state,
                    control_cmd=cmd,
                    prediction=self.latest_prediction,
                    camera_images=rec_camera_images,
                    spectator_image=spectator_img,
                    collision_event=self._last_collision_event,
                    lane_invasion_event=self._last_lane_invasion_event,
                )
                self._last_collision_event = None
                self._last_lane_invasion_event = None

            # Update simulator window spectator camera to follow ego vehicle
            if self.spectator_follow:
                self._update_spectator(tf)

            # Render Pygame HUD dashboard
            if self.enable_display and self.display is not None:
                camera_images = self.rig.get_latest_images() if self.rig else {}
                if not camera_images:
                    for name, spec in self.specs.items():
                        camera_images[name] = np.zeros(
                            (spec.intrinsics.height, spec.intrinsics.width, 3), dtype=np.uint8
                        )
                model_display_name = (
                    f"{self.model_type.value} [SIMULATED]"
                    if self.is_simulated_backend
                    else f"{self.model_type.value} [REAL]"
                )
                keep_running = self.display.draw(
                    camera_images=camera_images,
                    model_name=model_display_name,
                    high_level_command=high_level_command.value,
                    text_prompt=self.text_prompt,
                    current_speed_mps=speed,
                    target_speed_mps=self.target_speed,
                    throttle=cmd.throttle,
                    brake=cmd.brake,
                    steer=cmd.steer,
                    cross_track_error=cmd.cross_track_error,
                    heading_error_rad=cmd.heading_error_rad,
                    reasoning=self.latest_prediction.reasoning,
                    waypoints=self.latest_prediction.waypoints,
                    step=self.step_count,
                    sim_time_s=self.step_count * self.dt,
                    wall_time_s=time.time() - self.start_wall_time,
                )
                if not keep_running:
                    logger.info("User requested exit via Pygame window.")
                    raise KeyboardInterrupt

                if self.screenshot_path and self.step_count >= self.screenshot_tick and not self._screenshot_taken:
                    try:
                        self.display.screenshot(self.screenshot_path)
                        logger.info(f"[COCKPIT SCREENSHOT] Captured and saved to {self.screenshot_path}")
                        self._screenshot_taken = True
                    except Exception as e:
                        logger.warning(f"Failed to capture cockpit screenshot: {e}")

            return step_data

        return {"step": self.step_count, "speed_mps": speed}

    def _update_spectator(self, ego_tf: Any) -> None:
        """Update CARLA main simulator window camera to follow the ego vehicle from above."""
        try:
            if not hasattr(self.world, "get_spectator"):
                return
            spectator = self.world.get_spectator()
            if spectator is None:
                return

            yaw_rad = math.radians(ego_tf.rotation.yaw)
            if self.spectator_view_mode == "topdown":
                cam_x = ego_tf.location.x - 2.0 * math.cos(yaw_rad)
                cam_y = ego_tf.location.y - 2.0 * math.sin(yaw_rad)
                cam_z = ego_tf.location.z + 18.0
                pitch = -75.0
            else:
                # Normal elevated third-person chase camera (from above and behind)
                dist = 8.5
                height = 4.2
                cam_x = ego_tf.location.x - dist * math.cos(yaw_rad)
                cam_y = ego_tf.location.y - dist * math.sin(yaw_rad)
                cam_z = ego_tf.location.z + height
                pitch = -16.0

            try:
                import carla
                spectator_tf = carla.Transform(
                    carla.Location(x=cam_x, y=cam_y, z=cam_z),
                    carla.Rotation(pitch=pitch, yaw=ego_tf.rotation.yaw, roll=0.0),
                )
                spectator.set_transform(spectator_tf)
            except Exception:
                pass
        except Exception:
            pass

    def cleanup(self) -> None:
        """Teardown sensors, display, and sockets."""
        logger.info("Cleaning up E2E closed-loop runner...")
        if hasattr(self, "display") and self.display is not None:
            try:
                self.display.close()
            except Exception:
                pass
        if self.rig:
            self.rig.destroy()
        if self.collision_sensor:
            try:
                self.collision_sensor.destroy()
            except Exception:
                pass
        if self.lane_sensor:
            try:
                self.lane_sensor.destroy()
            except Exception:
                pass
        if self.remote_sock:
            try:
                self.remote_sock.close()
            except Exception:
                pass
        if self.local_adapter:
            self.local_adapter.shutdown()
        if self.artifact_recorder is not None:
            try:
                self.artifact_recorder.finalize(
                    extra_metadata={
                        "is_simulated_backend": self.is_simulated_backend,
                        "target_speed_mps": self.target_speed,
                        "model_path": self.model_path,
                    }
                )
            except Exception as e:
                logger.warning(f"Error finalizing simulation artifacts: {e}")
            self.artifact_recorder = None
        self.telemetry.close()
