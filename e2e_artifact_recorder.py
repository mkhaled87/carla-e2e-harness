"""Simulation Artifact Recorder for CARLA E2E Autonomous Driving.

Captures and packages closed-loop simulation artifacts into a structured directory:
- Spectator Chase Video (.mp4) encoded at real-world simulation rate (1.0 / dt FPS).
- Individual stream image sequences for each onboard camera sensor.
- Vehicle pose, kinematics, steering/throttle/brake controls, and model predictions.
- Sensor events (collisions, lane invasions).
- Machine-readable metadata.json and sensors summary.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

logger = logging.getLogger("CARLA_E2E_RECORDER")


class E2EArtifactRecorder:
    """Manages the creation and export of closed-loop simulation artifacts."""

    def __init__(
        self,
        record_dir: str | Path,
        dt: float = 0.05,
        model_type: str = "autoe2e",
        spectator_view_mode: str = "chase",
    ) -> None:
        self.record_dir = Path(record_dir).expanduser().resolve()
        self.dt = dt
        self.fps = max(1, int(round(1.0 / dt)))
        self.model_type = model_type
        self.spectator_view_mode = spectator_view_mode
        self.step_count = 0
        self.start_time = time.time()

        # Artifact directory layout
        self.cameras_dir = self.record_dir / "cameras"
        self.spectator_dir = self.cameras_dir / "spectator"
        self.sensors_dir = self.record_dir / "sensors"

        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.cameras_dir.mkdir(parents=True, exist_ok=True)
        self.spectator_dir.mkdir(parents=True, exist_ok=True)
        self.sensors_dir.mkdir(parents=True, exist_ok=True)

        # Telemetry and sensor files
        self.telemetry_path = self.record_dir / "vehicle_pose_telemetry.jsonl"
        self.collisions_path = self.sensors_dir / "collision_events.jsonl"
        self.lane_invasions_path = self.sensors_dir / "lane_invasions.jsonl"
        self.metadata_path = self.record_dir / "metadata.json"
        self.video_path = self.record_dir / "spectator_drive.mp4"

        self.telemetry_file = open(self.telemetry_path, "w", encoding="utf-8", buffering=1)
        self.collisions_file = open(self.collisions_path, "w", encoding="utf-8", buffering=1)
        self.lane_invasions_file = open(self.lane_invasions_path, "w", encoding="utf-8", buffering=1)

        self.recorded_cameras: set[str] = set()
        self.total_collisions = 0
        self.total_lane_invasions = 0
        self.speeds: list[float] = []

        logger.info(f"[RECORDER] Initialized simulation artifact directory: {self.record_dir}")

    def record_step(
        self,
        step_idx: int,
        ego_state: Any,
        control_cmd: Any = None,
        prediction: Any = None,
        camera_images: dict[str, np.ndarray] | None = None,
        spectator_image: np.ndarray | None = None,
        collision_event: dict[str, Any] | None = None,
        lane_invasion_event: dict[str, Any] | None = None,
    ) -> None:
        """Record state, sensor frames, and controls for a single simulation step."""
        self.step_count = max(self.step_count, step_idx + 1)
        sim_time_s = round(step_idx * self.dt, 4)
        wall_time_s = time.time()

        if ego_state is not None:
            speed_val = float(getattr(ego_state, "speed_mps", 0.0))
            self.speeds.append(speed_val)

        # 1. Save Spectator Image
        if spectator_image is not None and spectator_image.size > 0:
            frame_path = self.spectator_dir / f"{step_idx:05d}.jpg"
            self._save_image(spectator_image, frame_path)

        # 2. Save Onboard Multi-Camera Stream Images
        if camera_images:
            for cam_name, img_arr in camera_images.items():
                if img_arr is not None and img_arr.size > 0:
                    self.recorded_cameras.add(cam_name)
                    cam_dir = self.cameras_dir / cam_name
                    if not cam_dir.exists():
                        cam_dir.mkdir(parents=True, exist_ok=True)
                    cam_frame_path = cam_dir / f"{step_idx:05d}.jpg"
                    self._save_image(img_arr, cam_frame_path)

        # 3. Log Pose, Kinematics, Controls, and Model Telemetry
        telemetry_entry = {
            "step": step_idx,
            "sim_time_s": sim_time_s,
            "wall_time_s": wall_time_s,
        }

        if ego_state is not None:
            telemetry_entry["pose"] = {
                "x": float(getattr(ego_state, "x", 0.0)),
                "y": float(getattr(ego_state, "y", 0.0)),
                "z": float(getattr(ego_state, "z", 0.0)),
                "roll": float(getattr(ego_state, "roll", 0.0)),
                "pitch": float(getattr(ego_state, "pitch", 0.0)),
                "yaw": float(getattr(ego_state, "yaw", 0.0)),
                "frame": getattr(getattr(ego_state, "frame", None), "value", "carla_lhd"),
            }
            telemetry_entry["velocity"] = {
                "vx": float(getattr(ego_state, "vx", 0.0)),
                "vy": float(getattr(ego_state, "vy", 0.0)),
                "vz": float(getattr(ego_state, "vz", 0.0)),
                "speed_mps": float(getattr(ego_state, "speed_mps", 0.0)),
                "speed_kmh": float(getattr(ego_state, "speed_mps", 0.0) * 3.6),
            }
            telemetry_entry["acceleration"] = {
                "ax": float(getattr(ego_state, "ax", 0.0)),
                "ay": float(getattr(ego_state, "ay", 0.0)),
                "az": float(getattr(ego_state, "az", 0.0)),
            }
            wz_val = float(getattr(ego_state, "wz", 0.0))
            telemetry_entry["angular_velocity"] = {
                "wx_deg_s": float(getattr(ego_state, "wx", 0.0)),
                "wy_deg_s": float(getattr(ego_state, "wy", 0.0)),
                "wz_deg_s": wz_val,
                "wz_rad_s": math.radians(wz_val),
            }

        if control_cmd is not None:
            telemetry_entry["controls"] = {
                "throttle": float(getattr(control_cmd, "throttle", 0.0)),
                "brake": float(getattr(control_cmd, "brake", 0.0)),
                "steer": float(getattr(control_cmd, "steer", 0.0)),
                "target_speed_mps": float(getattr(control_cmd, "target_speed_mps", 0.0)),
                "target_steer_rad": float(getattr(control_cmd, "target_steer_rad", 0.0)),
                "target_accel_mps2": float(getattr(control_cmd, "target_accel_mps2", 0.0)),
                "curvature": float(getattr(control_cmd, "curvature", 0.0)),
                "cross_track_error": float(getattr(control_cmd, "cross_track_error", 0.0)),
                "heading_error_rad": float(getattr(control_cmd, "heading_error_rad", 0.0)),
                "stop_hold_active": bool(getattr(control_cmd, "stop_hold_active", False)),
            }

        if prediction is not None:
            dc = getattr(prediction, "direct_control", None) or {}
            telemetry_entry["prediction"] = {
                "target_speed_mps": float(dc.get("target_speed_mps", 0.0)),
                "target_acceleration": float(dc.get("target_acceleration", 0.0)),
                "target_curvature": float(dc.get("target_curvature", 0.0)),
                "steering_tire_angle": float(dc.get("steering_tire_angle", 0.0)),
                "confidence": float(getattr(prediction, "confidence", 1.0)),
                "inference_seconds": float(getattr(prediction, "inference_seconds", 0.0)),
                "reasoning": str(getattr(prediction, "reasoning", "")),
            }

        telemetry_entry["sensors"] = {
            "total_collisions": self.total_collisions,
            "total_lane_invasions": self.total_lane_invasions,
        }

        self.telemetry_file.write(json.dumps(telemetry_entry) + "\n")

        # 4. Sensor Events
        if collision_event:
            self.total_collisions += 1
            col_record = {
                "step": step_idx,
                "sim_time_s": sim_time_s,
                "wall_time_s": wall_time_s,
                **collision_event,
            }
            self.collisions_file.write(json.dumps(col_record) + "\n")

        if lane_invasion_event:
            self.total_lane_invasions += 1
            lane_record = {
                "step": step_idx,
                "sim_time_s": sim_time_s,
                "wall_time_s": wall_time_s,
                **lane_invasion_event,
            }
            self.lane_invasions_file.write(json.dumps(lane_record) + "\n")

    def finalize(self, extra_metadata: dict[str, Any] | None = None) -> Path:
        """Compile the spectator video, write metadata.json, and finalize the artifact directory."""
        # Close open JSONL streams
        if self.telemetry_file:
            self.telemetry_file.close()
            self.telemetry_file = None
        if self.collisions_file:
            self.collisions_file.close()
            self.collisions_file = None
        if self.lane_invasions_file:
            self.lane_invasions_file.close()
            self.lane_invasions_file = None

        wall_duration = time.time() - self.start_time
        sim_duration = self.step_count * self.dt
        avg_speed = float(np.mean(self.speeds)) if self.speeds else 0.0
        max_speed = float(np.max(self.speeds)) if self.speeds else 0.0

        # Construct Spectator Video at exact simulation rate (dt -> fps)
        video_success = self._compile_spectator_video()

        # Metadata bundle
        metadata = {
            "model_type": self.model_type,
            "spectator_view_mode": self.spectator_view_mode,
            "total_steps": self.step_count,
            "simulation_dt": self.dt,
            "target_framerate_fps": self.fps,
            "simulation_duration_s": round(sim_duration, 3),
            "wall_clock_duration_s": round(wall_duration, 3),
            "lockstep_speed_ratio": round(sim_duration / max(wall_duration, 1e-4), 3),
            "average_speed_mps": round(avg_speed, 2),
            "max_speed_mps": round(max_speed, 2),
            "total_collisions": self.total_collisions,
            "total_lane_invasions": self.total_lane_invasions,
            "spectator_video": self.video_path.name if video_success else None,
            "onboard_cameras": sorted(list(self.recorded_cameras)),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Log artifact report
        video_info = (
            f"{self.video_path} ({self.fps} FPS, {self.step_count} frames, {sim_duration:.2f}s)"
            if video_success
            else "Encoding skipped (no spectator frames)"
        )
        logger.info(
            f"\n{'='*75}\n"
            f"[RECORDER] Simulation Artifacts Saved Successfully!\n"
            f"  - Output Directory: {self.record_dir}\n"
            f"  - Spectator Video:  {video_info}\n"
            f"  - Vehicle Pose:     {self.telemetry_path}\n"
            f"  - Camera Streams:   {self.cameras_dir} ({len(self.recorded_cameras)} cameras)\n"
            f"  - Sensor Events:    {self.sensors_dir}\n"
            f"  - Metadata:         {self.metadata_path}\n"
            f"{'='*75}"
        )

        return self.video_path

    def _compile_spectator_video(self) -> bool:
        """Encode JPEG frames from cameras/spectator into spectator_drive.mp4 at exact simulation FPS."""
        frames = sorted(list(self.spectator_dir.glob("*.jpg")))
        if not frames:
            logger.warning("[RECORDER] No spectator frames found to compile video.")
            return False

        logger.info(
            f"[RECORDER] Encoding {len(frames)} spectator frames into {self.video_path.name} "
            f"at {self.fps} FPS ({len(frames)*self.dt:.2f}s playback duration)..."
        )

        # Method 1: ffmpeg CLI (High-fidelity H.264 yuv420p compatible with QuickTime, Safari, Chrome)
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            try:
                frame_pattern = str(self.spectator_dir / "%05d.jpg")
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-framerate", str(self.fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "20",
                    "-preset", "fast",
                    str(self.video_path),
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0 and self.video_path.exists():
                    logger.info(f"[RECORDER] Successfully rendered H.264 video with ffmpeg: {self.video_path}")
                    return True
                else:
                    logger.warning(f"[RECORDER] ffmpeg returned code {res.returncode}: {res.stderr[-300:]}")
            except Exception as e:
                logger.warning(f"[RECORDER] ffmpeg encoding attempt failed: {e}")

        # Method 2: OpenCV VideoWriter Fallback
        try:
            import cv2
            first_frame = cv2.imread(str(frames[0]))
            if first_frame is None:
                return False
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.video_path), fourcc, float(self.fps), (w, h))
            for frame_path in frames:
                img = cv2.imread(str(frame_path))
                if img is not None:
                    writer.write(img)
            writer.release()
            if self.video_path.exists():
                logger.info(f"[RECORDER] Successfully rendered MP4 video with OpenCV: {self.video_path}")
                return True
        except Exception as e:
            logger.warning(f"[RECORDER] OpenCV video encoding failed: {e}")

        return False

    def _save_image(self, rgb_image: np.ndarray, file_path: Path) -> None:
        """Save RGB image array as JPEG."""
        try:
            import cv2
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(file_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        except ImportError:
            from PIL import Image
            img = Image.fromarray(rgb_image)
            img.save(str(file_path), "JPEG", quality=92)

