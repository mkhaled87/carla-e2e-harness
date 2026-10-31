"""Unit tests for E2EArtifactRecorder (video compilation and artifact packaging)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from e2e_artifact_recorder import E2EArtifactRecorder
from e2e_schema import EgoKinematicState, CoordinateFrame


class TestE2EArtifactRecorder(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_carla_artifacts_"))

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_steps_and_finalize(self):
        recorder = E2EArtifactRecorder(
            record_dir=self.test_dir,
            dt=0.05,
            model_type="autoe2e",
            spectator_view_mode="chase",
        )

        ego_state = EgoKinematicState(
            timestamp_us=1000000,
            x=10.0, y=20.0, z=0.5,
            roll=0.0, pitch=0.0, yaw=90.0,
            vx=2.0, vy=0.0, vz=0.0,
            speed_mps=2.0,
            wz=1.5,
            frame=CoordinateFrame.CARLA_LHD,
        )

        dummy_spectator = np.full((720, 1280, 3), 120, dtype=np.uint8)
        dummy_front = np.full((450, 800, 3), 80, dtype=np.uint8)

        # Record 5 frames
        for step in range(5):
            recorder.record_step(
                step_idx=step,
                ego_state=ego_state,
                control_cmd=None,
                prediction=None,
                camera_images={"CAM_FRONT": dummy_front},
                spectator_image=dummy_spectator,
                collision_event={"other_actor": "static.prop.barrier"} if step == 3 else None,
            )

        video_path = recorder.finalize()

        # Verify artifacts
        self.assertTrue(self.test_dir.exists())
        self.assertTrue(video_path.exists(), f"Video {video_path} was not created")
        self.assertGreater(video_path.stat().st_size, 0)

        # Check spectator frames
        spectator_frames = list((self.test_dir / "cameras" / "spectator").glob("*.jpg"))
        self.assertEqual(len(spectator_frames), 5)

        # Check CAM_FRONT frames
        front_frames = list((self.test_dir / "cameras" / "CAM_FRONT").glob("*.jpg"))
        self.assertEqual(len(front_frames), 5)

        # Check telemetry jsonl
        telemetry_file = self.test_dir / "vehicle_pose_telemetry.jsonl"
        self.assertTrue(telemetry_file.exists())
        with open(telemetry_file) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0]["step"], 0)
        self.assertEqual(lines[0]["pose"]["x"], 10.0)

        # Check collision event
        col_file = self.test_dir / "sensors" / "collision_events.jsonl"
        self.assertTrue(col_file.exists())
        with open(col_file) as f:
            col_lines = [json.loads(line) for line in f]
        self.assertEqual(len(col_lines), 1)
        self.assertEqual(col_lines[0]["step"], 3)
        self.assertEqual(col_lines[0]["other_actor"], "static.prop.barrier")

        # Check metadata.json
        meta_file = self.test_dir / "metadata.json"
        self.assertTrue(meta_file.exists())
        with open(meta_file) as f:
            meta = json.load(f)
        self.assertEqual(meta["total_steps"], 5)
        self.assertEqual(meta["target_framerate_fps"], 20)
        self.assertEqual(meta["total_collisions"], 1)
        self.assertIn("CAM_FRONT", meta["onboard_cameras"])


if __name__ == "__main__":
    unittest.main()
