"""Unit tests for E2E schema and types."""

from __future__ import annotations

import unittest
import numpy as np

from e2e_schema import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraSpec,
    CoordinateFrame,
    E2EObservation,
    E2EPrediction,
    EgoKinematicState,
    HighLevelCommand,
    ModelType,
)


class TestE2ESchema(unittest.TestCase):
    def test_camera_intrinsics_derivation(self):
        intr = CameraIntrinsics.from_fov(width=800, height=450, fov=70.0)
        self.assertEqual(intr.width, 800)
        self.assertEqual(intr.height, 450)
        self.assertEqual(intr.cx, 400.0)
        self.assertEqual(intr.cy, 225.0)
        self.assertAlmostEqual(intr.fx, intr.fy, places=3)
        self.assertGreater(intr.fx, 0.0)

        K = intr.to_matrix()
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 2], 400.0)
        self.assertEqual(K[1, 2], 225.0)

    def test_ego_kinematic_state(self):
        state = EgoKinematicState(
            timestamp_us=1000000,
            x=10.0,
            y=20.0,
            z=0.5,
            roll=0.0,
            pitch=1.0,
            yaw=45.0,
            vx=5.0,
            vy=0.0,
            vz=0.0,
            speed_mps=5.0,
        )
        d = state.to_dict()
        self.assertEqual(d["x"], 10.0)
        self.assertEqual(d["speed_mps"], 5.0)
        self.assertEqual(d["frame"], CoordinateFrame.CARLA_LHD.value)

    def test_e2e_observation_validation(self):
        intr = CameraIntrinsics.from_fov(640, 360, 90.0)
        extr = CameraExtrinsics(x=1.5, y=0.0, z=1.6)
        spec = CameraSpec(name="CAM_FRONT", intrinsics=intr, extrinsics=extr)

        state = EgoKinematicState(
            timestamp_us=1000, x=0, y=0, z=0,
            roll=0, pitch=0, yaw=0, vx=0, vy=0, vz=0, speed_mps=0,
        )

        # Missing images
        obs_empty = E2EObservation(
            model_type=ModelType.MOCK,
            frame_id=1,
            timestamp_us=1000,
            camera_images={},
            camera_specs={"CAM_FRONT": spec},
            ego_state=state,
        )
        with self.assertRaises(ValueError):
            obs_empty.validate()

        # Valid observation
        valid_img = np.zeros((360, 640, 3), dtype=np.uint8)
        obs_valid = E2EObservation(
            model_type=ModelType.MOCK,
            frame_id=1,
            timestamp_us=1000,
            camera_images={"CAM_FRONT": valid_img},
            camera_specs={"CAM_FRONT": spec},
            ego_state=state,
            high_level_command=HighLevelCommand.LANE_FOLLOW,
        )
        obs_valid.validate()  # Should not raise

    def test_e2e_prediction_post_init(self):
        waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
        pred = E2EPrediction(waypoints=waypoints)
        self.assertEqual(pred.num_points, 3)
        self.assertEqual(len(pred.timestamps), 3)
        self.assertAlmostEqual(pred.timestamps[0], 0.1)
        self.assertAlmostEqual(pred.timestamps[2], 0.3)


if __name__ == "__main__":
    unittest.main()

