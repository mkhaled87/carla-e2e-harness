"""Unit tests for E2ETrajectoryTracker."""

from __future__ import annotations

import unittest
import numpy as np

from e2e_schema import CoordinateFrame, E2EPrediction
from e2e_tracker import E2ETrajectoryTracker


class TestE2ETracker(unittest.TestCase):
    def setUp(self):
        self.tracker = E2ETrajectoryTracker()

    def test_straight_cruising(self):
        # Straight line trajectory at 8 m/s
        waypoints = np.array([
            [0.8 * i, 0.0] for i in range(1, 21)
        ], dtype=np.float32)
        prediction = E2EPrediction(
            waypoints=waypoints,
            frame=CoordinateFrame.CARLA_LHD,
            target_speeds=np.full(len(waypoints), 8.0, dtype=np.float32),
        )

        cmd = self.tracker.compute_control(prediction, current_speed_mps=5.0, dt=0.05)
        # Vehicle is going 5 m/s, target is 8 m/s -> needs acceleration (throttle > 0)
        self.assertGreater(cmd.throttle, 0.0)
        self.assertEqual(cmd.brake, 0.0)
        # Straight path -> steer close to 0
        self.assertAlmostEqual(cmd.steer, 0.0, places=2)
        self.assertFalse(cmd.stop_hold_active)

    def test_left_turn_tracking(self):
        # Left turn path in CARLA left-handed: y is negative
        waypoints = np.array([
            [i * 1.0, -0.3 * (i ** 1.2)] for i in range(1, 21)
        ], dtype=np.float32)
        prediction = E2EPrediction(
            waypoints=waypoints,
            frame=CoordinateFrame.CARLA_LHD,
            target_speeds=np.full(len(waypoints), 6.0, dtype=np.float32),
        )

        cmd = self.tracker.compute_control(prediction, current_speed_mps=6.0, dt=0.05)
        # In CARLA left-handed: left is negative steer
        self.assertLess(cmd.steer, 0.0)

    def test_right_turn_tracking(self):
        # Right turn path in CARLA left-handed: y is positive
        waypoints = np.array([
            [i * 1.0, 0.3 * (i ** 1.2)] for i in range(1, 21)
        ], dtype=np.float32)
        prediction = E2EPrediction(
            waypoints=waypoints,
            frame=CoordinateFrame.CARLA_LHD,
            target_speeds=np.full(len(waypoints), 6.0, dtype=np.float32),
        )

        cmd = self.tracker.compute_control(prediction, current_speed_mps=6.0, dt=0.05)
        # In CARLA left-handed: right is positive steer
        self.assertGreater(cmd.steer, 0.0)

    def test_stop_command(self):
        waypoints = np.array([[0.1, 0.0], [0.2, 0.0]], dtype=np.float32)
        prediction = E2EPrediction(
            waypoints=waypoints,
            frame=CoordinateFrame.CARLA_LHD,
            target_speeds=np.array([0.0, 0.0], dtype=np.float32),
        )

        # Vehicle moving at 4 m/s -> commanded to stop
        cmd = self.tracker.compute_control(prediction, current_speed_mps=4.0, dt=0.05)
        self.assertEqual(cmd.throttle, 0.0)
        self.assertGreaterEqual(cmd.brake, 0.25)

        # Vehicle already stationary (0.05 m/s) -> hold brake active
        cmd_stop = self.tracker.compute_control(prediction, current_speed_mps=0.05, dt=0.05)
        self.assertEqual(cmd_stop.throttle, 0.0)
        self.assertEqual(cmd_stop.brake, 1.0)
        self.assertTrue(cmd_stop.stop_hold_active)


if __name__ == "__main__":
    unittest.main()

