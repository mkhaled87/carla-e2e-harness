"""Unit tests for E2E coordinate transforms."""

from __future__ import annotations

import unittest
import numpy as np

from e2e_schema import CoordinateFrame
from e2e_coordinates import (
    carla_to_iso_points,
    iso_to_carla_points,
    carla_to_iso_yaw,
    iso_to_carla_yaw,
    rear_axle_to_center_offset,
    center_to_rear_axle_offset,
    local_to_world_2d,
    world_to_local_2d,
    normalize_trajectory_to_carla,
)


class TestE2ECoordinates(unittest.TestCase):
    def test_point_reflection(self):
        pts_carla = np.array([[5.0, 2.0, 1.0], [10.0, -3.0, 0.0]], dtype=np.float32)
        pts_iso = carla_to_iso_points(pts_carla)
        np.testing.assert_allclose(pts_iso[:, 0], pts_carla[:, 0])
        np.testing.assert_allclose(pts_iso[:, 1], -pts_carla[:, 1])
        np.testing.assert_allclose(pts_iso[:, 2], pts_carla[:, 2])

        # Invert back
        pts_roundtrip = iso_to_carla_points(pts_iso)
        np.testing.assert_allclose(pts_roundtrip, pts_carla)

    def test_yaw_conversion(self):
        yaw_carla = 30.0  # 30 deg right in CARLA
        yaw_iso = carla_to_iso_yaw(yaw_carla)
        self.assertEqual(yaw_iso, -30.0)
        self.assertEqual(iso_to_carla_yaw(yaw_iso), yaw_carla)

    def test_rear_axle_offsets(self):
        rear_axle_x = -1.425
        rear_axle_pts = np.array([[0.0, 0.0]], dtype=np.float32)
        center_pts = rear_axle_to_center_offset(rear_axle_pts, rear_axle_x)
        self.assertAlmostEqual(center_pts[0, 0], -1.425)

        roundtrip = center_to_rear_axle_offset(center_pts, rear_axle_x)
        self.assertAlmostEqual(roundtrip[0, 0], 0.0)

    def test_local_world_roundtrip(self):
        local_pts = np.array([[10.0, 2.0], [20.0, -1.0]], dtype=np.float32)
        origin_x, origin_y, origin_yaw = 100.0, 200.0, 45.0
        world_pts = local_to_world_2d(local_pts, origin_x, origin_y, origin_yaw, is_carla_frame=True)
        rec_local = world_to_local_2d(world_pts, origin_x, origin_y, origin_yaw, is_carla_frame=True)
        np.testing.assert_allclose(rec_local, local_pts, atol=1e-5)

    def test_normalize_trajectory_to_carla(self):
        # Trajectory in ISO 8855 FLU defined relative to rear axle
        flu_pts = np.array([[5.0, 1.0]], dtype=np.float32)  # 5m forward, 1m left
        carla_pts = normalize_trajectory_to_carla(
            flu_pts,
            source_frame=CoordinateFrame.ISO_8855_FLU,
            rear_axle_relative=True,
            rear_axle_x=-1.425,
        )
        # x_carla = 5.0 - 1.425 = 3.575
        # y_carla = -1.0 (left is -y in CARLA)
        self.assertAlmostEqual(carla_pts[0, 0], 3.575)
        self.assertAlmostEqual(carla_pts[0, 1], -1.0)


if __name__ == "__main__":
    unittest.main()

