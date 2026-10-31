"""Unit tests for sensor rig definitions."""

from __future__ import annotations

import unittest

from e2e_schema import ModelType
from e2e_rigs import build_rig_specs


class TestE2ERigs(unittest.TestCase):
    def test_nuscenes_rig(self):
        # AutoE2E and VAD use nuScenes 6-camera rig
        specs = build_rig_specs(ModelType.AUTO_E2E, image_width=800, image_height=450)
        self.assertEqual(len(specs), 6)
        expected_names = {
            "CAM_FRONT", "CAM_BACK", "CAM_FRONT_LEFT",
            "CAM_BACK_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
        }
        self.assertEqual(set(specs.keys()), expected_names)
        self.assertEqual(specs["CAM_FRONT"].intrinsics.width, 800)
        self.assertEqual(specs["CAM_FRONT"].intrinsics.height, 450)
        self.assertEqual(specs["CAM_BACK"].intrinsics.fov, 110.0)

    def test_alpamayo_rig(self):
        specs = build_rig_specs(ModelType.ALPAMAYO)
        self.assertEqual(len(specs), 7)
        expected_names = {
            "front_cross_left", "front_wide", "front_cross_right",
            "rear_left", "rear_tele", "rear_right", "front_tele",
        }
        self.assertEqual(set(specs.keys()), expected_names)
        self.assertEqual(specs["front_wide"].intrinsics.fov, 120.0)
        self.assertEqual(specs["front_tele"].intrinsics.fov, 30.0)

    def test_qwendrive_rig(self):
        specs = build_rig_specs(ModelType.QWEN_DRIVE)
        self.assertEqual(len(specs), 7)
        self.assertIn("CAM_FRONT_WIDE", specs)
        self.assertIn("CAM_FRONT_TELE", specs)
        self.assertEqual(specs["CAM_FRONT_WIDE"].intrinsics.fov, 100.0)
        self.assertEqual(specs["CAM_FRONT_TELE"].intrinsics.fov, 35.0)


if __name__ == "__main__":
    unittest.main()

