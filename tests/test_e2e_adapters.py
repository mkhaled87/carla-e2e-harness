"""Unit tests for all E2E model adapters."""

from __future__ import annotations

import unittest
import numpy as np

from adapters.alpamayo_adapter import AlpamayoAdapter
from adapters.autoe2e_adapter import AutoE2EAdapter
from adapters.mock_adapter import MockE2EAdapter
from adapters.qwendrive_adapter import QwenDriveAdapter
from adapters.vad_adapter import VADAdapter
from e2e_rigs import build_rig_specs
from e2e_schema import (
    CoordinateFrame,
    E2EObservation,
    EgoKinematicState,
    HighLevelCommand,
    ModelType,
)


class TestE2EAdapters(unittest.TestCase):
    def _create_test_observation(self, model_type: ModelType, cmd: HighLevelCommand = HighLevelCommand.LANE_FOLLOW) -> E2EObservation:
        specs = build_rig_specs(model_type, image_width=64, image_height=36)
        images = {
            name: np.zeros((spec.intrinsics.height, spec.intrinsics.width, 3), dtype=np.uint8)
            for name, spec in specs.items()
        }
        state = EgoKinematicState(
            timestamp_us=1000000,
            x=0.0, y=0.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            vx=6.0, vy=0.0, vz=0.0,
            speed_mps=6.0,
        )
        return E2EObservation(
            model_type=model_type,
            frame_id=1,
            timestamp_us=1000000,
            camera_images=images,
            camera_specs=specs,
            ego_state=state,
            high_level_command=cmd,
            text_prompt="Follow route and cruise smoothly.",
        )

    def test_mock_adapter(self):
        adapter = MockE2EAdapter(default_target_speed_mps=8.0)
        adapter.load()
        obs = self._create_test_observation(ModelType.MOCK)
        pred = adapter.infer(obs)
        self.assertEqual(pred.frame, CoordinateFrame.CARLA_LHD)
        self.assertEqual(pred.num_points, 50)
        self.assertGreater(pred.waypoints[-1, 0], 10.0)
        self.assertIn("cruising", pred.reasoning.lower())

    def test_qwendrive_adapter(self):
        adapter = QwenDriveAdapter()
        adapter.load(force_simulated=True)
        obs = self._create_test_observation(ModelType.QWEN_DRIVE, cmd=HighLevelCommand.TURN_LEFT)
        pred = adapter.infer(obs)
        self.assertEqual(pred.num_points, 50)
        self.assertIn("Flow Matching", pred.reasoning)
        self.assertIn("left", pred.reasoning.lower())
        # In CARLA left-handed: left turn has negative y
        self.assertLess(pred.waypoints[-1, 1], 0.0)

    def test_autoe2e_adapter(self):
        adapter = AutoE2EAdapter()
        adapter.load(force_simulated=True)
        obs = self._create_test_observation(ModelType.AUTO_E2E, cmd=HighLevelCommand.LANE_FOLLOW)
        pred = adapter.infer(obs)
        self.assertEqual(pred.num_points, 30)
        self.assertIn("AutoE2E", pred.reasoning)
        self.assertIn("steering_tire_angle", pred.direct_control)
        self.assertIn("target_acceleration", pred.direct_control)

    def test_alpamayo_adapter(self):
        adapter = AlpamayoAdapter()
        adapter.load(force_simulated=True)
        obs = self._create_test_observation(ModelType.ALPAMAYO, cmd=HighLevelCommand.LANE_FOLLOW)
        pred = adapter.infer(obs)
        self.assertEqual(pred.frame, CoordinateFrame.ISO_8855_FLU)
        self.assertEqual(pred.num_points, 64)
        self.assertIn("Chain-of-Causation", pred.reasoning)

    def test_vad_adapter(self):
        adapter = VADAdapter()
        adapter.load(force_simulated=True)
        obs = self._create_test_observation(ModelType.VAD, cmd=HighLevelCommand.STOP)
        pred = adapter.infer(obs)
        self.assertEqual(pred.frame, CoordinateFrame.ISO_8855_FLU)
        self.assertEqual(pred.num_points, 30)
        self.assertIn("VAD", pred.reasoning)
        self.assertAlmostEqual(pred.direct_control["target_speed_mps"], 0.0)

    def test_lane_change_right_and_left(self):
        adapter = QwenDriveAdapter()
        adapter.load(force_simulated=True)

        # Right lane change
        obs_right = self._create_test_observation(ModelType.QWEN_DRIVE, cmd=HighLevelCommand.LANE_CHANGE_RIGHT)
        pred_right = adapter.infer(obs_right)
        self.assertIn("right", pred_right.reasoning.lower())
        self.assertGreater(pred_right.waypoints[-1, 1], 2.5)  # Lateral displacement > 2.5m

        # Left lane change
        obs_left = self._create_test_observation(ModelType.QWEN_DRIVE, cmd=HighLevelCommand.LANE_CHANGE_LEFT)
        pred_left = adapter.infer(obs_left)
        self.assertIn("left", pred_left.reasoning.lower())
        self.assertLess(pred_left.waypoints[-1, 1], -2.5)  # Lateral displacement < -2.5m

    def test_curved_lane_centerline_tracking(self):
        adapter = AutoE2EAdapter()
        adapter.load(force_simulated=True)

        # Create curved centerline bending sharply to the right (x=1..40, y=0.05*x^2)
        curve_cl = [[float(x), float(0.02 * (x ** 2))] for x in range(1, 41)]
        obs = self._create_test_observation(ModelType.AUTO_E2E, cmd=HighLevelCommand.LANE_FOLLOW)
        obs.custom_metadata["local_lane_centerline"] = curve_cl

        pred = adapter.infer(obs)
        self.assertGreater(pred.waypoints[-1, 1], 2.0)  # Followed the right curve
        self.assertGreater(pred.direct_control["target_curvature"], 0.0)

    def test_autoe2e_real_weights_path(self):
        try:
            import torch
        except ImportError:
            from unittest.mock import MagicMock
            import sys
            mock_torch = MagicMock()
            mock_torch.from_numpy = lambda arr: MagicMock(to=lambda dev: arr)
            sys.modules["torch"] = mock_torch

        adapter = AutoE2EAdapter(device="cpu")
        class MockAutoE2EModel:
            def __call__(self, *args, **kwargs):
                import torch
                # Output 128 elements (64 timesteps x [accel, curvature])
                return torch.zeros((1, 128), dtype=torch.float32)

        adapter.pytorch_model = MockAutoE2EModel()
        adapter.use_simulated_backend = False
        adapter.is_loaded = True

        obs = self._create_test_observation(ModelType.AUTO_E2E, cmd=HighLevelCommand.LANE_FOLLOW)
        pred = adapter.infer(obs)
        self.assertEqual(pred.num_points, 30)
        self.assertIn("Real PyTorch", pred.reasoning)
        self.assertTrue(pred.perception_outputs.get("real_weights_executed"))

    def test_alpamayo_real_weights_path(self):
        try:
            import torch
        except ImportError:
            from unittest.mock import MagicMock
            import sys
            mock_torch = MagicMock()
            mock_torch.from_numpy = lambda arr: MagicMock(unsqueeze=lambda dim: MagicMock(to=lambda dev: arr))
            sys.modules["torch"] = mock_torch

        adapter = AlpamayoAdapter(device="cpu")
        class MockDiffusionModel:
            def sample(self, cameras, ego_history, steps, command):
                from unittest.mock import MagicMock
                out = MagicMock()
                out.cpu.return_value.numpy.return_value.squeeze.return_value = np.zeros((64, 3), dtype=np.float32)
                return out

        adapter.diffusion_model = MockDiffusionModel()
        adapter.use_simulated_backend = False
        adapter.is_loaded = True

        obs = self._create_test_observation(ModelType.ALPAMAYO, cmd=HighLevelCommand.LANE_FOLLOW)
        pred = adapter.infer(obs)
        self.assertEqual(pred.num_points, 64)
        self.assertIn("Real Diffusion", pred.reasoning)
        self.assertTrue(pred.perception_outputs.get("real_weights_executed"))

    def test_vad_real_weights_path(self):
        try:
            import torch
        except ImportError:
            from unittest.mock import MagicMock
            import sys
            mock_torch = MagicMock()
            mock_torch.from_numpy = lambda arr: MagicMock(to=lambda dev: arr)
            sys.modules["torch"] = mock_torch

        adapter = VADAdapter(device="cpu")
        class MockVADModel:
            def __call__(self, img, ego):
                from unittest.mock import MagicMock
                out = MagicMock()
                out.cpu.return_value.numpy.return_value.squeeze.return_value = np.zeros((30, 3), dtype=np.float32)
                return out

        adapter.model = MockVADModel()
        adapter.use_simulated_backend = False
        adapter.is_loaded = True

        obs = self._create_test_observation(ModelType.VAD, cmd=HighLevelCommand.LANE_FOLLOW)
        pred = adapter.infer(obs)
        self.assertEqual(pred.num_points, 30)
        self.assertIn("Real Vectorized", pred.reasoning)
        self.assertTrue(pred.perception_outputs.get("real_weights_executed"))


if __name__ == "__main__":
    unittest.main()
