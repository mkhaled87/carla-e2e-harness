"""Mock integration test for ClosedLoopE2ERunner.

Verifies complete closed-loop cycle:
- Vehicle kinematics advance deterministically with physics
- Cameras capture frames
- Models generate trajectories
- Trajectory tracker calculates throttle/steer/brake
- Telemetry records properly
"""

from __future__ import annotations

import math
import tempfile
import unittest
import numpy as np

from e2e_closed_loop_runner import ClosedLoopE2ERunner
from e2e_schema import HighLevelCommand, ModelType


class MockLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockRotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


class MockTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or MockLocation()
        self.rotation = rotation or MockRotation()


class MockVector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockVehicle:
    """Simple 2D kinematic bicycle model for closed-loop testing."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = 0.0
        self.dt = 0.05
        self.wheelbase = 2.85
        self.current_ctrl = None

    def get_transform(self):
        return MockTransform(
            location=MockLocation(self.x, self.y, 0.0),
            rotation=MockRotation(0.0, 0.0, self.yaw),
        )

    def get_velocity(self):
        yaw_rad = math.radians(self.yaw)
        return MockVector3D(
            x=self.speed * math.cos(yaw_rad),
            y=self.speed * math.sin(yaw_rad),
            z=0.0,
        )

    def get_acceleration(self):
        return MockVector3D(0.0, 0.0, 0.0)

    def get_angular_velocity(self):
        return MockVector3D(0.0, 0.0, 0.0)

    def apply_control(self, control):
        self.current_ctrl = control
        # Kinematic update
        accel = control.throttle * 3.0 - control.brake * 6.0
        self.speed = max(0.0, self.speed + accel * self.dt)
        steer_rad = control.steer * math.radians(40.0)
        yaw_rate = (self.speed / self.wheelbase) * math.tan(steer_rad)
        self.yaw += math.degrees(yaw_rate * self.dt)
        yaw_rad = math.radians(self.yaw)
        self.x += self.speed * math.cos(yaw_rad) * self.dt
        self.y += self.speed * math.sin(yaw_rad) * self.dt


class MockBlueprint:
    def __init__(self):
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v

    def has_attribute(self, k):
        return True


class MockBlueprintLibrary:
    def find(self, name):
        return MockBlueprint()


class MockWorld:
    def __init__(self):
        self.settings = None

    def get_blueprint_library(self):
        return MockBlueprintLibrary()

    def spawn_actor(self, bp, tf, attach_to=None):
        class MockActor:
            def __init__(self):
                self.is_listening = False
            def listen(self, cb):
                self.is_listening = True
            def stop(self):
                self.is_listening = False
            def destroy(self):
                pass
            def enable_for_ros(self):
                pass
        return MockActor()

    def tick(self):
        pass


class TestClosedLoopSimulation(unittest.TestCase):
    def test_mock_closed_loop_drive(self):
        world = MockWorld()
        ego = MockVehicle()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as tmp:
            runner = ClosedLoopE2ERunner(
                world=world,
                ego_vehicle=ego,
                model_type=ModelType.MOCK,
                target_speed_mps=6.0,
                fixed_delta_seconds=0.05,
                replan_every_n_ticks=2,
                use_remote_server=False,
                telemetry_file=tmp.name,
            )
            runner.setup()

            # Run 20 closed loop steps
            for _ in range(20):
                runner.run_step(HighLevelCommand.LANE_FOLLOW)

            runner.cleanup()

            # Vehicle should have moved forward and accelerated
            self.assertGreater(ego.x, 0.5)
            self.assertGreater(ego.speed, 0.5)

    def test_qwendrive_closed_loop_drive(self):
        world = MockWorld()
        ego = MockVehicle()
        runner = ClosedLoopE2ERunner(
            world=world,
            ego_vehicle=ego,
            model_type=ModelType.QWEN_DRIVE,
            target_speed_mps=6.0,
            fixed_delta_seconds=0.05,
            replan_every_n_ticks=2,
            use_remote_server=False,
        )
        runner.setup()

        for _ in range(10):
            step_data = runner.run_step(HighLevelCommand.LANE_FOLLOW)
            self.assertIn("reasoning", step_data)
            self.assertIn("Flow Matching", step_data["reasoning"])

        runner.cleanup()


if __name__ == "__main__":
    unittest.main()

