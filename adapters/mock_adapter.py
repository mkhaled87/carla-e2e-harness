"""Mock E2E model adapter for unit testing, CI/CD, and lightweight simulation."""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np

from adapters.base_adapter import BaseE2EAdapter, plan_trajectory_with_lane_geometry
from e2e_schema import (
    CoordinateFrame,
    E2EObservation,
    E2EPrediction,
    HighLevelCommand,
    ModelType,
)


class MockE2EAdapter(BaseE2EAdapter):
    """Generates realistic kinematics-aware trajectories from observation context."""

    def __init__(
        self,
        default_target_speed_mps: float = 8.0,
        trajectory_duration_s: float = 5.0,
        trajectory_dt: float = 0.1,
    ) -> None:
        super().__init__(model_type=ModelType.MOCK, model_id="mock_e2e_v1")
        self.default_speed = default_target_speed_mps
        self.duration = trajectory_duration_s
        self.dt = trajectory_dt
        self.num_points = int(round(trajectory_duration_s / trajectory_dt))

    def load(self, **kwargs: Any) -> None:
        self.use_simulated_backend = True
        self.is_loaded = True
        print("[MockE2E NOTICE] Mock adapter initialized. Running with SIMULATED BACKEND (pure test harness).")

    def infer(self, observation: E2EObservation) -> E2EPrediction:
        t0 = time.perf_counter()
        speed = observation.target_speed_mps or self.default_speed
        cmd = observation.high_level_command
        prompt_text = (observation.text_prompt or "").lower()

        waypoints, timestamps, target_speeds, curvature, effective_speed = plan_trajectory_with_lane_geometry(
            observation=observation,
            num_points=self.num_points,
            dt=self.dt,
            target_speed=speed,
            is_iso_flu=False,
            lane_change_length_m=22.0,
        )

        if cmd == HighLevelCommand.TURN_LEFT:
            reason = "Executing protected left turn at junction; following left lane guidance."
        elif cmd == HighLevelCommand.TURN_RIGHT:
            reason = "Executing right turn; checking cross traffic and following curb."
        elif cmd == HighLevelCommand.LANE_CHANGE_LEFT:
            reason = "Initiating lane change left to prepare for upcoming navigation split."
        elif cmd == HighLevelCommand.LANE_CHANGE_RIGHT:
            reason = "Initiating lane change right into target driving lane."
        elif cmd == HighLevelCommand.STOP or "stop" in prompt_text:
            reason = "Red light detected or yield condition active. Planning smooth stop."
        elif abs(curvature) > 0.008:
            direction = "right" if curvature > 0 else "left"
            reason = f"Road curves {direction}. Maintaining lane center and cruising along curve."
        else:
            reason = "Driving lane clear. Maintaining lane center and cruising speed."

        latency = time.perf_counter() - t0

        return E2EPrediction(
            waypoints=waypoints,
            frame=CoordinateFrame.CARLA_LHD,
            timestamps=timestamps,
            target_speeds=target_speeds,
            direct_control={
                "target_speed_mps": effective_speed,
                "curvature": curvature,
            },
            reasoning=f"[SIMULATED BACKEND] {reason}",
            perception_outputs={
                "backend": "simulated",
                "real_weights_executed": False,
                "drivable_area_confidence": 0.98,
                "detected_objects_count": 0,
            },
            inference_seconds=latency,
            confidence=0.95,
            model_version="mock_e2e_v1 (SIMULATED BACKEND)",
        )

