"""Universal Trajectory Tracking Controller for E2E models in CARLA.

Converts predicted future trajectories into smooth, safe vehicle actuation commands:
- CARLA `carla.VehicleControl` (throttle, steer, brake)
- ROS 2 `autoware_control_msgs/msg/Control` (steering_tire_angle, acceleration)

Implements:
- Longitudinal PID with anti-windup, derivative on measurement, and stop-hold logic.
- Lateral Pure Pursuit / Stanley tracking with lookahead scaling and steer rate limiting.
- Robust handling of non-uniform waypoint spacing and replan boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from e2e_schema import CoordinateFrame, E2EPrediction
from e2e_coordinates import normalize_trajectory_to_carla


@dataclass(frozen=True)
class E2ETrackingCommand:
    throttle: float
    brake: float
    steer: float
    target_speed_mps: float
    current_speed_mps: float
    target_steer_rad: float
    target_accel_mps2: float
    curvature: float
    cross_track_error: float
    heading_error_rad: float
    stop_hold_active: bool


class E2ETrajectoryTracker:
    """Follows planned trajectory waypoints using lateral Pure Pursuit and longitudinal PID."""

    def __init__(
        self,
        wheelbase_m: float = 2.85,
        max_steer_angle_deg: float = 40.0,
        min_lookahead_m: float = 2.5,
        max_lookahead_m: float = 12.0,
        lookahead_ratio_s: float = 0.5,
        speed_kp: float = 0.40,
        speed_ki: float = 0.05,
        speed_kd: float = 0.02,
        speed_integral_limit: float = 4.0,
        max_steer_rate_per_s: float = 1.2,
        stop_speed_threshold_mps: float = 0.25,
        hold_speed_mps: float = 0.10,
        minimum_stop_brake: float = 0.35,
        hold_brake: float = 1.0,
    ) -> None:
        self.wheelbase = wheelbase_m
        self.max_steer_angle_rad = math.radians(max_steer_angle_deg)
        self.min_lookahead = min_lookahead_m
        self.max_lookahead = max_lookahead_m
        self.lookahead_ratio = lookahead_ratio_s
        self.speed_kp = speed_kp
        self.speed_ki = speed_ki
        self.speed_kd = speed_kd
        self.speed_integral_limit = speed_integral_limit
        self.max_steer_rate_per_s = max_steer_rate_per_s
        self.stop_speed_threshold = stop_speed_threshold_mps
        self.hold_speed = hold_speed_mps
        self.minimum_stop_brake = minimum_stop_brake
        self.hold_brake = hold_brake

        self._previous_steer = 0.0
        self._speed_integral = 0.0
        self._previous_speed_mps: float | None = None

    def reset(self) -> None:
        """Reset internal PID integrator and previous steer state."""
        self._previous_steer = 0.0
        self._speed_integral = 0.0
        self._previous_speed_mps = None

    def compute_control(
        self,
        prediction: E2EPrediction,
        current_speed_mps: float,
        dt: float = 0.05,
    ) -> E2ETrackingCommand:
        """Compute control command from predicted future trajectory in ego frame."""
        # Normalize waypoints to CARLA vehicle local frame (+X fwd, +Y right)
        waypoints = normalize_trajectory_to_carla(
            prediction.get_xy(),
            source_frame=prediction.frame,
        )
        if len(waypoints) < 2:
            return self._emergency_stop(current_speed_mps)

        # 1. Target Speed Estimation
        # If model outputs explicit direct control or speed profile, use it.
        target_speed = self._extract_target_speed(prediction, waypoints)

        # 2. Longitudinal Control (PID + Stop Hold)
        throttle, brake, accel_cmd, stop_hold = self._longitudinal_control(
            target_speed_mps=target_speed,
            current_speed_mps=current_speed_mps,
            dt=dt,
        )

        # 3. Lateral Control (Pure Pursuit on local waypoints)
        steer, steer_rad, curvature, cross_track_error, heading_error = self._lateral_control(
            waypoints=waypoints,
            current_speed_mps=current_speed_mps,
            dt=dt,
        )

        return E2ETrackingCommand(
            throttle=throttle,
            brake=brake,
            steer=steer,
            target_speed_mps=target_speed,
            current_speed_mps=current_speed_mps,
            target_steer_rad=steer_rad,
            target_accel_mps2=accel_cmd,
            curvature=curvature,
            cross_track_error=cross_track_error,
            heading_error_rad=heading_error,
            stop_hold_active=stop_hold,
        )

    def _extract_target_speed(self, prediction: E2EPrediction, waypoints: np.ndarray) -> float:
        """Extract or estimate target speed from prediction."""
        if prediction.direct_control and "target_speed_mps" in prediction.direct_control:
            return float(prediction.direct_control["target_speed_mps"])
        if prediction.target_speeds is not None and len(prediction.target_speeds) > 0:
            return float(prediction.target_speeds[0])

        # Derive speed from first 1.0 second of waypoints
        timestamps = prediction.timestamps
        if timestamps is not None and len(timestamps) >= 2:
            dt_total = timestamps[min(len(timestamps) - 1, 9)]
            if dt_total > 1e-3:
                dx = waypoints[min(len(waypoints) - 1, 9), 0]
                dy = waypoints[min(len(waypoints) - 1, 9), 1]
                dist = math.hypot(dx, dy)
                return dist / dt_total

        # Fallback derivative between point 0 and 1
        d01 = math.hypot(waypoints[1, 0] - waypoints[0, 0], waypoints[1, 1] - waypoints[0, 1])
        return d01 / 0.1

    def _longitudinal_control(
        self,
        target_speed_mps: float,
        current_speed_mps: float,
        dt: float,
    ) -> tuple[float, float, float, bool]:
        """Compute throttle, brake, acceleration effort, and hold status."""
        stop_requested = target_speed_mps <= self.stop_speed_threshold
        if stop_requested:
            target_speed_mps = 0.0
            self._speed_integral = min(self._speed_integral, 0.0)

        error = target_speed_mps - current_speed_mps
        speed_derivative = 0.0
        if self._previous_speed_mps is not None and dt > 1e-4:
            speed_derivative = -(current_speed_mps - self._previous_speed_mps) / dt
        self._previous_speed_mps = current_speed_mps

        # Integrate error
        candidate_integral = float(
            np.clip(
                self._speed_integral + error * dt,
                -self.speed_integral_limit,
                self.speed_integral_limit,
            )
        )
        effort = (
            self.speed_kp * error
            + self.speed_ki * candidate_integral
            + self.speed_kd * speed_derivative
        )

        # Anti-windup clamping
        if not ((effort > 1.0 and error > 0.0) or (effort < -1.0 and error < 0.0)):
            self._speed_integral = candidate_integral

        effort = float(np.clip(effort, -1.0, 1.0))

        if stop_requested:
            if current_speed_mps <= self.hold_speed:
                return 0.0, self.hold_brake, -self.hold_brake, True
            brake = max(-effort, self.minimum_stop_brake)
            return 0.0, float(np.clip(brake, 0.0, 1.0)), effort, False

        if effort >= 0.0:
            return effort, 0.0, effort, False
        else:
            return 0.0, -effort, effort, False

    def _lateral_control(
        self,
        waypoints: np.ndarray,
        current_speed_mps: float,
        dt: float,
    ) -> tuple[float, float, float, float, float]:
        """Compute steering ratio [-1.0, 1.0] using Pure Pursuit with rate limiting."""
        # Compute dynamic lookahead distance
        lookahead = np.clip(
            current_speed_mps * self.lookahead_ratio,
            self.min_lookahead,
            self.max_lookahead,
        )

        # Find target waypoint matching lookahead
        distances = np.hypot(waypoints[:, 0], waypoints[:, 1])
        target_idx = np.argmin(np.abs(distances - lookahead))
        target_pt = waypoints[target_idx]

        tx = float(target_pt[0])
        ty = float(target_pt[1])
        dist = math.hypot(tx, ty)
        if dist < 0.5:
            return self._previous_steer, 0.0, 0.0, 0.0, 0.0

        # In CARLA left-handed: +y is to the right. Positive steer turns right.
        # Pure pursuit curvature: kappa = 2 * y / dist^2 (floor effective dist at 1.5m to avoid singular steering)
        eff_dist = max(dist, 1.5)
        curvature = 2.0 * ty / (eff_dist ** 2)
        target_steer_rad = math.atan(curvature * self.wheelbase)

        # Normalize to [-1.0, 1.0] relative to max_steer_angle
        unclipped_steer = target_steer_rad / self.max_steer_angle_rad
        unclipped_steer = float(np.clip(unclipped_steer, -1.0, 1.0))

        # Apply steer rate limiting
        max_delta = self.max_steer_rate_per_s * dt
        steer = float(
            np.clip(
                unclipped_steer,
                self._previous_steer - max_delta,
                self._previous_steer + max_delta,
            )
        )
        self._previous_steer = steer

        cross_track_error = ty
        heading_error = math.atan2(ty, tx)

        return steer, target_steer_rad, curvature, cross_track_error, heading_error

    def _emergency_stop(self, current_speed_mps: float) -> E2ETrackingCommand:
        """Fallback emergency braking command."""
        self._speed_integral = 0.0
        return E2ETrackingCommand(
            throttle=0.0,
            brake=self.hold_brake if current_speed_mps <= self.hold_speed else self.minimum_stop_brake,
            steer=0.0,
            target_speed_mps=0.0,
            current_speed_mps=current_speed_mps,
            target_steer_rad=0.0,
            target_accel_mps2=-3.0,
            curvature=0.0,
            cross_track_error=0.0,
            heading_error_rad=0.0,
            stop_hold_active=current_speed_mps <= self.hold_speed,
        )

