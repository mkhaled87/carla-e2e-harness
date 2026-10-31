"""Coordinate conversions and rigid body transformations for E2E models in CARLA.

Handles conversions between:
1. CARLA Local / Actor Frame (Left-handed, +X Forward, +Y Right, +Z Up, Yaw +Right)
2. ISO 8855 / ROS 2 / FLU Frame (Right-handed, +X Forward, +Y Left, +Z Up, Yaw +Left)
3. Vehicle Center vs Rear-Axle (base_link) reference frames.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple
import numpy as np

from e2e_schema import CoordinateFrame


def carla_to_iso_points(points: np.ndarray) -> np.ndarray:
    """Convert points from CARLA left-handed (+X fwd, +Y right, +Z up)
    to ISO 8855 right-handed (+X fwd, +Y left, +Z up).
    
    Accepts shape (N, 2) or (N, 3).
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts[:, 1] = -pts[:, 1]
    return pts


def iso_to_carla_points(points: np.ndarray) -> np.ndarray:
    """Convert points from ISO 8855 right-handed (+X fwd, +Y left, +Z up)
    to CARLA left-handed (+X fwd, +Y right, +Z up).
    
    Accepts shape (N, 2) or (N, 3).
    """
    return carla_to_iso_points(points)


def carla_to_iso_yaw(yaw_deg: float) -> float:
    """Convert yaw angle from CARLA (deg, positive right) to ISO 8855 (deg, positive left)."""
    return -yaw_deg


def iso_to_carla_yaw(yaw_deg: float) -> float:
    """Convert yaw angle from ISO 8855 (deg, positive left) to CARLA (deg, positive right)."""
    return -yaw_deg


def carla_to_iso_yaw_rad(yaw_rad: float) -> float:
    """Convert yaw angle in radians."""
    return -yaw_rad


def iso_to_carla_yaw_rad(yaw_rad: float) -> float:
    """Convert yaw angle in radians."""
    return -yaw_rad


def rear_axle_to_center_offset(points: np.ndarray, rear_axle_x: float) -> np.ndarray:
    """Shift points defined relative to rear axle into vehicle center frame.
    
    In CARLA MKZ: rear_axle_x = -1.425 m (behind vehicle center).
    So x_center = x_rear_axle + rear_axle_x.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] += rear_axle_x
    return pts


def center_to_rear_axle_offset(points: np.ndarray, rear_axle_x: float) -> np.ndarray:
    """Shift points defined relative to vehicle center into rear axle frame."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] -= rear_axle_x
    return pts


def rotation_matrix_2d(yaw_rad: float) -> np.ndarray:
    """2D rotation matrix for yaw in radians."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def local_to_world_2d(
    local_pts: np.ndarray,
    origin_x: float,
    origin_y: float,
    origin_yaw_deg: float,
    is_carla_frame: bool = True,
) -> np.ndarray:
    """Transform 2D points from vehicle local coordinates to world coordinates.
    
    If is_carla_frame is True, yaw rotates clockwise (+deg is to the right).
    """
    pts = np.asarray(local_pts, dtype=np.float32)[:, :2]
    yaw_rad = math.radians(origin_yaw_deg)
    if is_carla_frame:
        # In CARLA left-handed:
        # x_w = x_0 + x_l * cos(yaw) - y_l * sin(yaw)
        # y_w = y_0 + x_l * sin(yaw) + y_l * cos(yaw)
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
    else:
        # Standard right-handed Cartesian rotation
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

    world_pts = pts @ R.T
    world_pts[:, 0] += origin_x
    world_pts[:, 1] += origin_y
    return world_pts


def world_to_local_2d(
    world_pts: np.ndarray,
    origin_x: float,
    origin_y: float,
    origin_yaw_deg: float,
    is_carla_frame: bool = True,
) -> np.ndarray:
    """Transform 2D points from world coordinates to vehicle local coordinates."""
    pts = np.asarray(world_pts, dtype=np.float32)[:, :2].copy()
    pts[:, 0] -= origin_x
    pts[:, 1] -= origin_y

    yaw_rad = math.radians(origin_yaw_deg)
    if is_carla_frame:
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        R = np.array([[c, s], [-s, c]], dtype=np.float32)
    else:
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        R = np.array([[c, s], [-s, c]], dtype=np.float32)

    return pts @ R.T


def normalize_trajectory_to_carla(
    points: np.ndarray,
    source_frame: CoordinateFrame,
    rear_axle_relative: bool = False,
    rear_axle_x: float = -1.425,
) -> np.ndarray:
    """Normalize arbitrary model trajectory output to CARLA vehicle center coordinates."""
    pts = np.asarray(points, dtype=np.float32).copy()
    if source_frame == CoordinateFrame.ISO_8855_FLU:
        pts = iso_to_carla_points(pts)
    if rear_axle_relative:
        pts = rear_axle_to_center_offset(pts, rear_axle_x)
    return pts

