"""ROS 2 Bridge and Integration Module for E2E Models in CARLA.

Enables models running inside ROS 2 environments (e.g. Autoware Universe E2E,
TensorRT-VAD node, or AutoE2E ROS 2 stack) to communicate directly with CARLA:
- Configures CycloneDDS unicast bridge between host and Docker containers.
- Subscribes to Autoware control commands (/control/command/control_cmd).
- Subscribes to planned E2E trajectories (/planning/trajectory), tracking them via E2ETrajectoryTracker.
- Synchronizes with CARLA /clock.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_ros2_bridge")


def detect_host_ip() -> str:
    """Detect LAN IPv4 address for CycloneDDS peer configuration."""
    if platform.system() == "Darwin":
        try:
            route_out = subprocess.check_output(["route", "-n", "get", "default"], text=True)
            iface = None
            for line in route_out.splitlines():
                if line.strip().startswith("interface:"):
                    iface = line.strip().split(":", 1)[1].strip()
                    break
            if iface:
                ip = subprocess.check_output(["ipconfig", "getifaddr", iface], text=True).strip()
                if ip:
                    return ip
        except Exception:
            pass

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


def generate_cyclonedds_xml(host_ip: str, host_path: Path, container_path: Path) -> None:
    """Write CycloneDDS XML configurations for host and container communication."""
    host_cfg = f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="{host_ip}" multicast="false"/>
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>0</ParticipantIndex>
      <MaxAutoParticipantIndex>80</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="{host_ip}"/>
        <Peer address="127.0.0.1"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""
    container_cfg = f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="eth0" multicast="false"/>
      </Interfaces>
      <ExternalNetworkAddress>127.0.0.1</ExternalNetworkAddress>
      <AllowMulticast>false</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>80</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="{host_ip}"/>
        <Peer address="127.0.0.1"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""
    host_path.write_text(host_cfg, encoding="utf-8")
    container_path.write_text(container_cfg, encoding="utf-8")
    logger.info(f"Generated CycloneDDS configs at {host_path} and {container_path}")


def spawn_ros2_e2e_rig(world: Any, ego_vehicle: Any) -> list[Any]:
    """Spawn the 6 surround cameras with native ROS 2 publishing enabled."""
    import carla
    from e2e_rigs import NUSCENES_CAMERAS

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")

    sensors = []
    for name, fov, (x, y, z, roll, pitch, yaw) in NUSCENES_CAMERAS:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "800")
        bp.set_attribute("image_size_y", "450")
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.1")
        bp.set_attribute("role_name", name)
        if bp.has_attribute("ros_name"):
            bp.set_attribute("ros_name", name)
        if bp.has_attribute("ros_topic_name"):
            bp.set_attribute("ros_topic_name", f"/sensing/camera/{name}")

        tf = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(roll=roll, pitch=pitch, yaw=yaw),
        )
        sensor = world.spawn_actor(bp, tf, attach_to=ego_vehicle)
        if hasattr(sensor, "enable_for_ros"):
            sensor.enable_for_ros()
        sensors.append(sensor)

    # Autoware vehicle status sensor
    status_bp = bp_lib.find("sensor.other.vehicle_status")
    status_bp.set_attribute("ros_name", "vehicle_status")
    status_bp.set_attribute("role_name", "vehicle_status")
    status_bp.set_attribute("sensor_tick", "0.05")
    status_sensor = world.spawn_actor(status_bp, carla.Transform(), attach_to=ego_vehicle)
    if hasattr(status_sensor, "enable_for_ros"):
        status_sensor.enable_for_ros()
    sensors.append(status_sensor)

    logger.info(f"Spawned {len(sensors)} native ROS 2 sensors for E2E surround rig.")
    return sensors

