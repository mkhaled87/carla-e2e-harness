#!/usr/bin/env python3
"""Closed-loop verification script for E2E AI Driving Models in CARLA.

Validates closed-loop driving for:
- QwenDrive 1.0 (VLA foundation model with Flow Matching planning)
- AutoE2E (Autoware compact latent BEV model)
- Alpamayo 2 Super (PhysicalAI-AV 7-camera diffusion model)
- VAD (Vectorized Autonomous Driving)

Usage:
    # Run mock/dry-run verification without live CARLA server:
    python3 test_e2e_closed_loop.py --model-type qwendrive_1.0 --dry-run
    python3 test_e2e_closed_loop.py --model-type autoe2e --dry-run
    python3 test_e2e_closed_loop.py --model-type alpamayo_2_super --dry-run
    python3 test_e2e_closed_loop.py --model-type vad --dry-run

    # Run closed-loop against a live CARLA server:
    python3 test_e2e_closed_loop.py --host 127.0.0.1 --port 2000 --model-type qwendrive_1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure local imports work
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from e2e_closed_loop_runner import ClosedLoopE2ERunner
from e2e_schema import HighLevelCommand, ModelType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_e2e_closed_loop")


class KinematicBicycleVehicle:
    """Mock ego vehicle for testing when live simulator is not connected."""

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = 0.0
        self.dt = 0.05
        self.wheelbase = 2.85

    def get_transform(self):
        class _Rot:
            def __init__(self, yaw): self.roll = 0.0; self.pitch = 0.0; self.yaw = yaw
        class _Loc:
            def __init__(self, x, y): self.x = x; self.y = y; self.z = 0.0
        class _Tf:
            def __init__(self, loc, rot): self.location = loc; self.rotation = rot
        return _Tf(_Loc(self.x, self.y), _Rot(self.yaw))

    def get_velocity(self):
        yaw_rad = math.radians(self.yaw)
        class _Vel:
            def __init__(self, vx, vy): self.x = vx; self.y = vy; self.z = 0.0
        return _Vel(self.speed * math.cos(yaw_rad), self.speed * math.sin(yaw_rad))

    def get_acceleration(self):
        class _Acc:
            def __init__(self): self.x = 0.0; self.y = 0.0; self.z = 0.0
        return _Acc()

    def get_angular_velocity(self):
        class _AngVel:
            def __init__(self): self.x = 0.0; self.y = 0.0; self.z = 0.0
        return _AngVel()

    def apply_control(self, ctrl):
        accel = ctrl.throttle * 3.0 - ctrl.brake * 6.0
        self.speed = max(0.0, self.speed + accel * self.dt)
        steer_rad = ctrl.steer * math.radians(40.0)
        yaw_rate = (self.speed / self.wheelbase) * math.tan(steer_rad)
        self.yaw += math.degrees(yaw_rate * self.dt)
        yaw_rad = math.radians(self.yaw)
        self.x += self.speed * math.cos(yaw_rad) * self.dt
        self.y += self.speed * math.sin(yaw_rad) * self.dt


class KinematicWorld:
    """Mock CARLA world for offline tests."""
    def get_blueprint_library(self):
        class _BpLib:
            def find(self, name):
                class _Bp:
                    def set_attribute(self, k, v): pass
                    def has_attribute(self, k): return True
                return _Bp()
        return _BpLib()

    def spawn_actor(self, bp, tf, attach_to=None):
        class _Actor:
            def __init__(self): self.is_listening = False
            def listen(self, cb): self.is_listening = True
            def stop(self): self.is_listening = False
            def destroy(self): pass
            def enable_for_ros(self): pass
        return _Actor()

    def tick(self):
        pass


def ensure_carla_available() -> Any | None:
    """Import carla, automatically finding and extracting bundled wheels if needed."""
    try:
        import carla
        return carla
    except ImportError:
        pass

    # Search for bundled .whl files across common paths or CARLA_ROOT
    search_dirs = [
        HERE.parent.parent.parent / "carla" / "dist",
        HERE.parent.parent.parent.parent / "PythonAPI" / "carla" / "dist",
        HERE.parent / "carla" / "Build" / "Package" / "Carla-0.10.0-Mac-Shipping" / "PythonAPI" / "carla" / "dist",
    ]
    if os.environ.get("CARLA_ROOT"):
        c_root = Path(os.environ["CARLA_ROOT"])
        search_dirs.extend([
            c_root / "PythonAPI" / "carla" / "dist",
            c_root / "dist",
        ])
    whl_files: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            whl_files.extend(d.glob("carla-*.whl"))

    if not whl_files:
        return None

    whl = whl_files[0]
    # Check Python ABI version from wheel tag (e.g. cp310 -> (3, 10))
    import re
    match = re.search(r"cp3(\d+)", whl.name)
    if match:
        expected_minor = int(match.group(1))
        if sys.version_info.major != 3 or sys.version_info.minor != expected_minor:
            logger.error(
                f"CARLA Python wheel mismatch: '{whl.name}' requires Python 3.{expected_minor}, "
                f"but current interpreter is Python {sys.version_info.major}.{sys.version_info.minor} ({sys.executable}).\n"
                f"Please run with Python 3.{expected_minor}: python3.{expected_minor} {sys.argv[0]}"
            )
            return None

    # Extract wheel to dist/extracted or cache dir
    import zipfile
    extracted_dir = whl.parent / "extracted"
    try:
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(whl, "r") as zf:
            zf.extractall(extracted_dir)
        if str(extracted_dir) not in sys.path:
            sys.path.insert(0, str(extracted_dir))
        import carla
        return carla
    except Exception as e:
        logger.warning(f"Failed to auto-extract wheel '{whl}': {e}")
        return None


def find_curved_spawn_point(world: Any, carla_map: Any, min_curve_deg: float = 35.0) -> tuple[int, Any]:
    """Find a map spawn point leading directly into a sharp curve (and ideally with multiple lanes)."""
    try:
        import carla
    except ImportError:
        carla = None

    spawn_points = carla_map.get_spawn_points()
    best_idx = 1 if len(spawn_points) > 1 else 0
    best_sp = spawn_points[best_idx]
    best_score = -1.0

    for idx, sp in enumerate(spawn_points):
        lane_type = carla.LaneType.Driving if carla else None
        wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=lane_type)
        if wp is None:
            continue
        curr = wp
        wps = [curr]
        for _ in range(35):
            nxt = curr.next(1.0)
            if not nxt:
                break
            curr = nxt[0]
            wps.append(curr)
        if len(wps) < 15:
            continue

        yaw_start = wps[0].transform.rotation.yaw
        yaw_end = wps[-1].transform.rotation.yaw
        diff = abs((yaw_end - yaw_start + 180.0) % 360.0 - 180.0)
        has_right = False
        try:
            r_lane = wp.get_right_lane()
            has_right = r_lane is not None and (carla is None or r_lane.lane_type == carla.LaneType.Driving)
        except Exception:
            pass

        score = diff + (50.0 if has_right else 0.0)
        if diff >= min_curve_deg and score > best_score:
            best_score = score
            best_idx = idx
            best_sp = sp

    return best_idx, best_sp


def run_live_simulation(args: argparse.Namespace) -> int:
    """Run closed-loop test against a live CARLA RPC server."""
    carla = ensure_carla_available()
    if carla is None:
        logger.error("CARLA Python package not found. Use --dry-run or run with matching Python 3.10 interpreter.")
        return 1

    logger.info(f"Connecting to CARLA Server at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Save original settings
    orig_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    world.apply_settings(settings)

    ego = None
    runner = None
    try:
        # Find hero/ego vehicle or spawn Lincoln MKZ
        spawned_ego = False
        actors = world.get_actors().filter("vehicle.*")
        for a in actors:
            if a.attributes.get("role_name") in ("hero", "ego"):
                if args.curve or args.spawn_point is not None or args.respawn:
                    logger.info(f"Destroying existing ego vehicle {a.id} to respawn at requested location...")
                    a.destroy()
                else:
                    ego = a
                    break

        if ego is None:
            bp_lib = world.get_blueprint_library()
            vehicle_bps = bp_lib.filter("vehicle.lincoln.mkz*")
            if not vehicle_bps:
                vehicle_bps = [
                    bp for bp in bp_lib.filter("vehicle.*")
                    if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels").as_int()) == 4
                ]
            ego_bp = vehicle_bps[0]
            ego_bp.set_attribute("role_name", "hero")

            carla_map = world.get_map()
            spawn_points = carla_map.get_spawn_points()
            if not spawn_points:
                raise RuntimeError("Map has no spawn points")

            chosen_sp = None
            if args.spawn_point is not None:
                sp_idx = args.spawn_point % len(spawn_points)
                chosen_sp = spawn_points[sp_idx]
                logger.info(f"Using explicitly specified spawn point {sp_idx}")
            elif args.curve:
                sp_idx, chosen_sp = find_curved_spawn_point(world, carla_map, min_curve_deg=35.0)
                logger.info(f"Selected curved spawn point {sp_idx}")

            if chosen_sp is not None:
                ego = world.try_spawn_actor(ego_bp, chosen_sp)
                if ego is not None:
                    spawned_ego = True

            if ego is None:
                for sp in spawn_points:
                    ego = world.try_spawn_actor(ego_bp, sp)
                    if ego is not None:
                        spawned_ego = True
                        break

            if ego is None:
                raise RuntimeError("Failed to spawn ego vehicle at any spawn point")
            logger.info(f"Spawned ego vehicle ({ego_bp.id}, id={ego.id})")

        runner = ClosedLoopE2ERunner(
            world=world,
            ego_vehicle=ego,
            model_type=ModelType(args.model_type),
            target_speed_mps=args.speed,
            fixed_delta_seconds=args.dt,
            replan_every_n_ticks=args.replan_ticks,
            use_remote_server=args.use_remote,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            telemetry_file=args.telemetry,
            enable_display=not args.no_display,
            spectator_follow=not args.no_spectator,
            spectator_view_mode=args.spectator_view,
            model_path=args.model_path,
            record_dir=args.record,
        )
        runner.setup()

        start_loc = ego.get_location()
        logger.info(f"Starting closed loop run ({args.ticks} ticks)...")

        backend_tag = "SIMULATED" if runner.is_simulated_backend else "REAL_WEIGHTS"
        for step in range(args.ticks):
            step_res = runner.run_step(HighLevelCommand(args.command))
            if step % 20 == 0:
                logger.info(
                    f"Step {step:03d} [{backend_tag}] | Speed: {step_res['speed_mps']:.2f} m/s | "
                    f"Throttle: {step_res.get('throttle', 0):.2f} | Steer: {step_res.get('steer', 0):.3f}"
                )

        end_loc = ego.get_location()
        displacement = math.sqrt(
            (end_loc.x - start_loc.x) ** 2 +
            (end_loc.y - start_loc.y) ** 2 +
            (end_loc.z - start_loc.z) ** 2
        )
        logger.info(f"Run completed. Total displacement: {displacement:.2f} m. Collisions: {runner.collision_count}")
        return 0

    finally:
        if runner:
            runner.cleanup()
        if spawned_ego and ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass
        world.apply_settings(orig_settings)


def run_dry_run_simulation(args: argparse.Namespace) -> int:
    """Run closed-loop test against mock kinematic world."""
    logger.info(f"[DRY-RUN] Testing closed-loop driving for model '{args.model_type}' ({args.ticks} ticks)...")
    world = KinematicWorld()
    ego = KinematicBicycleVehicle()

    runner = ClosedLoopE2ERunner(
        world=world,
        ego_vehicle=ego,
        model_type=ModelType(args.model_type),
        target_speed_mps=args.speed,
        fixed_delta_seconds=args.dt,
        replan_every_n_ticks=args.replan_ticks,
        use_remote_server=False,
        telemetry_file=args.telemetry,
        enable_display=not args.no_display,
        spectator_follow=False,
        model_path=args.model_path,
        record_dir=args.record,
    )
    runner.setup()

    backend_tag = "SIMULATED" if runner.is_simulated_backend else "REAL_WEIGHTS"
    for step in range(args.ticks):
        step_res = runner.run_step(HighLevelCommand(args.command))
        if step % 10 == 0:
            logger.info(
                f"[DRY-RUN:{backend_tag}] Step {step:03d} | Speed: {step_res['speed_mps']:.2f} m/s | "
                f"Displacement: {ego.x:.2f}m | Throttle: {step_res.get('throttle', 0):.2f} | "
                f"Steer: {step_res.get('steer', 0):.3f}"
            )

    runner.cleanup()
    logger.info(
        f"[DRY-RUN] Completed successfully! Final vehicle displacement: {ego.x:.2f} m, "
        f"Speed: {ego.speed:.2f} m/s."
    )
    assert ego.x > 0.5, "Vehicle failed to move forward in dry-run simulation."
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-Loop Verification for E2E Driving Models in CARLA")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port (default: 2000)")
    parser.add_argument(
        "--model-type",
        default=ModelType.QWEN_DRIVE.value,
        choices=[m.value for m in ModelType],
        help=f"E2E Model family to evaluate (default: {ModelType.QWEN_DRIVE.value})",
    )
    parser.add_argument("--model-path", default=None, help="Path to real model weights (.pt, .pth, .onnx, or HuggingFace ID)")
    parser.add_argument("--speed", type=float, default=7.0, help="Target cruising speed in m/s (default: 7.0)")
    parser.add_argument("--ticks", type=int, default=50, help="Simulation steps to execute (default: 50)")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation delta seconds (default: 0.05)")
    parser.add_argument("--replan-ticks", type=int, default=2, help="Ticks per inference replan (default: 2)")
    parser.add_argument(
        "--command",
        default=HighLevelCommand.LANE_FOLLOW.value,
        choices=[c.value for c in HighLevelCommand],
        help="High level navigation intent (default: lane_follow)",
    )
    parser.add_argument("--use-remote", action="store_true", help="Connect to remote model server")
    parser.add_argument("--remote-host", default="127.0.0.1", help="Remote model server host")
    parser.add_argument("--remote-port", type=int, default=8090, help="Remote model server port")
    parser.add_argument("--telemetry", default="/tmp/e2e_telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--no-display", action="store_true", help="Disable Pygame HUD window")
    parser.add_argument("--no-spectator", action="store_true", help="Disable following ego vehicle in main simulator window")
    parser.add_argument(
        "--spectator-view",
        default="chase",
        choices=["chase", "topdown"],
        help="Main simulator camera follow view mode: chase (elevated behind) or topdown (birds-eye view)",
    )
    parser.add_argument("--curve", action="store_true", help="Spawn ego vehicle at a curved road segment in the map (>35 deg turn)")
    parser.add_argument("--spawn-point", type=int, default=None, help="Explicit spawn point index to spawn ego vehicle")
    parser.add_argument("--respawn", action="store_true", help="Force destroying existing ego vehicle and respawning anew")
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="ARTIFACT_DIR",
        help="Record simulation artifact bundle (spectator video, camera streams, vehicle pose, and sensors) to specified directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run offline dry-run test without live CARLA server")
    args = parser.parse_args()

    if args.dry_run:
        return run_dry_run_simulation(args)

    # Check if live server is reachable
    import socket
    reachable = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect((args.host, args.port))
            reachable = True
        except Exception:
            reachable = False

    if reachable:
        return run_live_simulation(args)
    else:
        logger.error(
            f"\n{'='*75}\n"
            f"[ERROR] CARLA server is not running or unreachable at {args.host}:{args.port}!\n"
            f"Closed-loop E2E simulation requires an active CARLA server.\n"
            f"Please start the CARLA simulator first (e.g. ./CarlaUnreal.sh or ./CarlaUE4.sh).\n"
            f"If you explicitly intended to test offline without CARLA, pass: --dry-run\n"
            f"{'='*75}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

