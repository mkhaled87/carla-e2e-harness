"""Persistent E2E Model Server for isolated inference in CARLA simulations.

Runs model execution (QwenDrive, AutoE2E, Alpamayo, VAD, Mock) in a dedicated process
over the binary TCP protocol, decoupling simulation physics from inference latency.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from typing import Any
import numpy as np

from adapters.alpamayo_adapter import AlpamayoAdapter
from adapters.autoe2e_adapter import AutoE2EAdapter
from adapters.base_adapter import BaseE2EAdapter
from adapters.mock_adapter import MockE2EAdapter
from adapters.qwendrive_adapter import QwenDriveAdapter
from adapters.vad_adapter import VADAdapter
from e2e_protocol import (
    E2EProtocolError,
    ProtocolMessage,
    recv_protocol_message,
    send_protocol_message,
)
from e2e_schema import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraSpec,
    CoordinateFrame,
    E2EObservation,
    E2EPrediction,
    EgoKinematicState,
    HighLevelCommand,
    ModelType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_model_server")


def create_adapter(model_type: ModelType, model_path: str | None = None, **kwargs: Any) -> BaseE2EAdapter:
    """Instantiate the matching model adapter."""
    if model_type == ModelType.QWEN_DRIVE:
        adapter = QwenDriveAdapter(model_id=model_path or "Qwen/Qwen-Drive-1.0-4B", **kwargs)
    elif model_type == ModelType.AUTO_E2E:
        adapter = AutoE2EAdapter(model_path=model_path, **kwargs)
    elif model_type == ModelType.ALPAMAYO:
        adapter = AlpamayoAdapter(model_path=model_path, **kwargs)
    elif model_type == ModelType.VAD:
        adapter = VADAdapter(model_path=model_path, **kwargs)
    else:
        adapter = MockE2EAdapter(**kwargs)
    adapter.load()
    return adapter


def unpack_observation(msg: ProtocolMessage) -> E2EObservation:
    """Convert ProtocolMessage into strongly-typed E2EObservation."""
    meta = msg.metadata
    ego_meta = meta["ego_state"]
    ego_state = EgoKinematicState(
        timestamp_us=int(ego_meta.get("timestamp_us", 0)),
        x=float(ego_meta["x"]),
        y=float(ego_meta["y"]),
        z=float(ego_meta["z"]),
        roll=float(ego_meta["roll"]),
        pitch=float(ego_meta["pitch"]),
        yaw=float(ego_meta["yaw"]),
        vx=float(ego_meta.get("vx", 0.0)),
        vy=float(ego_meta.get("vy", 0.0)),
        vz=float(ego_meta.get("vz", 0.0)),
        speed_mps=float(ego_meta["speed_mps"]),
        ax=float(ego_meta.get("ax", 0.0)),
        ay=float(ego_meta.get("ay", 0.0)),
        az=float(ego_meta.get("az", 0.0)),
        wx=float(ego_meta.get("wx", 0.0)),
        wy=float(ego_meta.get("wy", 0.0)),
        wz=float(ego_meta.get("wz", 0.0)),
        frame=CoordinateFrame(ego_meta.get("frame", CoordinateFrame.CARLA_LHD.value)),
    )

    specs: dict[str, CameraSpec] = {}
    for name, spec_data in meta.get("camera_specs", {}).items():
        intr = spec_data["intrinsics"]
        extr = spec_data["extrinsics"]
        specs[name] = CameraSpec(
            name=name,
            intrinsics=CameraIntrinsics(
                width=int(intr["width"]),
                height=int(intr["height"]),
                fov=float(intr["fov"]),
                fx=float(intr["fx"]),
                fy=float(intr["fy"]),
                cx=float(intr["cx"]),
                cy=float(intr["cy"]),
            ),
            extrinsics=CameraExtrinsics(
                x=float(extr["x"]),
                y=float(extr["y"]),
                z=float(extr["z"]),
                roll=float(extr.get("roll", 0.0)),
                pitch=float(extr.get("pitch", 0.0)),
                yaw=float(extr.get("yaw", 0.0)),
                frame=CoordinateFrame(extr.get("frame", CoordinateFrame.CARLA_LHD.value)),
            ),
        )

    # Reconstruct images from message arrays
    images: dict[str, np.ndarray] = {}
    for name in specs:
        arr_key = f"cam_{name}"
        if arr_key in msg.arrays:
            images[name] = msg.arrays[arr_key]

    return E2EObservation(
        model_type=ModelType(meta.get("model_type", ModelType.MOCK.value)),
        frame_id=int(meta.get("frame_id", 0)),
        timestamp_us=int(meta.get("timestamp_us", 0)),
        camera_images=images,
        camera_specs=specs,
        ego_state=ego_state,
        high_level_command=HighLevelCommand(meta.get("high_level_command", HighLevelCommand.LANE_FOLLOW.value)),
        text_prompt=str(meta.get("text_prompt", "")),
        target_speed_mps=meta.get("target_speed_mps"),
        custom_metadata=meta.get("custom_metadata", {}),
    )


def pack_prediction(prediction: E2EPrediction) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Convert E2EPrediction into protocol metadata and arrays."""
    arrays: dict[str, np.ndarray] = {
        "waypoints": prediction.waypoints,
    }
    if prediction.timestamps is not None:
        arrays["timestamps"] = prediction.timestamps
    if prediction.target_speeds is not None:
        arrays["target_speeds"] = prediction.target_speeds

    meta = {
        "frame": prediction.frame.value,
        "reasoning": prediction.reasoning,
        "direct_control": prediction.direct_control or {},
        "perception_outputs": prediction.perception_outputs or {},
        "inference_seconds": prediction.inference_seconds,
        "confidence": prediction.confidence,
        "model_version": prediction.model_version,
    }
    return meta, arrays


def serve_model(
    model_type: ModelType,
    host: str = "0.0.0.0",
    port: int = 8090,
    **adapter_kwargs: Any,
) -> None:
    """Run server socket loop handling inference requests."""
    logger.info(f"Starting E2E Model Server ({model_type.value}) on {host}:{port}...")
    adapter = create_adapter(model_type, **adapter_kwargs)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    logger.info("E2E Model Server ready and listening.")

    try:
        while True:
            conn, client_addr = server_sock.accept()
            logger.info(f"Accepted client connection from {client_addr}")
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        msg = recv_protocol_message(conn)
                    except (EOFError, ConnectionResetError):
                        logger.info(f"Client {client_addr} disconnected.")
                        break

                    if msg.kind == "PING":
                        send_protocol_message(conn, "PONG", metadata={"status": "ok", "model": model_type.value})
                        continue

                    if msg.kind == "INFER":
                        try:
                            obs = unpack_observation(msg)
                            pred = adapter.infer(obs)
                            pred_meta, pred_arrays = pack_prediction(pred)
                            send_protocol_message(conn, "PREDICTION", metadata=pred_meta, arrays=pred_arrays)
                        except Exception as e:
                            logger.error(f"Inference error: {traceback.format_exc()}")
                            send_protocol_message(conn, "ERROR", metadata={"error": str(e)})
                    else:
                        send_protocol_message(conn, "ERROR", metadata={"error": f"Unknown message kind: {msg.kind}"})
            finally:
                conn.close()
    finally:
        server_sock.close()
        adapter.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA E2E Model Server")
    parser.add_argument("--model-type", default=ModelType.MOCK.value, choices=[m.value for m in ModelType],
                        help="Model family to host (default: mock_e2e)")
    parser.add_argument("--model-path", default=None, help="Path to real model checkpoint (.pt, .pth, .onnx, or HuggingFace ID)")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Listen port (default: 8090)")
    parser.add_argument("--device", default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    serve_model(
        model_type=ModelType(args.model_type),
        host=args.host,
        port=args.port,
        device=args.device,
        model_path=args.model_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

