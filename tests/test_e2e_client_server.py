"""Integration test for E2E Model Server and Client round-trip."""

from __future__ import annotations

import socket
import threading
import time
import unittest
import numpy as np

from e2e_model_server import pack_prediction, serve_model, unpack_observation
from e2e_protocol import (
    connect_e2e_service,
    recv_protocol_message,
    send_protocol_message,
)
from e2e_rigs import build_rig_specs
from e2e_schema import (
    CoordinateFrame,
    E2EObservation,
    EgoKinematicState,
    HighLevelCommand,
    ModelType,
)


class TestE2EClientServer(unittest.TestCase):
    def test_model_server_remote_inference(self):
        # Pick an ephemeral port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        # Start server in thread
        server_th = threading.Thread(
            target=serve_model,
            args=(ModelType.MOCK, "127.0.0.1", port),
            daemon=True,
        )
        server_th.start()
        time.sleep(0.1)

        # Connect client
        client_sock = connect_e2e_service("127.0.0.1", port)

        # Test PING
        send_protocol_message(client_sock, "PING")
        pong = recv_protocol_message(client_sock)
        self.assertEqual(pong.kind, "PONG")
        self.assertEqual(pong.metadata["status"], "ok")

        # Test INFER
        specs = build_rig_specs(ModelType.MOCK, 64, 36)
        cam_images = {k: np.zeros((36, 64, 3), dtype=np.uint8) for k in specs}
        state = EgoKinematicState(
            timestamp_us=2000000,
            x=5.0, y=10.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            vx=7.0, vy=0.0, vz=0.0,
            speed_mps=7.0,
        )

        obs = E2EObservation(
            model_type=ModelType.MOCK,
            frame_id=10,
            timestamp_us=2000000,
            camera_images=cam_images,
            camera_specs=specs,
            ego_state=state,
            high_level_command=HighLevelCommand.LANE_FOLLOW,
            text_prompt="Test prompt",
            target_speed_mps=8.0,
        )

        from dataclasses import asdict
        meta = {
            "model_type": obs.model_type.value,
            "frame_id": obs.frame_id,
            "timestamp_us": obs.timestamp_us,
            "ego_state": obs.ego_state.to_dict(),
            "camera_specs": {
                k: {"intrinsics": asdict(v.intrinsics), "extrinsics": asdict(v.extrinsics)}
                for k, v in specs.items()
            },
            "high_level_command": obs.high_level_command.value,
            "text_prompt": obs.text_prompt,
            "target_speed_mps": obs.target_speed_mps,
        }
        arrays = {f"cam_{k}": v for k, v in cam_images.items()}

        send_protocol_message(client_sock, "INFER", metadata=meta, arrays=arrays)
        resp = recv_protocol_message(client_sock)

        self.assertEqual(resp.kind, "PREDICTION")
        self.assertIn("waypoints", resp.arrays)
        self.assertEqual(resp.arrays["waypoints"].shape[1], 3)
        self.assertIn("cruising", resp.metadata["reasoning"].lower())

        client_sock.close()


if __name__ == "__main__":
    unittest.main()

