"""Unit tests for binary TCP protocol."""

from __future__ import annotations

import socket
import threading
import unittest
import numpy as np

from e2e_protocol import (
    E2EProtocolError,
    ProtocolMessage,
    recv_protocol_message,
    send_protocol_message,
)


class TestE2EProtocol(unittest.TestCase):
    def test_protocol_socket_transmission(self):
        # Create listening server and client sockets
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        received_msg = []

        def _server_thread():
            conn, _ = server.accept()
            msg = recv_protocol_message(conn)
            received_msg.append(msg)
            # Reply
            send_protocol_message(conn, "ACK", metadata={"status": "ok"})
            conn.close()

        th = threading.Thread(target=_server_thread)
        th.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))

        img = np.arange(100 * 50 * 3, dtype=np.uint8).reshape((50, 100, 3))
        waypoints = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        send_protocol_message(
            client,
            kind="INFER",
            metadata={"model": "test", "step": 42},
            arrays={"image": img, "waypoints": waypoints},
        )

        ack = recv_protocol_message(client)
        client.close()
        th.join()
        server.close()

        self.assertEqual(ack.kind, "ACK")
        self.assertEqual(ack.metadata["status"], "ok")

        self.assertEqual(len(received_msg), 1)
        msg = received_msg[0]
        self.assertEqual(msg.kind, "INFER")
        self.assertEqual(msg.metadata["step"], 42)
        self.assertIn("image", msg.arrays)
        self.assertIn("waypoints", msg.arrays)
        np.testing.assert_array_equal(msg.arrays["image"], img)
        np.testing.assert_array_equal(msg.arrays["waypoints"], waypoints)


if __name__ == "__main__":
    unittest.main()

