"""High-performance, versioned binary TCP protocol for decoupled E2E inference.

Enables running large foundation models (QwenDrive, Alpamayo, AutoE2E, VAD)
in isolated Python environments, Docker containers, or remote GPU clusters
without serializing via insecure or slow Python pickles.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import struct
from typing import Any, Mapping
import numpy as np


MAGIC = b"E2EP"
PROTOCOL_VERSION = 1
MAX_HEADER_SIZE = 4 * 1024 * 1024       # 4 MB max metadata
MAX_PAYLOAD_SIZE = 512 * 1024 * 1024    # 512 MB max payload per message

_PREFIX_STRUCT = struct.Struct("!4sBII")  # magic (4), version (1), header_len (4), payload_len (4)


class E2EProtocolError(RuntimeError):
    """Raised when protocol deserialization or socket communication fails."""
    pass


@dataclass(frozen=True)
class ProtocolMessage:
    kind: str
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _recv_all(sock: socket.socket, length: int) -> bytes:
    """Read exactly length bytes from socket."""
    buf = bytearray(length)
    view = memoryview(buf)
    offset = 0
    while offset < length:
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise EOFError("Remote peer disconnected")
        offset += n
    return bytes(buf)


def send_protocol_message(
    sock: socket.socket,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Serialize and send metadata and numpy arrays."""
    ordered_arrays: list[np.ndarray] = []
    array_descriptors: list[dict[str, Any]] = []
    total_payload = 0

    for name, arr in (arrays or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("Array name must be non-empty string")
        arr_contiguous = np.ascontiguousarray(arr)
        if arr_contiguous.dtype.hasobject:
            raise ValueError(f"Array '{name}' has object dtype (unsupported)")
        nbytes = arr_contiguous.nbytes
        if total_payload + nbytes > MAX_PAYLOAD_SIZE:
            raise ValueError(f"Payload size exceeds limit of {MAX_PAYLOAD_SIZE} bytes")
        ordered_arrays.append(arr_contiguous)
        array_descriptors.append({
            "name": name,
            "dtype": arr_contiguous.dtype.str,
            "shape": list(arr_contiguous.shape),
            "nbytes": nbytes,
        })
        total_payload += nbytes

    meta_payload = {
        "kind": kind,
        "metadata": dict(metadata or {}),
        "arrays": array_descriptors,
    }
    header_bytes = json.dumps(meta_payload, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_SIZE:
        raise ValueError(f"Header size ({len(header_bytes)}) exceeds limit of {MAX_HEADER_SIZE}")

    # Pack and send prefix
    prefix = _PREFIX_STRUCT.pack(MAGIC, PROTOCOL_VERSION, len(header_bytes), total_payload)
    sock.sendall(prefix)
    sock.sendall(header_bytes)
    for arr in ordered_arrays:
        sock.sendall(memoryview(arr).cast("B"))


def recv_protocol_message(sock: socket.socket) -> ProtocolMessage:
    """Receive and unpack a ProtocolMessage from socket."""
    prefix_data = _recv_all(sock, _PREFIX_STRUCT.size)
    magic, version, header_len, payload_len = _PREFIX_STRUCT.unpack(prefix_data)

    if magic != MAGIC:
        raise E2EProtocolError(f"Invalid protocol magic: {magic!r}")
    if version != PROTOCOL_VERSION:
        raise E2EProtocolError(f"Unsupported protocol version: {version} (expected {PROTOCOL_VERSION})")
    if header_len > MAX_HEADER_SIZE:
        raise E2EProtocolError(f"Header too large: {header_len}")
    if payload_len > MAX_PAYLOAD_SIZE:
        raise E2EProtocolError(f"Payload too large: {payload_len}")

    header_bytes = _recv_all(sock, header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        raise E2EProtocolError(f"Failed to decode header JSON: {e}") from e

    kind = header.get("kind", "")
    metadata = header.get("metadata", {})
    array_descriptors = header.get("arrays", [])

    arrays: dict[str, np.ndarray] = {}
    for desc in array_descriptors:
        name = desc["name"]
        dtype = np.dtype(desc["dtype"])
        shape = tuple(desc["shape"])
        nbytes = desc["nbytes"]
        raw = _recv_all(sock, nbytes)
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        arrays[name] = arr

    return ProtocolMessage(kind=kind, metadata=metadata, arrays=arrays)


def connect_e2e_service(
    host: str = "127.0.0.1",
    port: int = 8090,
    timeout_s: float = 30.0,
) -> socket.socket:
    """Establish connection to an E2E model server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(timeout_s)
    sock.connect((host, port))
    return sock

