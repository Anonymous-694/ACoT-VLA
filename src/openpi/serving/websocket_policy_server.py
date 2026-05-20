import asyncio
import http
import logging
import time
import traceback

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


_IMAGE_KEYS = ("top_head", "hand_left", "hand_right")


def _get(d: dict, key: str):
    """Get a key from a msgpack-decoded dict, tolerating bytes vs str keys."""
    if key in d:
        return d[key]
    bkey = key.encode("utf-8")
    if bkey in d:
        return d[bkey]
    return None


def _maybe_decode_image(d):
    """Return HWC RGB uint8. Pass through if already a numpy array (legacy clients).

    Client encodes obs RGB → BGR → JPEG; we invert that here so downstream policy
    code sees RGB HWC, matching the legacy uncompressed contract.
    """
    if isinstance(d, dict) and (_get(d, "__jpeg__")):
        raw = _get(d, "data")
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC BGR
        if img is None:
            raise ValueError("Failed to JPEG-decode image payload")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return d


def _maybe_decode_depth(d):
    """Return HW uint16 (scaled, same as before). Pass through legacy numpy arrays."""
    if isinstance(d, dict) and (_get(d, "__png16__")):
        raw = _get(d, "data")
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # HW uint16
        if img is None:
            raise ValueError("Failed to PNG-decode depth payload")
        return img
    return d


def _normalize_payload(payload: dict) -> dict:
    """Decode compressed images/depth fields in-place produced by PiPolicy clients."""
    images = payload.get("images") if isinstance(payload, dict) else None
    if isinstance(images, dict):
        for k in _IMAGE_KEYS:
            if k in images:
                images[k] = _maybe_decode_image(images[k])
    depth = payload.get("depth") if isinstance(payload, dict) else None
    if isinstance(depth, dict):
        for k in _IMAGE_KEYS:
            if k in depth:
                depth[k] = _maybe_decode_depth(depth[k])
    return payload


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs = _normalize_payload(obs)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
