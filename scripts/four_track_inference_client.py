"""Minimal simulation-only request/response client; no hardware control."""

import argparse
import asyncio
import json
from pathlib import Path

from openpi_client import msgpack_numpy
import websockets


async def main(args):
    request = {
        "method": "infer",
        "params": {
            "task_name": args.task,
            "robot_type": args.robot,
            "prompt": args.prompt,
            "states": json.loads(Path(args.states).read_text()),
            "images": {
                "head": {"encoding": "JPEG", "image_data": Path(args.head).read_bytes()},
                "hand_left": {"encoding": "JPEG", "image_data": Path(args.left_camera).read_bytes()},
                "hand_right": {"encoding": "JPEG", "image_data": Path(args.right_camera).read_bytes()},
            },
        },
    }
    async with websockets.connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None) as socket:
        await socket.send(msgpack_numpy.Packer().pack(request))
        response = msgpack_numpy.unpackb(await socket.recv())
    if response.get("error"):
        raise RuntimeError(response["error"])
    print(json.dumps(response, default=lambda value: value.tolist(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--prompt", required=True)
    for key in ("head", "left-camera", "right-camera", "states"):
        parser.add_argument("--" + key, required=True)
    asyncio.run(main(parser.parse_args()))
