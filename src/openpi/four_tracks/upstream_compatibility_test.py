"""Check that the extension and the upstream ACoT entrypoints can coexist."""

import asyncio
import os
import subprocess
import sys

import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server


def test_importing_extension_does_not_patch_upstream_dataset():
    script = """
from lerobot.common.datasets import lerobot_dataset
from openpi.training import config as legacy_config
before = lerobot_dataset.LeRobotDatasetMetadata.load_metadata
aggregate = lerobot_dataset.aggregate_stats
from openpi.four_tracks.training import data_loader
from openpi.four_tracks.policies import policy_config
assert lerobot_dataset.LeRobotDatasetMetadata.load_metadata is before
assert lerobot_dataset.aggregate_stats is aggregate
assert legacy_config.get_config('acot_icra_simulation_challenge_reasoning_to_action')
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr


def test_original_acot_config_and_model_are_available():
    from openpi.models.acot_vla import ACOTConfig
    from openpi.training.config import get_config

    cfg = get_config("acot_icra_simulation_challenge_reasoning_to_action")
    assert isinstance(cfg.model, ACOTConfig)


def test_legacy_and_four_track_postprocessing_keep_distinct_layouts():
    from openpi.four_tracks.policies.policy import Policy as FourTrackPolicy
    from openpi.policies.policy import Policy as LegacyPolicy

    legacy = LegacyPolicy.__new__(LegacyPolicy)
    four_track = FourTrackPolicy.__new__(FourTrackPolicy)
    outputs = {"actions": np.zeros((50, 62)), "policy_timing": {"infer_ms": 1.0}}
    legacy_result = legacy.post_process({"task_name": "legacy_gripper_task"}, outputs)
    assert legacy_result["actions"].shape == (50, 16)

    outputs = {"actions": np.zeros((50, 62)), "policy_timing": {"infer_ms": 1.0}}
    new_result = four_track.post_process({"task_name": "clean_the_blackboard"}, outputs)
    assert new_result["actions"].shape == (50, 62)

    state = np.arange(21, dtype=np.float32)
    outputs = {"actions": np.zeros((50, 21)), "policy_timing": {"infer_ms": 1.0}}
    waist = legacy.post_process({"task_name": "sorting_packages", "state": state}, outputs)
    np.testing.assert_array_equal(waist["actions"][:, 16:20], np.tile(state[16:20], (50, 1)))


def test_original_websocket_metadata_handshake_and_raw_observation():
    # Import the extension first to catch accidental changes to the legacy server.
    from openpi.four_tracks.serving import websocket_policy_server as four_track_server
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    assert four_track_server.WebsocketPolicyServer is not WebsocketPolicyServer

    class Policy:
        def __init__(self):
            self.seen = []

        def infer(self, obs):
            self.seen.append(obs)
            return {"actions": np.zeros((2, 16))}

    async def check():
        policy = Policy()
        metadata = {"legacy": True}
        server = WebsocketPolicyServer(policy, host="127.0.0.1", port=0, metadata=metadata)
        async with websockets.asyncio.server.serve(server._handler, "127.0.0.1", 0) as running:
            port = running.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                greeting = msgpack_numpy.unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                assert greeting == metadata
                obs = {"state": [0.0] * 16, "prompt": "legacy observation"}
                await client.send(msgpack_numpy.Packer().pack(obs))
                response = msgpack_numpy.unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                assert response["actions"].shape == (2, 16)
                assert policy.seen == [obs]

    asyncio.run(check())
