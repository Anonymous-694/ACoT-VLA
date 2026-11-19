"""CPU-only checks for the unified checkpoint's public routing and wire layout."""

import numpy as np
import pytest

from openpi.four_tracks import transforms
from openpi.four_tracks.policies import go1_policy
from openpi.four_tracks.serving import task_routing
from openpi.four_tracks.serving import unified_layout
from openpi.four_tracks.serving import websocket_policy_server
from openpi.four_tracks.shared import gripper
from openpi.shared import normalize


@pytest.mark.parametrize("adapter", [go1_policy.Go1Inputs])
def test_robot_normalization_survives_policy_input_adapter(adapter):
    item = {
        "state": np.full(64, 12.0),
        "images": {name: np.zeros((3, 4, 4), dtype=np.uint8) for name in adapter.EXPECTED_CAMERAS},
        "_robot_id": "agilex",
    }
    adapted = adapter(action_dim=64)(item)
    stats = normalize.NormStats(mean=np.full(64, 10.0), std=np.full(64, 2.0))
    normalized = transforms.PerRobotNormalize({"agilex": {"state": stats}})(adapted)
    np.testing.assert_allclose(normalized["state"], np.ones(64), atol=1e-6)


@pytest.mark.parametrize(("field", "closed"), [("state", 120.0), ("action", 1.0)])
def test_committed_gripper_api_keeps_local_a2d_units(field, closed):
    raw = np.array([0.0, closed / 2, closed])
    unified = gripper.to_unified(raw, "a2d_sim", field, end_effector_type="omnipicker")
    np.testing.assert_allclose(unified, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(gripper.to_raw(unified, "a2d_sim", field, "omnipicker"), raw)


def test_task_table_covers_all_fifty_embodiment_routes():
    table = task_routing.load()
    counts = {group: len(names) for group, names in task_routing.by_group(table).items()}
    assert len(table) == 47
    assert counts == {
        "g2_ctek90d_wbc": 8,
        "g2_ctek90d_manip": 8,
        "g2_wuji": 10,
        "g2op": 6,
        "g1op": 6,
        "arx": 6,
        "agilex": 6,
    }
    task_routing.validate(table, counts)


@pytest.mark.parametrize(
    ("robot_type", "group"),
    [("g2a_sim", "g2op"), ("a2d_sim", "g1op"), ("arx_sim", "arx"), ("agilex_sim", "agilex")],
)
def test_shared_task_uses_the_requested_embodiment(robot_type, group):
    assert (
        task_routing.resolve(b" OPEN_SHORT_TRASH_CAN_AND_THROW_GARBAGE_MULTI_ROBOT ", robot_type, task_routing.load())
        == group
    )


@pytest.mark.parametrize("task", [None, "", "not_a_registered_task"])
def test_unknown_task_never_selects_default_normalization(task):
    with pytest.raises(task_routing.TaskRouteError):
        task_routing.resolve(task, "g2a_sim", task_routing.load())


def test_shared_task_rejects_unknown_embodiment():
    with pytest.raises(task_routing.TaskRouteError):
        task_routing.resolve("open_short_trash_can_and_throw_garbage_multi_robot", "unknown", task_routing.load())


@pytest.mark.parametrize("dof", [6, 7])
def test_each_arm_is_padded_separately_and_grippers_keep_raw_units(dof):
    state = unified_layout.assemble_state(
        {b"arm_joint_states": list(range(2 * dof)), b"gripper_states": [-3.5, -0.2]}, arm_dof_per_side=dof
    )
    np.testing.assert_array_equal(state[:dof], np.arange(dof))
    np.testing.assert_array_equal(state[7 : 7 + dof], np.arange(dof, 2 * dof))
    np.testing.assert_array_equal(state[dof:7], 0)
    np.testing.assert_array_equal(state[7 + dof : 14], 0)
    assert state[22] == -3.5
    assert state[42] == -0.2
    np.testing.assert_array_equal(state[23:42], 0)
    np.testing.assert_array_equal(state[43:62], 0)


def test_wbc_state_keeps_waist_but_masks_chassis_pose():
    payload = {"waist_joint_states": [1, 2, 3, 4, 5], "chassis_states": [99, 88, 77]}
    state = unified_layout.assemble_state(payload, fill_waist=True)
    np.testing.assert_array_equal(state[14:19], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(state[19:22], 0)
    np.testing.assert_array_equal(unified_layout.assemble_state(payload)[14:22], 0)


def test_dexterous_hands_preserve_all_forty_joints():
    state = unified_layout.assemble_state({"gripper_states": np.arange(40)}, dex=True)
    np.testing.assert_array_equal(state[22:42], np.arange(20))
    np.testing.assert_array_equal(state[42:62], np.arange(20, 40))
    result = unified_layout.to_result(np.tile(state, (50, 1)), dex=True)
    np.testing.assert_array_equal(result["left_effector"], np.tile(np.arange(20), (50, 1)))
    np.testing.assert_array_equal(result["right_effector"], np.tile(np.arange(20, 40), (50, 1)))


def test_wbc_outputs_integrate_velocity_and_keep_action_horizon():
    actions = np.zeros((50, 64))
    actions[:, 14:19] = [1, 2, 3, 4, 5]
    actions[:, 19:22] = [0.3, -0.6, 0.9]
    result = unified_layout.to_result(actions, emit_waist=True, emit_base=True)
    assert np.asarray(result["left_arm"]["values"]).shape == (50, 7)
    assert np.asarray(result["waist"]["values"]).shape == (50, 5)
    np.testing.assert_allclose(result["wheels"][-1], [0.5, -1.0, 1.5])


@pytest.mark.parametrize(
    ("robot", "ee", "group", "dof", "dex", "wbc"),
    [
        ("g2a_sim", "zhiyuan_gripper_ctek90d", "g2_ctek90d_wbc", 7, False, True),
        ("g2a_sim", "zhiyuan_gripper_ctek90d", "g2_ctek90d_manip", 7, False, False),
        ("g2a_sim", "wuji", "g2_wuji", 7, True, False),
        ("g2a_sim", "zhiyuan_gripper_omnipicker", "g2op", 7, False, False),
        ("a2d_sim", "zhiyuan_gripper_omnipicker", "g1op", 7, False, False),
        ("arx_sim", "arx_gripper", "arx", 6, False, False),
        ("agilex_sim", "agilex_gripper", "agilex", 6, False, False),
    ],
)
def test_serving_entry_uses_the_selected_group_layout(robot, ee, group, dof, dex, wbc):
    states = {
        "arm_joint_states": list(range(2 * dof)),
        "gripper_states": list(range(40)) if dex else [-0.7, -0.1],
        "waist_joint_states": [1, 2, 3, 4, 5],
        "chassis_states": [99, 88, 77],
    }
    params = {"robot_type": robot, "ee_type": ee, "states": states, "images": {}, "prompt": "test action"}
    assert websocket_policy_server._check_group_matches_payload(group, params) is None  # noqa: SLF001
    obs = websocket_policy_server._simulation_params_to_obs(  # noqa: SLF001
        params,
        embodiment="dex" if dex else "gseries",
        prepend_robot_type=True,
        arm_canonical_per_arm_dim=7,
        unified_layout=True,
        unified_scenario="wbc" if wbc else "manip",
        prefix_override=group,
    )
    assert obs["prompt"] == f"current robot is {group}, test action"
    np.testing.assert_array_equal(obs["state"][:dof], np.arange(dof))
    np.testing.assert_array_equal(obs["state"][7 : 7 + dof], np.arange(dof, 2 * dof))
    np.testing.assert_array_equal(obs["state"][19:22], 0)
    np.testing.assert_array_equal(obs["state"][14:19], [1, 2, 3, 4, 5] if wbc else 0)


def test_wrong_hand_width_is_rejected_before_policy_inference():
    error = websocket_policy_server._check_group_matches_payload("g2_wuji", {"states": {"gripper_states": [0, 0]}})  # noqa: SLF001
    assert error
    assert "40 gripper values" in error


@pytest.mark.parametrize("stage", ["routing", "input", "normalization", "inference", "output", "decode"])
def test_rpc_errors_keep_exception_details_in_server_logs(monkeypatch, caplog, stage):
    import asyncio
    import logging

    from openpi_client import msgpack_numpy
    import websockets
    import websockets.asyncio.server

    detail = "sensitive-model-detail"

    def fail(*args, **kwargs):
        raise ValueError(detail)

    class Policy:
        def infer(self, obs):
            if stage == "inference":
                fail()
            return {"actions": np.zeros((50, 1 if stage == "output" else 62))}

    if stage == "routing":
        monkeypatch.setattr(task_routing, "resolve", fail)
    elif stage == "input":
        monkeypatch.setattr(websocket_policy_server, "_simulation_params_to_obs", fail)
    elif stage == "normalization":
        monkeypatch.setattr(websocket_policy_server, "_check_group_matches_payload", fail)
    caplog.set_level(logging.ERROR)

    async def check():
        server = websocket_policy_server.WebsocketPolicyServer(
            Policy(),
            prepend_robot_type=True,
            arm_canonical_per_arm_dim=7,
            scenario_routing=True,
            unified_layout=True,
            prefix_from_norm_group=True,
            task_routes=task_routing.load(),
            waist_groups=("g2_ctek90d_wbc",),
        )
        async with websockets.asyncio.server.serve(server._handler, "127.0.0.1", 0) as running:
            port = running.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                request = {
                    "method": "infer",
                    "params": {
                        "task_name": "clean_the_blackboard",
                        "robot_type": "g2a_sim",
                        "prompt": "test",
                        "images": {},
                        "states": {
                            "arm_joint_states": [0.0] * 14,
                            "gripper_states": [0.0] * 2,
                            "waist_joint_states": [0.0] * 5,
                        },
                    },
                }
                payload = b"\xc1" if stage == "decode" else msgpack_numpy.Packer().pack(request)
                await client.send(payload)
                response = msgpack_numpy.unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                assert response["error"]
                assert response["error_code"] in {
                    "NORMALIZATION_ERROR",
                    "INVALID_INPUT",
                    "INVALID_OUTPUT",
                    "INTERNAL_ERROR",
                }
                assert len(response["error_id"]) == 32
                text = str(response)
                assert detail not in text and "Traceback" not in text and "websocket_policy_server.py" not in text
                assert response["error_id"] in caplog.text
                assert "Traceback" in caplog.text
                if stage not in ("output", "decode"):
                    assert detail in caplog.text

    asyncio.run(check())
