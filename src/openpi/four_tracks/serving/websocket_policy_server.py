# Derived from ACoT-VLA/openpi: src/openpi/serving/websocket_policy_server.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import collections
import http
import logging
import os
import pickle
import time
import uuid

import cv2
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.four_tracks.serving import task_routing as _task_routing
from openpi.four_tracks.shared import robot_config as _robot_config

logger = logging.getLogger(__name__)


def _server_error(message: str, code: str) -> dict:
    """Keep exception details in server logs and return a safe diagnostic reference."""
    error_id = uuid.uuid4().hex
    logger.exception("%s [error_id=%s]", message, error_id)
    return {"error": message, "error_code": code, "error_id": error_id}


# simulation's image key naming
_SIMULATION_IMG_KEYS = ("head", "hand_left", "hand_right")
# how the training policy names the same images
_POLICY_IMG_KEY_MAP = {
    "head": "top_head",
    "hand_left": "hand_left",
    "hand_right": "hand_right",
}


def _get(d: dict, key: str):
    """Get a key from a msgpack-decoded dict, tolerating bytes vs str keys."""
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    bkey = key.encode("utf-8")
    if bkey in d:
        return d[bkey]
    return None


def _decode_simulation_image(d):
    """Decode a simulation JPEG dict {encoding, image_data, height, width} → HWC RGB uint8.

    Pass through if it's already a numpy array (legacy clients).
    """
    if isinstance(d, np.ndarray):
        return d
    if not isinstance(d, dict):
        raise ValueError(f"Unexpected image payload type: {type(d)}")
    encoding = _get(d, "encoding")
    raw = _get(d, "image_data")
    if raw is None:
        # Fallback: pi-style {"__jpeg__": True, "data": ...}
        raw = _get(d, "data")
    if raw is None:
        raise ValueError(f"Image dict missing image_data/data; keys={list(d.keys())}")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding and encoding.upper() not in ("JPEG", "PNG"):
        raise ValueError(f"Unsupported image encoding: {encoding}")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC BGR
    if img is None:
        raise ValueError("Failed to JPEG-decode image payload")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# simulation robot_type tag -> layout family that assemble/repack branch on.
_ROBOT_TYPE_TO_FAMILY = {
    "arx_sim": "arx",
    "agilex_sim": "aloha",
    "a2d_sim": "gseries",
    "g2a_sim": "gseries",
    "g2wuji": "dex",
}


# simulation ee_type tag -> layout family. Checked BEFORE robot_type: every G2 sends
# robot_type="g2a_sim" (the training-dataset tag), so only the end-effector tag
# tells a five-finger hand apart from a two-finger gripper.
_EE_TYPE_TO_FAMILY = {
    "wuji": "dex",
}


# Public simulation tags are also the training dataset tags.


def _resolve_dataset_robot_type(params: dict) -> str | None:
    """Map the payload's robot_type to the training-dataset tag for prompt prepending.

    Look up robot_config.json by robot_type+ee_type, then by robot_type alone.
    Other tags are returned verbatim for explicit routing validation.
    """
    rt = _get(params, "robot_type")
    if isinstance(rt, bytes):
        rt = rt.decode("utf-8")
    if not rt:
        return None

    ee = _get(params, "ee_type")
    if isinstance(ee, bytes):
        ee = ee.decode("utf-8")

    cfg_result = _robot_config.get_by_robot_ee(rt, ee or "") if ee else None
    if cfg_result is None:
        cfg_result = _robot_config.get_by_robot_type(rt)
    if cfg_result:
        return cfg_result[1].get("robot_type", rt)

    return rt


def _detect_embodiment(states: dict) -> str:
    """Fallback for clients without usable tags: >2 gripper DOFs -> dex,
    12-DOF arm -> arx, else gseries.

    arx and aloha share a layout, so guessing arx for a 12-DOF arm is safe.
    A two-finger gripper reports 1 DOF per hand, so anything wider is a
    dexterous hand and must not be squeezed into the 2-slot gseries layout.
    """
    grip = _get(states, "gripper_states")
    if grip is not None and np.asarray(grip).reshape(-1).shape[0] > 2:
        return "dex"
    arm = _get(states, "arm_joint_states")
    if arm is not None:
        n = np.asarray(arm).reshape(-1).shape[0]
        if n == 12:
            return "arx"
    return "gseries"


def _resolve_embodiment(params: dict) -> str:
    """Pick the layout family from the client's ee_type / robot_type tags, else fall back.

    Lookup order: robot_config.json (by robot_type+ee_type, then by robot_type alone)
    → legacy hardcoded tables → arm-length heuristic.
    """
    ee = _get(params, "ee_type")
    if isinstance(ee, bytes):
        ee = ee.decode("utf-8")
    rt = _get(params, "robot_type")
    if isinstance(rt, bytes):
        rt = rt.decode("utf-8")

    if rt:
        cfg_result = _robot_config.get_by_robot_ee(rt, ee or "") if ee else None
        if cfg_result is None:
            cfg_result = _robot_config.get_by_robot_type(rt)
        if cfg_result and cfg_result[1].get("embodiment"):
            return cfg_result[1]["embodiment"]

    if ee:
        family = _EE_TYPE_TO_FAMILY.get(ee)
        if family is not None:
            return family
    if rt:
        family = _ROBOT_TYPE_TO_FAMILY.get(rt)
        if family is not None:
            return family
        logger.warning("unknown robot_type=%r; falling back to arm-length heuristic", rt)
    return _detect_embodiment(_get(params, "states") or {})


# How wide the client's gripper_states must be for each norm group: the dexterous hand
# sends 20 joints per side, every gripper sends one. Used to catch a client wired to the
# wrong port -- simulation reports robot_type="g2a_sim" for ctek90d, omnipicker and wuji
# alike, so an explicit --scenario can otherwise route a wuji robot to ctek90d stats
# without any error.
_GROUP_GRIPPER_WIDTH = {
    "g2_wuji": 40,
    "g2_ctek90d_wbc": 2,
    "g2_ctek90d_manip": 2,
    "g2op": 2,
    "g1op": 2,
    "arx": 2,
    "agilex": 2,
}


def _check_group_matches_payload(group: str, params: dict) -> str | None:
    """Return an error string when the payload cannot belong to the resolved norm group."""
    want = _GROUP_GRIPPER_WIDTH.get(group)
    if want is None:
        return None
    states = _get(params, "states") or {}
    grip = _get(states, "gripper_states")
    if grip is None:
        return None
    got = len(np.asarray(grip).reshape(-1))
    if got != want:
        return (
            f"norm group {group!r} expects {want} gripper values but the client sent "
            f"{got}; the robot on this connection is probably not what --scenario/"
            f"--norm_group declares"
        )
    return None


def _simulation_params_to_obs(
    params: dict,
    embodiment: str = "gseries",
    prepend_robot_type: bool = False,
    unify_gripper: bool = False,
    wbc_layout: dict | None = None,
    arm_canonical_per_arm_dim: int | None = None,
    unified_layout: bool = False,
    unified_scenario: str | None = None,
    prefix_override: str | None = None,
) -> dict:
    """Translate the simulation JSON-RPC params block into the flat dict the policy expects."""
    # Per-group debug budget: each norm group gets its own OPENPI_DEBUG_LOG_N requests.
    _dbg_key = prefix_override or "default"
    images_in = _get(params, "images") or {}
    images_out = {}
    for k in _SIMULATION_IMG_KEYS:
        v = _get(images_in, k)
        if v is None:
            continue
        images_out[_POLICY_IMG_KEY_MAP[k]] = _decode_simulation_image(v)

    robot_type = (
        _resolve_dataset_robot_type(params)
        if (prepend_robot_type or unify_gripper or arm_canonical_per_arm_dim)
        else None
    )

    ee_type = _get(params, "ee_type")
    if isinstance(ee_type, bytes):
        ee_type = ee_type.decode("utf-8")

    detailed_cfg = None
    result = _robot_config.get_by_robot_ee(robot_type or "", ee_type or "") if ee_type else None
    if result is None and robot_type:
        result = _robot_config.get_by_robot_type(robot_type)
    if result:
        detailed_cfg = result[1]

    arm_dof = detailed_cfg.get("arm_dof_per_side") if detailed_cfg else None

    states_in = _get(params, "states") or {}
    raw_arm = _get(states_in, "arm_joint_states")
    raw_grip = _get(states_in, "gripper_states")
    if raw_arm is not None and _simulation_params_to_obs._log_count[_dbg_key] < _DEBUG_LOG_N:
        logger.info(
            "raw arm_joint_states(%d): %s",
            len(np.asarray(raw_arm).reshape(-1)),
            np.round(np.asarray(raw_arm, dtype=np.float64).reshape(-1), 4).tolist(),
        )
    if raw_grip is not None and _simulation_params_to_obs._log_count[_dbg_key] < _DEBUG_LOG_N:
        logger.info("raw gripper_states: %s", np.round(np.asarray(raw_grip, dtype=np.float64).reshape(-1), 4).tolist())
    if unified_layout:
        from openpi.four_tracks.serving import unified_layout as _ul

        state = _ul.assemble_state(
            states_in,
            arm_dof_per_side=arm_dof,
            dex=(embodiment == "dex"),
            fill_waist=(unified_scenario == "wbc"),
        )
    else:
        raise ValueError("this release only serves the four-track unified62 layout")

    if _simulation_params_to_obs._log_count[_dbg_key] < _DEBUG_LOG_N:
        logger.info("canonical state(%d): %s", len(state), np.round(state, 4).tolist())
        _simulation_params_to_obs._log_count[_dbg_key] += 1

    prompt = _get(params, "prompt") or ""
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8")

    if prepend_robot_type:
        prompt_rt = prefix_override or robot_type
        if prompt_rt:
            prompt = f"current robot is {prompt_rt}, {prompt}"

    if _simulation_params_to_obs._prompt_log_count[_dbg_key] < _DEBUG_LOG_N:
        logger.info("DEBUG prompt sent to model: %r", prompt)
        _simulation_params_to_obs._prompt_log_count[_dbg_key] += 1

    obs = {
        "state": state,
        "images": images_out,
        "prompt": prompt,
        "task_name": _get(params, "task_name") or "",
        "episode_idx": _get(params, "episode_idx") or 0,
        "episode_done": bool(_get(params, "episode_done") or False),
        "task_progress": _get(params, "task_progress") or [],
    }
    obs["_robot_id"] = robot_type
    return obs


# Number of requests to dump full raw/canonical/prompt detail for. Bump via
# OPENPI_DEBUG_LOG_N when diffing train vs serve over a whole eval run.
#
# The budget is PER NORM GROUP, not per process: one server now serves every data family, and
# a single shared counter would be spent by whichever robot connects first, leaving the other
# six groups with no diagnostics at all.
_DEBUG_LOG_N = int(os.environ.get("OPENPI_DEBUG_LOG_N", "3"))
_simulation_params_to_obs._log_count = collections.Counter()
_simulation_params_to_obs._prompt_log_count = collections.Counter()


class WebsocketPolicyServer:
    """Serves a policy over WebSocket using the simulation JSON-RPC protocol.

    Wire format
    -----------
    Request (client → server):
        {"method": "infer", "params": {images, states, prompt, task_name, ...}}
    Success response (server → client):
        {"result": {left_arm, right_arm, left_effector, right_effector, [waist]}}
    Failure response:
        {"error": "<message>", "error_code": "<code>", "error_id": "<log reference>"}

    There is **no metadata first packet** (simulation clients don't expect one).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,  # accepted for backward-compat; unused by simulation
        prepend_robot_type: bool = False,
        unify_gripper: bool = False,
        wbc_layout: dict | None = None,
        arm_canonical_per_arm_dim: int | None = None,
        action_mode: str = "joint",
        robot_type: str | None = None,
        save_request_dir: str | None = None,
        save_request_limit: int = 50,
        scenario: str | None = None,
        norm_group_override: str | None = None,
        scenario_routing: bool = False,
        unified_layout: bool = False,
        prefix_from_norm_group: bool = False,
        task_routes: dict | None = None,
        route_fallback: str = "none",
        waist_groups: tuple[str, ...] = (),
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._prepend_robot_type = prepend_robot_type
        self._unify_gripper = unify_gripper
        self._wbc_layout = wbc_layout
        self._arm_canonical_per_arm_dim = arm_canonical_per_arm_dim
        self._action_mode = action_mode
        self._robot_type_override = robot_type
        self._save_request_dir = os.path.abspath(save_request_dir) if save_request_dir else None
        self._save_request_limit = save_request_limit
        self._saved_count = 0
        self._scenario = scenario
        self._norm_group_override = norm_group_override
        if norm_group_override:
            logger.info("norm_group override=%s — resolve_norm_group bypassed for every request", norm_group_override)
        self._scenario_routing = scenario_routing
        self._unified_layout = unified_layout
        self._prefix_from_norm_group = prefix_from_norm_group
        self._task_routes = task_routes or None
        self._route_fallback = route_fallback
        # Which norm groups populate the waist/base blocks. Mirrors the training config's
        # waist_fill_only_for, so inference fills and emits exactly what training supervised.
        self._waist_groups = tuple(waist_groups or ())
        if self._task_routes:
            logger.info(
                "task_name routing ON: %d task names -> %d norm groups (one process serves every family)",
                len(self._task_routes),
                len(_task_routing.groups(self._task_routes)),
            )
            for _g, _names in sorted(_task_routing.by_group(self._task_routes).items()):
                logger.info("  %-18s (%d): %s", _g, len(_names), ", ".join(_names))
            logger.info("  waist/base groups: %s", list(self._waist_groups) or "none")
            if norm_group_override:
                logger.info(
                    "  NOTE --norm_group=%s overrides the table for every request, so "
                    "this process still serves one family only",
                    norm_group_override,
                )
            if scenario:
                logger.info(
                    "  NOTE --scenario=%s is only a fallback now; the table decides each request's family", scenario
                )
        if scenario_routing and not self._task_routes:
            logger.info(
                "norm-group routing ON (scenario=%s): _robot_id resolved via "
                "robot_config.resolve_norm_group (no task_name table)",
                scenario,
            )
        if self._save_request_dir:
            os.makedirs(self._save_request_dir, exist_ok=True)
            logger.info("save_request_dir=%s (first %d requests)", self._save_request_dir, save_request_limit)
        if robot_type:
            logger.info("robot_type override=%s — requests without a usable tag use this layout", robot_type)
        if metadata:
            logger.info("Server metadata=%s — ignored by simulation protocol", metadata)
        if prepend_robot_type:
            logger.info("prepend_robot_type=True — prompts will be prefixed with 'current robot is <tag>, '")
        if unify_gripper:
            logger.info(
                "unify_gripper=True — gripper state/action converted to/from the unified "
                "open=0/closed=1 space per robot_type (mixed-embodiment checkpoint)"
            )
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
        logger.info(f"Connection from {websocket.remote_address} opened (simulation)")
        packer = msgpack_numpy.Packer()

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                req = msgpack_numpy.unpackb(await websocket.recv())

                method = _get(req, "method")
                if isinstance(method, bytes):
                    method = method.decode("utf-8")
                if method != "infer":
                    await websocket.send(packer.pack({"error": f"unknown method: {method!r}"}))
                    continue

                params = _get(req, "params") or {}
                obs = None  # per-request; must reset or a failed infer would dump the previous one
                action = None
                if self._robot_type_override:
                    # Authoritative: clients commonly send the TRAINING-dataset tag
                    # ("g2a_sim"), which get_by_robot_type resolves to the FIRST such entry
                    # (a 2-gripper gseries robot). That silently truncates a 40-DOF hand, so
                    # --robot_type must win over whatever the payload claims.
                    params = dict(params)
                    params["robot_type"] = self._robot_type_override
                embodiment = _resolve_embodiment(params)
                robot_type = (
                    _resolve_dataset_robot_type(params)
                    if (self._unify_gripper or self._arm_canonical_per_arm_dim)
                    else None
                )
                if robot_type:
                    logger.info("robot_type=%s, embodiment=%s", robot_type, embodiment)

                # Resolve arm_dof_per_side from robot_config when doing variable-arm-DOF
                arm_dof = None
                if self._arm_canonical_per_arm_dim and robot_type:
                    ee_type = _get(params, "ee_type")
                    if isinstance(ee_type, bytes):
                        ee_type = ee_type.decode("utf-8")
                    cfg_result = _robot_config.get_by_robot_ee(robot_type, ee_type or "") if ee_type else None
                    if cfg_result is None:
                        cfg_result = _robot_config.get_by_robot_type(robot_type)
                    if cfg_result:
                        arm_dof = cfg_result[1].get("arm_dof_per_side")
                    logger.info(
                        "robot_type=%s, ee_type=%r, arm_dof_per_side=%s, canonical=%s",
                        robot_type,
                        ee_type,
                        arm_dof,
                        self._arm_canonical_per_arm_dim,
                    )

                # ---- one norm group per request -------------------------------------
                # The group decides four things at once: which norm_stats are applied, the
                # prompt prefix, whether the waist/base blocks are filled and emitted, and
                # whether the hand is a 2-slot gripper or a 20-joint dexterous hand. They must
                # all come from the SAME value, so it is resolved once here and reused below.
                _group = None
                _group_err = None
                if self._prefix_from_norm_group or self._scenario_routing:
                    _ee0 = _get(params, "ee_type")
                    if isinstance(_ee0, bytes):
                        _ee0 = _ee0.decode("utf-8")
                    _rt0 = _resolve_dataset_robot_type(params)
                    try:
                        if self._norm_group_override:
                            _group = self._norm_group_override
                        elif self._task_routes is not None:
                            try:
                                _group = _task_routing.resolve(
                                    _get(params, "task_name"),
                                    _rt0,
                                    self._task_routes,
                                    param_keys=list(params.keys()) if isinstance(params, dict) else None,
                                )
                            except _task_routing.TaskRouteError:
                                if self._route_fallback != "robot_type":
                                    raise
                                _group = _robot_config.resolve_norm_group(_rt0, _ee0, self._scenario)
                                logger.warning("task_name not routable; fell back to robot_type -> %s", _group)
                        else:
                            _group = _robot_config.resolve_norm_group(_rt0, _ee0, self._scenario)
                    except _task_routing.TaskRouteError as exc:
                        _group_err = {"error": f"cannot route request to a norm group: {exc.args[0]}"}
                    except Exception:
                        _group_err = _server_error("Could not resolve normalization group.", "NORMALIZATION_ERROR")
                    if _group_err:
                        await websocket.send(packer.pack(_group_err))
                        continue

                _prefix_override = _group if self._prefix_from_norm_group else None
                # Fill/emit waist+base exactly for the groups the config trained with them.
                _wbc_group = bool(_group) and _group in self._waist_groups
                _scenario_for_request = ("wbc" if _wbc_group else "manip") if _group else self._scenario

                if self._unified_layout and _prefix_override:
                    # The norm group is the authoritative identity for this layout; derive the
                    # hand width from it rather than from _resolve_embodiment. simulation reports
                    # robot_type="g2a_sim" for ctek90d, omnipicker and wuji alike, and the
                    # registry lookup then returns whichever entry happens to come first --
                    # which would squeeze a 20-joint hand into the 2-slot gripper layout.
                    _want_dex = _GROUP_GRIPPER_WIDTH.get(_prefix_override) == 40
                    if _want_dex and embodiment != "dex":
                        logger.info("embodiment %s -> dex (norm group %s)", embodiment, _prefix_override)
                        embodiment = "dex"
                    elif not _want_dex and embodiment == "dex":
                        logger.info("embodiment dex -> gseries (norm group %s)", _prefix_override)
                        embodiment = "gseries"

                try:
                    obs = _simulation_params_to_obs(
                        params,
                        embodiment,
                        self._prepend_robot_type,
                        self._unify_gripper,
                        wbc_layout=self._wbc_layout,
                        arm_canonical_per_arm_dim=self._arm_canonical_per_arm_dim,
                        unified_layout=self._unified_layout,
                        unified_scenario=_scenario_for_request,
                        prefix_override=_prefix_override,
                    )
                except Exception:
                    await websocket.send(packer.pack(_server_error("Invalid inference input.", "INVALID_INPUT")))
                    continue

                if self._scenario_routing:
                    # Training keys per-robot norm_stats by norm_group_ids, not by the coarse tag
                    # (g2a_sim covers four stats sets). Resolve explicitly; a miss must be an
                    # error, because an unknown key silently falls back to no normalization.
                    try:
                        _ee = _get(params, "ee_type")
                        if isinstance(_ee, bytes):
                            _ee = _ee.decode("utf-8")
                        # Already resolved above; reusing it is what keeps the norm_stats, the
                        # prompt prefix and the waist/base handling from disagreeing.
                        obs["_robot_id"] = (
                            _group
                            or self._norm_group_override
                            or _robot_config.resolve_norm_group(
                                _resolve_dataset_robot_type(params), _ee, self._scenario
                            )
                        )
                        _mismatch = _check_group_matches_payload(obs["_robot_id"], params)
                        if (
                            obs["_robot_id"] in self._waist_groups
                            and _get(_get(params, "states") or {}, "chassis_states") is None
                        ):
                            # Warn only. Never route ON chassis presence: a fine-manipulation
                            # client that also sends it would silently pick the WBC stats.
                            logger.warning(
                                "routed to %s (a waist/base group) but the payload has no "
                                "chassis_states; check the task_name is right for this robot",
                                obs["_robot_id"],
                            )
                        if _mismatch:
                            logger.error("%s", _mismatch)
                            await websocket.send(packer.pack({"error": _mismatch}))
                            continue
                        _rid_counts = getattr(self, "_rid_log_count", None)
                        if _rid_counts is None:
                            _rid_counts = self._rid_log_count = collections.Counter()
                        _rid_counts[obs["_robot_id"]] += 1
                        if _rid_counts[obs["_robot_id"]] <= _DEBUG_LOG_N:
                            logger.info(
                                "norm group: _robot_id=%s (robot_type=%s ee_type=%s scenario=%s)",
                                obs["_robot_id"],
                                _resolve_dataset_robot_type(params),
                                _ee,
                                self._scenario,
                            )
                    except Exception:
                        await websocket.send(
                            packer.pack(_server_error("Could not resolve normalization group.", "NORMALIZATION_ERROR"))
                        )
                        continue

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                try:
                    if self._unified_layout:
                        from openpi.four_tracks.serving import unified_layout as _ul

                        _acts = action.get("actions")
                        if _acts is None:
                            _acts = action[b"actions"]
                        _ul_counts = getattr(self, "_ul_log_count", None)
                        if _ul_counts is None:
                            _ul_counts = self._ul_log_count = collections.Counter()
                        _ul_key = _group or "default"
                        _ul_counts[_ul_key] += 1
                        if _ul_counts[_ul_key] <= _DEBUG_LOG_N:
                            _a = np.asarray(_acts)
                            logger.info("unified action shape=%s", _a.shape)
                            if _a.shape[-1] < _ul.DIM:
                                logger.error(
                                    "action has %d dims, the unified layout needs %d — something "
                                    "truncated the policy output (check Policy.post_process)",
                                    _a.shape[-1],
                                    _ul.DIM,
                                )
                            for _nm, _lo, _hi in (
                                ("arm_L", 0, 7),
                                ("arm_R", 7, 14),
                                ("waist", 14, 19),
                                ("base", 19, 22),
                                ("hand_L", 22, 42),
                                ("hand_R", 42, 62),
                            ):
                                _blk = _a[:, _lo : min(_hi, _a.shape[-1])]
                                if _blk.size == 0:
                                    logger.info("  action %-7s MISSING (array is only %d wide)", _nm, _a.shape[-1])
                                    continue
                                # Threshold above the 1e-6 epsilon in Unnormalize's quantile
                                # formula: padding dims have q01 == q99, so they come back as
                                # (x+1)/2 * 1e-6 rather than exactly zero.
                                _peak = np.abs(_blk).max(axis=0)
                                _live = [_lo + int(i) for i in np.flatnonzero(_peak > 1e-5)]
                                logger.info(
                                    "  action %-7s min=%+.4f max=%+.4f  live=%d/%d dims=%s",
                                    _nm,
                                    float(_blk.min()),
                                    float(_blk.max()),
                                    len(_live),
                                    _hi - _lo,
                                    _live if len(_live) <= 8 else f"{_live[:8]}+{len(_live) - 8}",
                                )
                        _wbc = _wbc_group if _group else self._scenario == "wbc"
                        result = _ul.to_result(
                            _acts,
                            arm_dof_per_side=arm_dof,
                            dex=(embodiment == "dex"),
                            emit_waist=_wbc,
                            emit_base=_wbc,
                        )
                    else:
                        raise ValueError("this release only serves unified62 checkpoints")
                except Exception:
                    await websocket.send(packer.pack(_server_error("Invalid policy output.", "INVALID_OUTPUT")))
                    continue

                resp = {"result": result}
                # Server timing piggy-backed alongside result for compat with PiPolicy clients
                # that ignored these fields; simulation clients should also ignore unknown keys.
                resp["server_timing"] = {"infer_ms": infer_time * 1000.0}
                if prev_total_time is not None:
                    resp["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                if self._save_request_dir and (
                    self._save_request_limit <= 0 or self._saved_count < self._save_request_limit
                ):
                    self._saved_count += 1
                    try:
                        # Simulation request/response records include parsed state and raw actions.
                        actions_raw = action.get("actions") if isinstance(action, dict) else action
                        actions_raw = None if actions_raw is None else np.asarray(actions_raw)
                        record = {
                            "index": self._saved_count,
                            "method": str(method) if method is not None else "infer",
                            "wall_time": time.time(),
                            # 原始请求：客户端到底发了什么字段/维度/图像编码
                            "request": req,
                            # 服务端解析后的中间量：state 组装结果、图像 shape/dtype、prompt
                            "parsed": obs,
                            "prompt": (obs or {}).get("prompt", ""),
                            "state": (obs or {}).get("state"),
                            "actions_raw": actions_raw,
                            "actions": actions_raw,
                            "result": resp.get("result"),
                            "error": "",
                            # 本服务独有：本体解析结果
                            "embodiment": embodiment,
                            "robot_type": robot_type,
                        }
                        path = _save_request(
                            record, self._save_request_dir, self._saved_count, str(method or "unknown")
                        )
                        logger.info("[save] %s", path)
                    except Exception:
                        logger.exception("failed to save request (non-fatal)")
                await websocket.send(packer.pack(resp))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(packer.pack(_server_error("Inference request failed.", "INTERNAL_ERROR")))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _safe_filename_part(value: object) -> str:
    text = str(value)
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text)
    return safe[:40] or "unknown"


def _save_request(record: dict, save_dir: str, index: int, method: str) -> str:
    """Dump a simulation request/response pair for field-by-field diagnostics."""
    filename = f"{time.time_ns()}_{index:06d}_{_safe_filename_part(method)}.pkl"
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
