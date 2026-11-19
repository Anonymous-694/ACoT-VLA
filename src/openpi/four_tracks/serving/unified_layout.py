"""Inference-side assembly/split for the 62-dim unified layout (LerobotUnifiedDataConfig).

Must mirror training (RemapToCanonicalGroups) exactly:
  [0:7] left arm | [7:14] right arm | [14:19] waist | [19:22] base vx,vy,yaw_rate
  | [22:42] left hand (20 slots) | [42:62] right hand (20 slots)

A 1-DoF gripper sits in slot 0 of its hand block (dim 22 / 42). Base STATE is never filled:
training masks it (two-frame finite difference there, single frame at serve time).
Grippers keep raw units; each embodiment has its own norm_stats.
"""

from __future__ import annotations

import numpy as np

ARM = 7
WAIST_START = 14
WAIST = 5
BASE_START = 19
BASE = 3
HAND_L = 22
HAND_R = 42
HAND = 20
DIM = 62


def _get(d, key):
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    return d.get(key.encode("utf-8"))


def _vec(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def assemble_state(states, *, arm_dof_per_side=None, dex=False, fill_waist=False):
    """Client states -> 62-dim state. Each arm is padded independently to 7 (6+6 robots)."""
    state = np.zeros(DIM, dtype=np.float64)
    dof = arm_dof_per_side or ARM
    arm = _get(states, "arm_joint_states")
    if arm is not None:
        arm = _vec(arm)
        left, right = arm[:dof], arm[dof : 2 * dof]
        state[: len(left)] = left
        state[ARM : ARM + len(right)] = right
    grip = _get(states, "gripper_states")
    if grip is not None:
        grip = _vec(grip)
        if dex:
            left, right = grip[:HAND], grip[HAND : 2 * HAND]
            state[HAND_L : HAND_L + len(left)] = left
            state[HAND_R : HAND_R + len(right)] = right
        else:
            if len(grip) > 0:
                state[HAND_L] = grip[0]
            if len(grip) > 1:
                state[HAND_R] = grip[1]
    if fill_waist:
        waist = _get(states, "waist_joint_states")
        if waist is not None:
            waist = _vec(waist)[:WAIST]
            state[WAIST_START : WAIST_START + len(waist)] = waist
    return state


def to_result(actions, *, arm_dof_per_side=None, dex=False, emit_waist=False, emit_base=False, fps=30.0):
    """Policy output (chunk, >=62) -> simulation result envelope. Blocks the robot lacks are dropped."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.shape[1] < DIM:
        raise ValueError(f"unified layout needs >= {DIM} action dims, got {actions.shape[1]}")
    dof = arm_dof_per_side or ARM
    if dex:
        left_eff = actions[:, HAND_L : HAND_L + HAND]
        right_eff = actions[:, HAND_R : HAND_R + HAND]
    else:
        left_eff = actions[:, HAND_L : HAND_L + 1]
        right_eff = actions[:, HAND_R : HAND_R + 1]
    result = {
        "left_arm": {"kind": "JOINT_ABS", "values": actions[:, :dof].tolist()},
        "right_arm": {"kind": "JOINT_ABS", "values": actions[:, ARM : ARM + dof].tolist()},
        "left_effector": left_eff.tolist(),
        "right_effector": right_eff.tolist(),
    }
    if emit_waist:
        result["waist"] = {"kind": "JOINT_ABS", "values": actions[:, WAIST_START : WAIST_START + WAIST].tolist()}
    if emit_base:
        vel = actions[:, BASE_START : BASE_START + BASE]
        result["wheels"] = np.cumsum(vel * (1.0 / fps), axis=0).tolist()
    return result
