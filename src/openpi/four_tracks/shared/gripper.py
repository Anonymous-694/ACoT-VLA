"""Per-embodiment gripper convention for mixing robots in one training run.

Different robots report the gripper in different units/signs (e.g. g2a_sim uses
-0.785..0 while a2d_sim uses 0..120 for state and 0..1 for action). To train them
together we map each robot's gripper to a single convention:

    0.0 = fully open, 1.0 = fully closed

via a per-robot, per-field linear map::

    unified = (raw - open) / (closed - open)
    raw     = unified * (closed - open) + open

Training applies the forward map per dataset (see transforms.GripperToOpenClosed) so
norm stats are computed on unified values. The inference server applies the SAME forward
map to the incoming gripper *state* and the inverse map to the outgoing gripper *action*
(see serving.websocket_policy_server), keeping train/inference in the same space.
"""

import numpy as np

from openpi.four_tracks.shared import robot_config as _robot_config

# Legacy hardcoded table — kept as fallback for datasets that predate robot_config.json.
GRIPPER_OPEN_CLOSED: dict[str, dict[str, tuple[float, float]]] = {
    "g2a_sim": {"state": (-0.785, 0.0), "action": (-0.785, 0.0)},
    "a2d_sim": {"state": (0.0, 120.0), "action": (0.0, 1.0)},
}


def bounds(robot_type: str | None, field: str, end_effector_type: str | None = None) -> tuple[float, float] | None:
    """(open, closed) raw values for a robot/field, or None if unknown.

    Lookup priority:
    1. robot_config.json by end_effector_type (as detailed_type or ee_type)
    2. robot_config.json by (robot_type, end_effector_type) pair
    3. Legacy GRIPPER_OPEN_CLOSED table by end_effector_type
    4. Legacy GRIPPER_OPEN_CLOSED table by robot_type
    """
    if end_effector_type:
        rc = _robot_config.get_gripper_bounds(
            end_effector_type,
            field,
            robot_type=robot_type,
            ee_type=end_effector_type,
        )
        if rc is not None:
            return rc
        conv = GRIPPER_OPEN_CLOSED.get(end_effector_type)
        if conv is not None:
            b = conv.get(field)
            if b is not None:
                return b
    if robot_type:
        rc = _robot_config.get_gripper_bounds(
            robot_type,
            field,
        )
        if rc is not None:
            return rc
    conv = GRIPPER_OPEN_CLOSED.get(robot_type)
    if conv is None:
        return None
    return conv.get(field)


def to_unified(raw, robot_type: str | None, field: str, end_effector_type: str | None = None):
    """raw -> unified [0,1] (0=open, 1=closed). Returns None if robot/field unknown."""
    b = bounds(robot_type, field, end_effector_type)
    if b is None:
        return None
    lo, hi = b
    return (np.asarray(raw, dtype=np.float64) - lo) / (hi - lo)


def to_raw(unified, robot_type: str | None, field: str, end_effector_type: str | None = None):
    """unified [0,1] -> raw robot units. Returns None if robot/field unknown."""
    b = bounds(robot_type, field, end_effector_type)
    if b is None:
        return None
    lo, hi = b
    return np.asarray(unified, dtype=np.float64) * (hi - lo) + lo
