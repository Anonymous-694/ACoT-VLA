"""Centralized robot embodiment configuration.

Loads ``robot_config.json`` (co-located with this module) and provides lookup
helpers so gripper bounds, prompt identifiers, and arm DOF info all come from
one place instead of scattered hardcoded tables.

JSON schema — keyed by a stable registry identifier::

    {
      "g2_ctek90d": {
        "robot_type": "g2a_sim",
        "ee_type": "zhiyuan_gripper_ctek90d",
        "end_effector_type": "ctek90d",
        "arm_dof_per_side": 7,
        "gripper": {
          "state": [<open>, <closed>],
          "action": [<open>, <closed>]
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "robot_config.json"

_CONFIGS: dict[str, dict] | None = None


def _ensure_loaded() -> dict[str, dict]:
    global _CONFIGS
    if _CONFIGS is None:
        if _CONFIG_PATH.exists():
            _CONFIGS = json.loads(_CONFIG_PATH.read_text())
            logger.info("Loaded %d robot configs from %s", len(_CONFIGS), _CONFIG_PATH)
        else:
            logger.warning("robot_config.json not found at %s", _CONFIG_PATH)
            _CONFIGS = {}
    return _CONFIGS


def all_configs() -> dict[str, dict]:
    return dict(_ensure_loaded())


def get_by_detailed_type(detailed_type: str) -> dict | None:
    return _ensure_loaded().get(detailed_type)


def get_by_robot_ee(robot_type: str, ee_type: str) -> tuple[str, dict] | None:
    """Look up config by (robot_type, ee_type) pair.

    Returns ``(detailed_type, config_dict)`` or ``None``.
    """
    for dt, cfg in _ensure_loaded().items():
        if cfg.get("robot_type") == robot_type and cfg.get("ee_type") == ee_type:
            return dt, cfg
    return None


def get_by_robot_type(robot_type: str) -> tuple[str, dict] | None:
    """Look up the first config matching *robot_type*.

    Returns ``(detailed_type, config_dict)`` or ``None``.
    """
    for dt, cfg in _ensure_loaded().items():
        if cfg.get("robot_type") == robot_type:
            return dt, cfg
    return None


def get_gripper_bounds(
    detailed_type: str | None = None,
    field: str = "state",
    *,
    robot_type: str | None = None,
    ee_type: str | None = None,
) -> tuple[float, float] | None:
    """Return ``(open, closed)`` gripper bounds from robot_config.json.

    Looks up by *detailed_type* first; falls back to *(robot_type, ee_type)* pair.
    Returns ``None`` if no match.
    """
    cfg = None
    if detailed_type:
        cfg = get_by_detailed_type(detailed_type)
    if cfg is None and robot_type and ee_type:
        result = get_by_robot_ee(robot_type, ee_type)
        if result:
            _, cfg = result
    if cfg is None:
        return None
    gripper = cfg.get("gripper")
    if not gripper:
        return None
    bounds = gripper.get(field)
    if bounds is None or len(bounds) != 2:
        return None
    return tuple(bounds)


def resolve_norm_group(
    robot_type: str,
    ee_type: str | None = None,
    scenario: str | None = None,
) -> str:
    """Return the per-robot norm_stats key (training ``norm_group_ids``) for a robot.

    Matches entries by ``robot_type`` (and ``ee_type`` when given). An entry carries either
    ``norm_group`` or a ``scenarios`` map (same body, several data families with different
    populated blocks, e.g. ctek90d WBC vs fine manipulation).

    Raises instead of guessing: an unknown robot, a scenario-dependent entry without a
    scenario, or candidates that disagree on the group. A wrong key would otherwise fall
    back to no normalization with no error.
    """
    cands = [
        (dt, cfg)
        for dt, cfg in _ensure_loaded().items()
        if cfg.get("robot_type") == robot_type
        and (not ee_type or ee_type in (cfg.get("ee_type"), cfg.get("end_effector_type")))
    ]
    if not cands:
        raise KeyError(f"no robot_config entry for robot_type={robot_type!r}, ee_type={ee_type!r}")
    # An explicit scenario narrows the candidates before anything else. Clients are not
    # required to send ee_type (simulation does not), so robot_type alone can match several
    # entries; when the operator started the server with --scenario, only entries that
    # declare that scenario can be meant. This is the operator's statement, not a guess.
    if scenario is not None:
        scoped = [(dt, cfg) for dt, cfg in cands if scenario in (cfg.get("scenarios") or {})]
        if scoped:
            cands = scoped

    groups: set[str] = set()
    for dt, cfg in cands:
        scenarios = cfg.get("scenarios")
        if scenarios:
            if scenario is None:
                raise ValueError(f"{dt} serves several data families; pass a scenario, one of {sorted(scenarios)}")
            if scenario not in scenarios:
                raise KeyError(f"{dt}: unknown scenario {scenario!r}, expected one of {sorted(scenarios)}")
            groups.add(scenarios[scenario])
        elif cfg.get("norm_group"):
            groups.add(cfg["norm_group"])
        else:
            raise KeyError(f"{dt} has no norm_group or scenarios in robot_config.json")
    if len(groups) > 1:
        raise ValueError(
            f"robot_type={robot_type!r}, ee_type={ee_type!r} is ambiguous: {sorted(groups)}; supply ee_type"
        )
    return groups.pop()


def validate_registry(registry):
    """Validate supported simulation kinematics without changing process state."""
    if not isinstance(registry, dict) or not registry:
        raise ValueError("robot registry must be a nonempty object")
    for name, cfg in registry.items():
        if cfg.get("arm_dof_per_side") not in (6, 7):
            raise ValueError(f"{name}: supported arm widths are 6 or 7")
        if cfg.get("hand_dof_per_side", 0) not in (0, 20):
            raise ValueError(f"{name}: unsupported hand width")
    return registry


def load_registry(path):
    """Use the registry packaged with the selected simulation checkpoint."""
    global _CONFIGS
    _CONFIGS = validate_registry(json.loads(Path(path).read_text()))
