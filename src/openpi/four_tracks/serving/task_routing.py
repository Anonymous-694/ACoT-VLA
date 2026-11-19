"""Route a client's ``task_name`` to the per-robot norm_stats group it was trained under.

Why this exists: a unified checkpoint (e.g. ``four-track unified62``) is one
model but seven sets of normalization statistics, and the group also decides the prompt
prefix, whether the waist/base blocks are populated, and whether the hand is a 2-slot gripper
or a 20-joint dexterous hand. The payload tags cannot tell the groups apart on their own --
simulation reports ``robot_type="g2a_sim"`` for ctek90d, omnipicker and wuji alike, and WBC vs
fine manipulation share every tag there is. Before this module the operator had to declare the
family at startup (``--scenario`` / ``--norm_group``), which forced one server process per
family. Routing per request on ``task_name`` lets a single process serve all of them.

Picking the wrong group does NOT raise anywhere downstream: ``PerRobotNormalize`` falls back
to ``default_norm_stats`` (``None`` for these configs), ``Normalize(None)`` is a no-op, and the
model infers on unnormalized input. The server answers normally and only the actions are
garbage. So every failure mode here is a loud error, never a guess.
"""

from __future__ import annotations

import difflib
import json
import pathlib
from typing import Any

# Shipped table for the unified 62-dim v2 checkpoint.
DEFAULT_ROUTES = pathlib.Path(__file__).with_name("task_routes_unified4v2.json")

# Keys in the JSON that carry documentation rather than a route.
_COMMENT_KEYS = ("_comment",)


def normalize(task_name: Any) -> str:
    """Canonical form used for both table keys and lookups."""
    if isinstance(task_name, bytes):
        task_name = task_name.decode("utf-8", "replace")
    return str(task_name or "").strip().lower()


def load(path: str | pathlib.Path | None = None) -> dict[str, str | dict[str, str]]:
    """Load a routing table. Values are a group name, or a ``{robot_type: group}`` map for a
    task name that several embodiments share."""
    path = pathlib.Path(path) if path else DEFAULT_ROUTES
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: routing table must be a JSON object, got {type(raw).__name__}")

    table: dict[str, str | dict[str, str]] = {}
    for key, value in raw.items():
        if key in _COMMENT_KEYS:
            continue
        norm_key = normalize(key)
        if not norm_key:
            raise ValueError(f"{path}: empty task name in the table")
        if norm_key in table:
            raise ValueError(f"{path}: duplicate task name {norm_key!r} (names are case-insensitive)")
        if isinstance(value, str):
            table[norm_key] = value
        elif isinstance(value, dict):
            if not value:
                raise ValueError(f"{path}: {key!r} maps to an empty robot_type branch")
            branches = {}
            for rt, group in value.items():
                if not isinstance(group, str):
                    raise ValueError(f"{path}: {key!r}/{rt!r} must map to a group name, got {group!r}")
                branches[str(rt)] = group
            table[norm_key] = branches
        else:
            raise ValueError(f"{path}: {key!r} must map to a group name or a robot_type object")
    if not table:
        raise ValueError(f"{path}: routing table is empty")
    return table


def groups(table: dict[str, str | dict[str, str]]) -> set[str]:
    """Every group the table can produce."""
    out: set[str] = set()
    for value in table.values():
        if isinstance(value, dict):
            out.update(value.values())
        else:
            out.add(value)
    return out


def by_group(table: dict[str, str | dict[str, str]]) -> dict[str, list[str]]:
    """``{group: [task names]}``, for the startup log."""
    out: dict[str, list[str]] = {}
    for name, value in sorted(table.items()):
        for group in sorted(set(value.values())) if isinstance(value, dict) else [value]:
            out.setdefault(group, []).append(name)
    return out


def validate(table: dict[str, str | dict[str, str]], known_groups) -> None:
    """Fail at startup when the table names a group the served config never trained with.

    ``known_groups`` is the config's ``norm_group_ids`` / ``per_robot_norm_stats`` keys. A typo
    caught here is a crash at launch instead of unnormalized inference discovered at eval time.
    """
    known = set(known_groups or ())
    if not known:
        return
    unknown = sorted(groups(table) - known)
    if unknown:
        raise ValueError(
            f"routing table references norm groups the served config does not have: {unknown}; "
            f"the config knows {sorted(known)}"
        )


class TaskRouteError(KeyError):
    """Raised when a request cannot be routed. The message is sent back to the client."""


def resolve(
    task_name: Any,
    robot_type: Any,
    table: dict[str, str | dict[str, str]],
    *,
    param_keys=None,
) -> str:
    """Return the norm group for this request, or raise ``TaskRouteError`` with diagnostics.

    ``param_keys`` is echoed into the error so one rejected request shows what the client
    actually sent -- the field names, and the raw ``task_name`` including whether it was absent
    versus an empty string.
    """
    key = normalize(task_name)
    rt = normalize(robot_type)

    if not key:
        raise TaskRouteError(
            "request has no usable task_name, so its norm group cannot be determined "
            f"(task_name={task_name!r}, robot_type={robot_type!r}, "
            f"payload keys={sorted(param_keys) if param_keys is not None else 'n/a'}). "
            "Either have the client send task_name, or start the server with "
            "--norm_group/--scenario to pin one family per process."
        )

    value = table.get(key)
    if value is None:
        close = difflib.get_close_matches(key, table.keys(), n=3, cutoff=0.6)
        hint = f" Did you mean {close}?" if close else ""
        raise TaskRouteError(
            f"task_name={task_name!r} is not in the routing table ({len(table)} names known), "
            f"robot_type={robot_type!r}, payload keys="
            f"{sorted(param_keys) if param_keys is not None else 'n/a'}.{hint} "
            "Add it to the --task_routes JSON; refusing rather than guessing, because a wrong "
            "group silently disables normalization instead of raising."
        )

    if isinstance(value, str):
        return value

    # Shared task name: several embodiments send the same string, so robot_type picks the branch.
    group = value.get(rt) or value.get(str(robot_type or ""))
    if group is None:
        raise TaskRouteError(
            f"task_name={task_name!r} is shared by several embodiments and robot_type="
            f"{robot_type!r} is not one of them ({sorted(value)}); cannot pick a norm group."
        )
    return group
