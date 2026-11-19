# Derived from ACoT-VLA/openpi: src/openpi/transforms.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import numpy as np
from openpi_client import image_tools
import torch

from openpi.four_tracks.models import tokenizer as _tokenizer
from openpi.four_tracks.shared import gripper as _gripper
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


_MISSING: object = object()


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]
    # New-side paths ('/'-separated) that are silently dropped when their source key is
    # absent, instead of raising KeyError. This allows optional input fields while
    # preserving strict handling of required observations.
    optional: Sequence[str] = ()

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        optional = set(self.optional) if self.optional else set()
        result = self._build(self.structure, flat_item, optional, prefix="")
        if result is _MISSING:
            result = {}
        if "_robot_id" in data:
            result["_robot_id"] = data["_robot_id"]
        return result

    @classmethod
    def _build(cls, struct, flat_item, optional, prefix):
        if isinstance(struct, dict):
            out = {}
            for k, v in struct.items():
                child_path = f"{prefix}/{k}" if prefix else str(k)
                child = cls._build(v, flat_item, optional, child_path)
                if child is not _MISSING:
                    out[k] = child
            return out if out else _MISSING
        if struct in flat_item:
            return flat_item[struct]
        if prefix in optional:
            return _MISSING
        raise KeyError(struct)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01 = pad_to_dim(stats.q01, x.shape[-1], axis=-1, value=0.0)
        q99 = pad_to_dim(stats.q99, x.shape[-1], axis=-1, value=1.0)
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=False,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01 = pad_to_dim(stats.q01, x.shape[-1], axis=-1, value=0.0)
        q99 = pad_to_dim(stats.q99, x.shape[-1], axis=-1, value=1.0)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class InjectRobotId(DataTransformFn):
    """Inject a ``_robot_id`` key into the data dict."""

    robot_id: str

    def __call__(self, data: DataDict) -> DataDict:
        data["_robot_id"] = self.robot_id
        return data


@dataclasses.dataclass(frozen=True)
class PerRobotNormalize(DataTransformFn):
    """Normalize using per-robot norm_stats selected by a ``_robot_id`` key in the data.

    Falls back to *default_norm_stats* when ``_robot_id`` is absent or unknown.
    """

    norm_stats_map: dict[str, at.PyTree[NormStats]]
    default_norm_stats: at.PyTree[NormStats] | None = None
    use_quantiles: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        robot_id = data.pop("_robot_id", None)
        ns = self.norm_stats_map.get(robot_id) if robot_id else None
        if ns is None:
            if not self.default_norm_stats:
                raise ValueError(f"unknown or missing normalization group: {robot_id!r}")
            ns = self.default_norm_stats
        return Normalize(ns, use_quantiles=self.use_quantiles)(data)


@dataclasses.dataclass(frozen=True)
class PerRobotUnnormalize(DataTransformFn):
    """Unnormalize using per-robot norm_stats selected by a ``_robot_id`` key in the data."""

    norm_stats_map: dict[str, at.PyTree[NormStats]]
    default_norm_stats: at.PyTree[NormStats] | None = None
    use_quantiles: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        robot_id = data.get("_robot_id")
        ns = self.norm_stats_map.get(robot_id) if robot_id else None
        if ns is None:
            if not self.default_norm_stats:
                raise ValueError(f"unknown or missing normalization group: {robot_id!r}")
            ns = self.default_norm_stats
        return Unnormalize(ns, use_quantiles=self.use_quantiles)(data)


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int
    key: str = "image"

    def to_numpy(self, array_like):
        if hasattr(array_like, "cpu"):
            return array_like.detach().cpu().numpy()

        if hasattr(array_like, "__array__"):
            return np.array(array_like)

        return array_like

    def __call__(self, data: DataDict) -> DataDict:
        data[self.key] = {
            k: image_tools.resize_with_pad(self.to_numpy(v), self.height, self.width) for k, v in data[self.key].items()
        }
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = min(mask.shape[-1], state.shape[-1])
        actions[..., :dims] -= np.expand_dims(np.where(mask[:dims], state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = min(mask.shape[-1], state.shape[-1])
        actions[..., :dims] += np.expand_dims(np.where(mask[:dims], state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        # print(f"Prompt: {prompt}")
        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PrependRobotType(DataTransformFn):
    """Prepend "current robot is {robot_type}, " to the prompt.

    `robot_type` is constant per dataset (read from meta/info.json["robot_type"]).
    Must run after whatever transform sets "prompt" (PromptFromLeRobotTask). A no-op if robot_type is empty or there is
    no prompt to prepend to.
    """

    robot_type: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if not self.robot_type or "prompt" not in data:
            return data
        prompt = data["prompt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        else:
            prompt = str(prompt)
        return {**data, "prompt": f"current robot is {self.robot_type}, {prompt}"}


def _scale_gripper_indices(arr, indices, bounds):
    """Map raw gripper values at `indices` (last axis) to unified [0,1] (open=0/closed=1).

    Copies before writing so the underlying dataset frame is not mutated. Handles both
    torch tensors and numpy arrays.
    """
    lo, hi = bounds
    span = hi - lo
    if hasattr(arr, "clone"):  # torch.Tensor
        arr = arr.clone().to(torch.float32) if not torch.is_floating_point(arr) else arr.clone()
    else:
        arr = np.array(arr, dtype=np.float64, copy=True)
    for i in indices:
        arr[..., i] = (arr[..., i] - lo) / span
    return arr


@dataclasses.dataclass(frozen=True)
class GripperToOpenClosed(DataTransformFn):
    """Map a robot's raw gripper to the unified open=0/closed=1 convention (training side).

    Used when one training run mixes multiple embodiments whose grippers use different
    units/signs. Operates on the *raw* LeRobot frame (before repack/Go1Inputs), so it is
    given this dataset's raw effector indices within `state_key` / `action_key`. The same
    per-robot bounds live in `openpi.four_tracks.shared.gripper`; the inference server applies the
    forward map (state) and inverse map (action) using them.
    """

    robot_type: str | None
    state_key: str = "observation.state"
    action_key: str = "action"
    state_gripper_indices: tuple[int, ...] = ()
    action_gripper_indices: tuple[int, ...] = ()
    end_effector_type: str | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if not self.robot_type and not self.end_effector_type:
            return data
        out = dict(data)
        state_bounds = _gripper.bounds(self.robot_type, "state", self.end_effector_type)
        if state_bounds is not None and self.state_key in out and self.state_gripper_indices:
            out[self.state_key] = _scale_gripper_indices(out[self.state_key], self.state_gripper_indices, state_bounds)
        action_bounds = _gripper.bounds(self.robot_type, "action", self.end_effector_type)
        if action_bounds is not None and self.action_key in out and self.action_gripper_indices:
            out[self.action_key] = _scale_gripper_indices(
                out[self.action_key], self.action_gripper_indices, action_bounds
            )
        return out


@dataclasses.dataclass(frozen=True)
class RemapGo1FieldsToCanonical(DataTransformFn):
    """Reorder a dataset's raw state/action into the canonical
    [joint(14), left_effector, right_effector, waist...] layout using THIS dataset's own
    field indices, then zero-pad the (variable-width) waist block to ``canonical_dim``.

    Runs per-dataset on the raw LeRobot frame (before repack). Because each dataset supplies
    its own indices, embodiments with different raw layouts and waist widths (G1: 2 waist
    joints, G2: 5) all land in one uniform layout, so the shared downstream Go1Inputs needs
    no per-robot remap. Fixes the silent no-op where a single global index set (from
    repo_id[0]) skipped remapping on smaller-dim embodiments and mislabeled their action dims.
    """

    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    canonical_dim: int
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if self.state_key in out and self.state_indices:
            out[self.state_key] = self._gather(out[self.state_key], self.state_indices)
        if self.action_key in out and self.action_indices:
            out[self.action_key] = self._gather(out[self.action_key], self.action_indices)
        return out

    def _gather(self, arr, indices):
        idx = list(indices)
        if hasattr(arr, "clone"):  # torch.Tensor
            g = arr[..., idx]
            pad = self.canonical_dim - g.shape[-1]
            if pad > 0:
                zeros = torch.zeros(*g.shape[:-1], pad, dtype=g.dtype)
                g = torch.cat([g, zeros], dim=-1)
            elif pad < 0:
                g = g[..., : self.canonical_dim]
            return g
        a = np.asarray(arr)
        g = a[..., idx]
        return pad_to_dim(g, self.canonical_dim, axis=-1)


@dataclasses.dataclass(frozen=True)
class RemapToCanonicalGroups(DataTransformFn):
    """Gather and pad each field group independently, then concatenate.

    Generalisation of ``RemapGo1FieldsToCanonical`` for variable-DOF cross-embodiment
    training.  Each group is a ``(source_indices, canonical_size)`` pair.  The transform
    gathers the source indices from the raw array, zero-pads (or truncates) to the
    canonical size, and concatenates the groups along the last axis.

    Example — aligning a 6+6 arm robot with a 7+7 arm robot (both with 2 grippers)::

        groups = [
            ((0,1,2,3,4,5), 7),       # left arm: 6 → pad to 7
            ((6,7,8,9,10,11), 7),      # right arm: 6 → pad to 7
            ((12, 13), 2),             # grippers: unchanged
        ]
        # output: [l1..l6, 0, r1..r6, 0, g_l, g_r] = 16 dims
    """

    state_groups: tuple[tuple[tuple[int, ...], int], ...]
    action_groups: tuple[tuple[tuple[int, ...], int], ...]
    state_key: str = "observation.state"
    action_key: str = "action"
    # Optional per-group (state, action) source keys. When entry ``i`` is not None, group
    # ``i`` takes ``data[source_key]`` wholesale (padded/truncated to the group size) instead
    # of gathering indices from the raw feature array. Used for blocks that have no field in
    # the dataset and are produced by an earlier transform, e.g. computed base velocity.
    state_source_keys: tuple[str | None, ...] | None = None
    action_source_keys: tuple[str | None, ...] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        if self.state_key in out and self.state_groups:
            out[self.state_key] = self._apply(out[self.state_key], self.state_groups, out, self.state_source_keys)
        if self.action_key in out and self.action_groups:
            out[self.action_key] = self._apply(out[self.action_key], self.action_groups, out, self.action_source_keys)
        return out

    @staticmethod
    def _apply(arr, groups, data=None, source_keys=None):
        parts = []
        for gi, (indices, size) in enumerate(groups):
            src = source_keys[gi] if source_keys and gi < len(source_keys) else None
            if src is not None:
                parts.append(RemapToCanonicalGroups._flatten_source(data, src, size))
                continue
            idx = list(indices)
            if hasattr(arr, "clone"):  # torch.Tensor
                g = arr[..., idx]
                pad_n = size - g.shape[-1]
                if pad_n > 0:
                    g = torch.cat([g, torch.zeros(*g.shape[:-1], pad_n, dtype=g.dtype)], dim=-1)
                elif pad_n < 0:
                    g = g[..., :size]
            else:
                g = pad_to_dim(np.asarray(arr)[..., idx], size, axis=-1)
            parts.append(g)
        if hasattr(parts[0], "clone"):
            return torch.cat(parts, dim=-1)
        return np.concatenate(parts, axis=-1)

    @staticmethod
    def _flatten_source(data, key, size):
        """Take ``data[key]`` as-is and pad/truncate its last axis to ``size``."""
        val = data[key]
        if hasattr(val, "clone"):  # torch.Tensor
            val = val.reshape(*val.shape[:-1], -1)
            pad_n = size - val.shape[-1]
            if pad_n > 0:
                val = torch.cat([val, torch.zeros(*val.shape[:-1], pad_n, dtype=val.dtype)], dim=-1)
            return val[..., :size]
        val = np.asarray(val).reshape(*np.shape(val)[:-1], -1)
        return pad_to_dim(val, size, axis=-1)


@dataclasses.dataclass(frozen=True)
class LinearMapDims(DataTransformFn):
    """Apply ``new = scale * raw + offset`` to the given last-axis indices.

    Used to bring a dataset's raw gripper unit onto the convention its embodiment is trained
    with, without rewriting the dataset on disk. Applied to both state and action.
    """

    indices: Sequence[int]
    scale: float
    offset: float = 0.0
    keys: Sequence[str] = ("observation.state", "action")

    def __call__(self, data: DataDict) -> DataDict:
        out = dict(data)
        idx = list(self.indices)
        for k in self.keys:
            if k not in out:
                continue
            arr = out[k]
            arr = arr.clone() if hasattr(arr, "clone") else np.array(arr, copy=True)
            arr[..., idx] = arr[..., idx] * self.scale + self.offset
            out[k] = arr
        return out


@dataclasses.dataclass(frozen=True)
class ComputeBaseVelocity(DataTransformFn):
    """Compute body-frame base velocity from consecutive absolute poses.

    Requires observation.state to have shape [2, D] (previous + current frame,
    via delta_timestamps with [-1/fps, 0.0]). Extracts position (x, y) and
    orientation (qz, qw) from both frames, computes body-frame displacement,
    and derives velocity. Stores result in data["base_velocity"] as [3] (vx, vy,
    yaw_rate). Replaces observation.state with the current frame only ([D]).

    Must run BEFORE RemapGo1FieldsToCanonical (needs raw 172-dim indices).
    """

    fps: float = 30.0
    position_indices: Sequence[int] = (128, 129)
    orientation_indices: Sequence[int] = (133, 134)
    # Set False for datasets that have no chassis pose at these indices, so the velocity is
    # reported as zero instead of being read out of whatever happens to sit there. In a mixed
    # run the indices are only meaningful for the embodiment they were measured on: a wider
    # raw state (e.g. a 328-dim dexterous-hand frame) would otherwise silently yield a
    # fabricated velocity computed from finger joints, and a narrower one would IndexError --
    # which SafeDataset swallows by resampling, silently dropping that embodiment's frames.
    enabled: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        state = np.asarray(data["observation.state"])

        needed = max(*self.position_indices, *self.orientation_indices) + 1
        if not self.enabled or state.shape[-1] < needed:
            if state.ndim == 2 and state.shape[0] == 2:
                data["observation.state"] = state[1]
            data["base_velocity"] = np.zeros(3, dtype=np.float32)
            return data

        if state.ndim == 2 and state.shape[0] == 2:
            prev, curr = state[0], state[1]

            x_prev, y_prev = prev[self.position_indices[0]], prev[self.position_indices[1]]
            x_curr, y_curr = curr[self.position_indices[0]], curr[self.position_indices[1]]
            qz_prev, qw_prev = prev[self.orientation_indices[0]], prev[self.orientation_indices[1]]
            qz_curr, qw_curr = curr[self.orientation_indices[0]], curr[self.orientation_indices[1]]

            yaw_prev = np.arctan2(2 * qw_prev * qz_prev, 1 - 2 * qz_prev**2)
            yaw_curr = np.arctan2(2 * qw_curr * qz_curr, 1 - 2 * qz_curr**2)

            dx_odom = x_curr - x_prev
            dy_odom = y_curr - y_prev
            dyaw = yaw_curr - yaw_prev

            dx_body = np.cos(yaw_curr) * dx_odom + np.sin(yaw_curr) * dy_odom
            dy_body = -np.sin(yaw_curr) * dx_odom + np.cos(yaw_curr) * dy_odom

            dt = 1.0 / self.fps
            base_vel = np.array([dx_body / dt, dy_body / dt, dyaw / dt], dtype=np.float32)

            data["observation.state"] = curr
        else:
            base_vel = np.zeros(3, dtype=np.float32)

        data["base_velocity"] = base_vel
        return data


@dataclasses.dataclass(frozen=True)
class AppendBaseVelocity(DataTransformFn):
    """Write base_velocity into the zero-padded slots of the state vector.

    After RemapGo1FieldsToCanonical, state is [n_joints, ...padding...]. This
    transform writes the computed base velocity into the padding slots starting
    at base_start, so the velocity occupies dims [base_start : base_start+3]
    instead of being appended past the mask boundary.
    """

    base_start: int = 21

    def __call__(self, data: DataDict) -> DataDict:
        if "base_velocity" in data:
            state_key = "state" if "state" in data else "observation.state"
            state = np.asarray(data[state_key]).copy()
            base_vel = np.asarray(data["base_velocity"])
            state[self.base_start : self.base_start + len(base_vel)] = base_vel
            data[state_key] = state
            del data["base_velocity"]
        return data


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad or truncate an array to the target dimension along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    if current_dim > target_dim:
        slices = [slice(None)] * len(x.shape)
        slices[axis] = slice(0, target_dim)
        return x[tuple(slices)]
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
