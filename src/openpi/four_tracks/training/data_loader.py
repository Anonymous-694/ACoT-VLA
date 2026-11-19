# Derived from ACoT-VLA/openpi: src/openpi/training/data_loader.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import random
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

from openpi.four_tracks.training import dataset_v21
import openpi.four_tracks.training.config as _config
import openpi.four_tracks.transforms as _transforms
import openpi.models.model as _model

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class SafeDataset(Dataset):
    def __init__(self, dataset: Dataset, seed: int = 0):
        self.dataset = dataset
        # Per-instance RNG so resampling on errors is deterministic and worker-local.
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: SupportsIndex):
        try:
            val = self.dataset[index]
            if val is None:
                raise ValueError("Dataset returned None")
            return val
        except Exception as e:
            print(f"[Data Load Error] Index {index} failed due to: {e}. Resampling...")
            new_index = self._rng.randint(0, len(self.dataset) - 1)
            return self.__getitem__(new_index)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'dataset'")

        return getattr(self.dataset, name)


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        if not hasattr(self._dataset, "_datasets"):
            item = self._dataset[index]
            return self._transform(item)
        else:
            idx = index.__index__()
            for d in self._dataset._datasets:
                if idx < len(d):
                    item = d[idx]
                    return self._transform(item)
                idx -= len(d)
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        if not hasattr(self._dataset, "_datasets"):
            length = len(self._dataset)
        else:
            length = 0
            for item in self._dataset._datasets:
                length += len(item)
        return length


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = np.random.default_rng(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            if spec.dtype == jnp.int32:
                return rng.integers(0, 2048, size=shape, dtype=np.int32)
            return np.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


# Coarse prompt-prefix aliases: some exports tag the same embodiment without the "_sim"
# suffix. The inference server (robot_config.json) only ever sends the suffixed form, so
# training must use it too or those datasets get a prefix that is never seen at serve time.
_COARSE_PREFIX_ALIASES = {"arx": "arx_sim", "agilex": "agilex_sim"}


def _coarse_prefix(robot_type):
    return _COARSE_PREFIX_ALIASES.get(robot_type, robot_type)


def _norm_group_for_prefix(data_config, n, total):
    ids = data_config.norm_group_ids
    if ids is None or len(ids) != total:
        raise ValueError("prepend_robot_type_from_norm_group needs norm_group_ids parallel to repo_id")
    return ids[n]


# agilex raw gripper unit (metres, 0.04 = open) -> trained convention (-3.5 = open, 0 = closed).
# Same linear map as scripts/convert_gripper_convention.py: new = -87.5 * raw.
_AGILEX_GRIPPER_SCALE = -3.5 / 0.04 * -1.0 * -1.0
_AGILEX_EMBEDDED_GRIPPER_IDX = (6, 13)


def _has_embedded_gripper(repo_id: str, field_map=None) -> bool:
    """agilex 14-dim format: gripper sits inside action/joint/position (idx 6 and 13).

    Detected as: robot_type is agilex, no effector field, and a 14-wide joint field.
    throw_garbage_aloha (12 joints + separate effector fields) is not affected.
    """
    import json
    import pathlib

    try:
        info = json.loads((pathlib.Path(repo_id) / "meta" / "info.json").read_text())
        fd = dataset_v21.feature_fields(info, "action", field_map)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    if not str(info.get("robot_type", "")).startswith("agilex"):
        return False
    if "action/left_effector/position" in fd:
        return False
    if len(fd.get("action/joint/position", {}).get("indices", ())) != 14:
        return False
    return True


def _embedded_gripper_needs_rescale(repo_id: str, field_map=None) -> bool:
    """Whether an embedded AgileX gripper still needs conversion from raw metres."""
    if not _has_embedded_gripper(repo_id, field_map):
        return False

    # Structure alone cannot tell a converted export from an unconverted one: both keep the
    # gripper at idx 6/13 inside a 14-wide joint field. scripts/convert_gripper_convention.py
    # rewrites those columns in place, and applying the rescale on top of that would silently
    # turn -3.5~0 into 0~+306. Decide on the VALUES instead.
    converted = _agilex_gripper_already_converted(repo_id)
    if converted is True:
        print(
            f"[canonical_groups] {repo_id}: gripper already in the trained -3.5~0 convention "
            f"on disk -> skipping the x{_AGILEX_GRIPPER_SCALE} rescale"
        )
        return False
    if converted is None:
        print(
            f"[canonical_groups] WARNING {repo_id}: could not read the gripper columns to check "
            f"whether this export was already converted; applying the x{_AGILEX_GRIPPER_SCALE} "
            f"rescale on the structural match alone. If the data WAS converted this doubles it "
            f"to 0~+306 and only a loss blow-up would show it."
        )
    return True


def _agilex_gripper_already_converted(repo_id: str) -> bool | None:
    """Whether idx 6/13 are already in the trained convention. None when undecidable.

    Every destination convention has closed == 0 as its MAXIMUM, so a converted export peaks at
    0 while raw metres peak at +0.04 -- the same test convert_gripper_convention.py uses to stay
    idempotent. One parquet file is enough; this runs once per dataset at construction time.
    """
    import glob
    import pathlib

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    files = sorted(glob.glob(str(pathlib.Path(repo_id) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        return None
    try:
        rows = pq.read_table(files[0], columns=["action"]).column("action").to_pylist()
        arr = np.asarray(rows, dtype=np.float64)
    except Exception:  # noqa: BLE001 - an unreadable probe must not break dataset construction
        return None
    if arr.ndim != 2 or arr.shape[1] <= max(_AGILEX_EMBEDDED_GRIPPER_IDX):
        return None
    return float(arr[:, list(_AGILEX_EMBEDDED_GRIPPER_IDX)].max()) <= 1e-3


def _embedded_gripper_override(group_name: str) -> tuple[int, ...] | None:
    return {
        "arm_left": (0, 1, 2, 3, 4, 5),
        "arm_right": (7, 8, 9, 10, 11, 12),
        "hand_left": (6,),
        "hand_right": (13,),
    }.get(group_name)


# Public hand contract: five fingers, thumb to pinky, four joints per finger.
_WUJI_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def _check_joint_names(repo_id: str, group_name: str, feature_key: str, names: Sequence[str]) -> None:
    """Fail loudly when a field's joint_names contradict the convention the clients send.

    Only validates what the names actually pin down, and is a no-op for exports without them
    when an export has no joint-name metadata.
    """
    if not names:
        return
    lowered = [str(n).lower() for n in names]

    if group_name in ("hand_left", "hand_right") and len(lowered) == 20:
        if not all("_joint" in n for n in lowered):
            # A hand named some other way is not necessarily wrong, and blocking a 5-hour norm
            # run over an unrecognised convention would be worse than letting it through. Only
            # a detected REORDER of the known convention is an error.
            logging.warning(
                "[canonical_groups] %s: %s joint_names do not follow the <finger>_joint<n> "
                "convention, so the finger order cannot be checked: %s",
                repo_id,
                feature_key,
                list(names)[:4],
            )
            return
        seen = [n.split("_joint")[0].split("_")[-1] for n in lowered]
        got = tuple(dict.fromkeys(seen))
        if set(got) != set(_WUJI_FINGER_ORDER):
            logging.warning(
                "[canonical_groups] %s: %s names %s are not the five known fingers; order not checked",
                repo_id,
                feature_key,
                got,
            )
            return
        if got != _WUJI_FINGER_ORDER:
            raise ValueError(
                f"{repo_id}: {feature_key} finger order is {got}, expected {_WUJI_FINGER_ORDER}.\n"
                f"  The eval client sends thumb->pinky (geniesim_benchmark WUJI_FINGERS); training "
                f"on a different order silently teaches the model the wrong fingers.\n"
                f"  joint_names = {list(names)}"
            )
        per_finger = [lowered[i : i + 4] for i in range(0, 20, 4)]
        for finger, block in zip(_WUJI_FINGER_ORDER, per_finger):
            want = [f"joint{j}" for j in range(1, 5)]
            if [b.split("_")[-1] for b in block] != want:
                raise ValueError(
                    f"{repo_id}: {feature_key} {finger} joints are {[b.split('_')[-1] for b in block]}, expected {want}"
                )

    elif group_name in ("arm_left", "arm_right", "waist"):
        # Only the ordering is checked: the joint token must ascend 1..N with no gaps. Names
        # differ per robot (arm_l_joint1 / arm_fl_joint1 / body_joint1), so matching the text
        # would reject valid embodiments.
        #
        # A PAIRED arm field concatenates both arms, so each half restarts at 1 (the G2 export
        # is 1..7 1..7, not 1..14) -- `_shape` splits it in half afterwards. Accept either a
        # single ascending run (waist, one arm) or two equal halves that each ascend 1..N.
        nums = []
        for n in lowered:
            tail = n.rsplit("joint", 1)[-1]
            if tail.isdigit():
                nums.append(int(tail))

        def _ascending(seq: list[int]) -> bool:
            return seq == list(range(1, len(seq) + 1))

        if nums:
            ok = _ascending(nums)
            if not ok and len(nums) % 2 == 0:
                half = len(nums) // 2
                ok = _ascending(nums[:half]) and _ascending(nums[half:])
            if not ok:
                raise ValueError(
                    f"{repo_id}: {feature_key} joints are neither ascending 1..{len(nums)} nor "
                    f"two ascending halves: {nums}\n  joint_names = {list(names)}"
                )


def _joint_names(repo_id: str, feature_key: str, field: str, field_map=None) -> tuple[str, ...]:
    """Optional public joint-name metadata for one semantic field."""
    import json
    import pathlib

    try:
        info = json.loads((pathlib.Path(repo_id) / "meta" / "info.json").read_text())
        entry = dataset_v21.feature_fields(info, feature_key, field_map)[field]
    except (FileNotFoundError, KeyError, ValueError):
        return ()
    return tuple(entry.get("joint_names") or ())


def _resolve_group_indices(
    repo_id: str, group: _config.CanonicalGroup, norm_group_id: str | None = None, field_map=None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve one canonical group's raw state/action indices from a dataset's info.json.

    A group whose field is absent yields ``()`` for that side, which RemapToCanonicalGroups
    turns into an all-zero block of the group's canonical size -- that is how a robot with no
    waist/chassis/dexterous hand still lines up with the others instead of raising.
    """
    from openpi.four_tracks.policies import go1_policy

    if group.fill_only_for is not None:
        if len(group.fill_only_for) == 0:
            # Empty allow-list: no dataset may populate this block, so no id is needed.
            return (), ()
        if norm_group_id is None:
            raise ValueError(
                f"block '{group.name}' is restricted via fill_only_for={list(group.fill_only_for)} "
                f"but no norm_group_id was given for {repo_id}; set norm_group_ids on the config."
            )
        if norm_group_id not in group.fill_only_for:
            return (), ()

    def _indices(feature_key: str, keys: Sequence[str]) -> tuple[int, ...]:
        if not keys:
            return ()
        try:
            idx = tuple(go1_policy.load_field_indices(repo_id, feature_key, tuple(keys), field_map=field_map))
            for key in keys:
                _check_joint_names(repo_id, group.name, key, _joint_names(repo_id, feature_key, key, field_map))
            return idx
        except (FileNotFoundError, KeyError) as e:
            logging.warning(
                "[canonical_groups] %s: field %s not usable for %s (%s); block will be zero-filled",
                group.name,
                keys,
                repo_id,
                e,
            )
            return ()

    def _shape(idx: tuple[int, ...], split: str | None, select: Sequence[int] | None) -> tuple[int, ...]:
        out = idx
        if split == "left":
            out = out[: len(out) // 2]
        elif split == "right":
            out = out[len(out) // 2 :]
        if select is not None:
            # Accept either planar (vx, vy, yaw_rate) or 6D linear/angular commands.
            if group.name == "base" and len(out) == 3 and tuple(select) == (0, 1, 5):
                select = (0, 1, 2)
            out = tuple(out[i] for i in select if i < len(out))
        return out

    if _has_embedded_gripper(repo_id, field_map):
        override = _embedded_gripper_override(group.name)
        if override is not None:
            return override, override

    s_idx = _shape(_indices("observation.state", group.state_keys), group.split, group.state_select)
    a_idx = _shape(_indices("action", group.action_keys), group.split, group.action_select)
    return s_idx, a_idx


def _canonical_group_transforms(
    repo: str, data_config: _config.DataConfig, norm_group_id: str | None = None
) -> list[_transforms.DataTransformFn]:
    """Build one dataset's unified-layout transforms: base velocity (if any) then the remap.

    Ordering matters: ComputeBaseVelocity reads RAW observation.state indices, which stop
    existing once RemapToCanonicalGroups has reordered the frame. It also collapses the
    two-frame [2, D] window that include_prev_state produces down to the current frame, so it
    must run before anything that assumes a 1-D state.
    """
    from pathlib import Path

    field_map = data_config.dataset_field_maps.get(str(Path(repo).resolve()), {})
    groups = tuple(data_config.canonical_groups or ())
    out: list[_transforms.DataTransformFn] = []

    s_groups, a_groups = [], []
    # Whether THIS dataset actually has a chassis. The raw pose indices are only meaningful
    # for the embodiment they were measured on, so a dataset without the velocity field must
    # not have them read out of whatever sits at those positions (see ComputeBaseVelocity).
    has_chassis = False
    for g in groups:
        gi_s, gi_a = _resolve_group_indices(repo, g, norm_group_id, field_map)
        if g.state_source == "base_velocity" and gi_a:
            has_chassis = True
        # A block absent on one side is still emitted at full width, so block boundaries are
        # identical across embodiments.
        s_groups.append((gi_s, g.size))
        a_groups.append((gi_a, g.size))
        if g.state_keys and not gi_s:
            print(f"[canonical_groups] {repo}: state block '{g.name}' missing -> zeros")
        if g.action_keys and not gi_a:
            print(f"[canonical_groups] {repo}: action block '{g.name}' missing -> zeros")

    if _embedded_gripper_needs_rescale(repo, field_map):
        out.append(_transforms.LinearMapDims(indices=_AGILEX_EMBEDDED_GRIPPER_IDX, scale=_AGILEX_GRIPPER_SCALE))
        print(f"[canonical_groups] {repo}: agilex embedded gripper idx 6/13 -> x{_AGILEX_GRIPPER_SCALE}")

    if data_config.canonical_base_velocity is not None:
        pos_idx, ori_idx = data_config.canonical_base_velocity
        out.append(
            _transforms.ComputeBaseVelocity(
                position_indices=tuple(pos_idx),
                orientation_indices=tuple(ori_idx),
                enabled=has_chassis,
            )
        )
        if not has_chassis:
            print(f"[canonical_groups] {repo}: no chassis velocity field -> base state = 0")

    out.append(
        _transforms.RemapToCanonicalGroups(
            state_groups=tuple(s_groups),
            action_groups=tuple(a_groups),
            state_source_keys=tuple(g.state_source for g in groups),
        )
    )
    return out


class MixedDataset(torch.utils.data.ConcatDataset):
    """Concatenate individually adapted datasets without aggregating raw-width stats."""

    def __init__(self, datasets):
        super().__init__(datasets)
        self._datasets = self.datasets


def create_torch_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Read native v2.1 datasets and adapt each embodiment before concatenation."""
    from pathlib import Path

    repo_id = data_config.repo_id
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    repos = repo_id if isinstance(repo_id, list) else [repo_id]
    if not repos:
        raise ValueError("At least one v2.1 dataset is required")
    ids = data_config.norm_group_ids
    if ids is not None and len(ids) != len(repos):
        raise ValueError("norm_group_ids must be parallel to the v2.1 datasets")
    if data_config.dataloader_sampler:
        raise ValueError("The v2.1 four-track loader uses all frames; sampler must be empty")
    datasets = []
    for n, repo in enumerate(repos):
        info = dataset_v21.require_v21(repo)
        fps = float(info["fps"])
        delta_timestamps = {
            key: [t / fps for t in range(model_config.action_horizon)] for key in data_config.action_sequence_keys
        }
        if data_config.include_prev_state:
            delta_timestamps["observation.state"] = [-1 / fps, 0.0]
        dataset = lerobot_dataset.LeRobotDataset(str(repo), root=Path(repo), delta_timestamps=delta_timestamps)
        gid = ids[n] if ids is not None else info.get("robot_type")
        transforms = []
        if data_config.prompt_from_task:
            transforms.append(_transforms.PromptFromLeRobotTask(dataset.meta.tasks))
        if data_config.prepend_robot_type:
            prefix = gid if data_config.prepend_robot_type_from_norm_group else _coarse_prefix(info.get("robot_type"))
            transforms.append(_transforms.PrependRobotType(prefix))
        if data_config.canonical_groups:
            transforms.extend(_canonical_group_transforms(repo, data_config, gid))
        if data_config.per_robot_norm_stats:
            if gid not in data_config.per_robot_norm_stats:
                raise ValueError(f"No normalization statistics for v2.1 dataset group {gid!r}")
            transforms.append(_transforms.InjectRobotId(gid))
        datasets.append(TransformedDataset(dataset, transforms))
    return MixedDataset(datasets) if isinstance(repo_id, list) else datasets[0]


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    if not skip_norm_stats and data_config.per_robot_norm_stats:
        normalize_transform = _transforms.PerRobotNormalize(
            norm_stats_map=data_config.per_robot_norm_stats,
            default_norm_stats=data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
    elif not skip_norm_stats:
        if data_config.norm_stats is None and data_config.repo_id != "fake":
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_four_track_norm_stats.py --manifest <manifest>`."
            )
        normalize_transform = _transforms.Normalize(
            data_config.norm_stats or {},
            use_quantiles=data_config.use_quantile_norm,
        )
    else:
        normalize_transform = _transforms.Normalize({})

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            normalize_transform,
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    dataset = SafeDataset(dataset, seed=seed)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        sampler=None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def close(self):
        iterator = getattr(self._data_loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            self._data_loader._iterator = None

    def __iter__(self):
        try:
            num_items = 0
            while True:
                data_iter = iter(self._data_loader)
                while True:
                    if self._num_batches is not None and num_items >= self._num_batches:
                        return
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break  # We've exhausted the dataset. Create a new iterator and start over.
                    num_items += 1
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
        finally:
            self.close()


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    filter_items = [x for x in items if x is not None]
    # return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *filter_items)

    def debug_stack(*args):
        arrays = [np.asarray(x) for x in args]
        try:
            return np.stack(arrays, axis=0)
        except ValueError as e:
            shapes = [x.shape for x in arrays]
            unique_shapes = set(shapes)
            print("\n======== DEBUG ERROR ========")
            print("Stacking failed!")
            print(f"Found varying shapes: {unique_shapes}")
            print(f"First 5 shapes: {shapes[:5]}")
            print(f"Sample data (first item): {arrays[0]}")
            print("=============================\n")
            raise e

    return jax.tree.map(debug_stack, *filter_items)


def _worker_init_fn(worker_id: int) -> None:
    """Configure JAX and seed Python/NumPy RNGs inside each worker process."""
    # Workers perform data IO/transforms on CPU; accelerators belong to the trainer.
    jax.config.update("jax_platforms", "cpu")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # PyTorch derives `torch.initial_seed()` per-worker from the seeded generator
    # passed to DataLoader, so this gives us a deterministic, worker-distinct seed
    # whenever the user sets `config.seed`. Mask to 32 bits because numpy requires it.
    base_seed = torch.initial_seed() & 0xFFFF_FFFF
    random.seed(base_seed)
    np.random.seed(base_seed)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def close(self):
        self._data_loader.close()

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
