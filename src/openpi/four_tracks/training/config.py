# Derived from ACoT-VLA/openpi: src/openpi/training/config.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for the four simulation tracks; no experiment or hardware-control configs."""

import abc
from collections.abc import Sequence
import dataclasses
import logging
import os
import pathlib
from typing import Any, Protocol, TypeAlias

import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.four_tracks.models.tokenizer as _tokenizer
import openpi.four_tracks.policies.go1_policy as go1_policy
import openpi.four_tracks.transforms as _transforms
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class CanonicalGroup:
    """One contiguous block of the unified cross-embodiment layout.

    A dataset's raw feature array is sliced by ``state_keys`` / ``action_keys`` (resolved
    from a manifest field_map or the dataset's own field_descriptions), optionally restricted
    to one side of a paired field (``split``), optionally sub-selected (``*_select``), then
    zero-padded (or truncated) to ``size``. Groups are concatenated in declaration order, so
    every embodiment lands in the same layout regardless of its own DOF or which fields it
    happens to ship.

    ``optional=False`` means the group MUST be present in every dataset; ``optional=True``
    (the default) lets a dataset omit it, in which case the whole block is zeros.
    """

    name: str
    # Raw fields to gather. Empty state_keys means "this block is always zero on the state
    # side" (e.g. base velocity, which has no observation counterpart in the dataset).
    state_keys: Sequence[str] = ()
    action_keys: Sequence[str] = ()
    # Canonical width of this block.
    size: int = 0
    # Delta-encode (actions only): action -= current state of the same block.
    delta: bool = False
    # ``"left"`` / ``"right"``: take the first / second half of the gathered indices BEFORE
    # padding. Needed for paired joint fields whose raw width varies (12 for a 6+6 robot,
    # 14 for a 7+7 one) so the halves pad independently instead of bleeding into each other.
    split: str | None = None
    # Sub-select within the gathered indices (e.g. pick vx, vy, yaw_rate out of a 6-dim
    # velocity field). Applied after ``split``.
    state_select: Sequence[int] | None = None
    action_select: Sequence[int] | None = None
    # If True the whole block is dropped from the model input (zeroed after canonicalization).
    mask_state: bool = False
    mask_action: bool = False
    # Intra-block dims to drop, relative to the block start (e.g. (1,) zeroes vy only).
    mask_state_dims: Sequence[int] = ()
    mask_action_dims: Sequence[int] = ()
    # Name of a data key produced by an earlier transform that supplies this block's STATE
    # wholesale (padded/truncated to ``size``) instead of gathering ``state_keys`` from the
    # raw feature array. Used for computed quantities with no dataset field, e.g. base
    # velocity derived from consecutive poses.
    state_source: str | None = None
    # If set, only datasets whose norm group id is in this list populate the block; every
    # other dataset gets the block zero-filled at full width. Used to keep waist data out of
    # embodiments/tasks where the waist barely moves (std ~1e-3) and would only add noise.
    fill_only_for: Sequence[str] | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    dataset_field_maps: dict[str, dict] = dataclasses.field(default_factory=dict)
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Per-robot norm_stats for training and inference. Maps robot_id to loaded NormStats.
    # When set, PerRobotNormalize is used instead of global Normalize.
    per_robot_norm_stats: dict[str, _transforms.NormStats] | None = None

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If True, prepend "current robot is {robot_type}, " to the prompt. At training
    # time robot_type is read from each LeRobot dataset's meta/info.json["robot_type"]
    # (e.g. "g2a_sim" / "a2d_sim"); at inference it is derived from the request payload.
    prepend_robot_type: bool = False
    # If True, the prefix uses only the coarse info.json robot_type (what the inference server
    # can supply) without relying on dataset-specific embodiment metadata.
    prepend_robot_type_coarse: bool = False
    # If True, the prefix is the per-dataset norm_group_ids key (needs norm_group_ids), so
    # families that share a coarse robot_type (WBC / fine manip / wuji / g2op) get distinct prefixes.
    prepend_robot_type_from_norm_group: bool = False

    # Cross-embodiment field canonicalization. When True, create_torch_dataset reorders each
    # dataset's raw state/action into a uniform [arm14, grippers, waist5] layout using that
    # dataset's own info.json field indices (see RemapGo1FieldsToCanonical). Required when a
    # run mixes embodiments with different raw layouts / waist widths (e.g. g2a_sim + a2d_sim),
    # otherwise a single global index set silently mislabels the smaller-dim embodiment.
    canonicalize_go1_fields: bool = False
    canonical_state_keys: Sequence[str] | None = None
    canonical_action_keys: Sequence[str] | None = None
    canonical_dim: int = 21

    # Variable-arm-DOF cross-embodiment alignment. When True, create_torch_dataset splits
    # each dataset's joint field into left/right arms (dim//2), pads each arm to
    # arm_canonical_per_arm_dim, and appends grippers — so robots with different per-arm
    # DOF (e.g. 6+6 vs 7+7) land in one uniform layout.
    canonicalize_arm_groups: bool = False
    arm_joint_state_key: str = "state/joint/position"
    arm_joint_action_key: str = "action/joint/position"
    arm_grip_state_keys: Sequence[str] = ("state/left_effector/position", "state/right_effector/position")
    arm_grip_action_keys: Sequence[str] = ("action/left_effector/position", "action/right_effector/position")
    arm_eef_state_key: str | None = None
    arm_eef_action_key: str | None = None
    arm_canonical_per_arm_dim: int | None = None

    # Unified cross-embodiment layout (supersedes canonicalize_go1_fields /
    # canonicalize_arm_groups when set). create_torch_dataset resolves each group's indices
    # from each dataset's own info.json, so robots with different DOF, different waist
    # widths, a 1-dim gripper vs a 20-DoF dexterous hand, or no chassis at all all land in
    # one layout. Optional groups a dataset lacks are zero-filled rather than raising.
    canonical_groups: Sequence[CanonicalGroup] | None = None
    # (position_indices, orientation_indices) into the RAW observation.state, used to derive
    # body-frame base velocity from consecutive frames. Must be applied before the canonical
    # remap (raw indices disappear afterwards), so create_torch_dataset runs it at the dataset
    # level alongside the remap rather than in repack_transforms. Requires include_prev_state.
    canonical_base_velocity: tuple[Sequence[int], Sequence[int]] | None = None

    # Explicit per-dataset override of the per_robot_norm_stats key, parallel to repo_id.
    # Default (None) resolves each dataset to the public robot_type, which is
    # wrong when one physical robot appears in two data families that populate DIFFERENT
    # canonical blocks -- e.g. g2_ctek90d is used both for WBC tasks (waist +
    # chassis populated) and for fine manipulation (those blocks are structural zeros).
    # Sharing one key there folds the fine-manipulation zeros into the WBC statistics and
    # shrinks/zeroes the chassis+waist distribution. Give the families distinct keys and
    # they get distinct stats files.
    norm_group_ids: Sequence[str] | None = None

    # Force GripperToOpenClosed even when only one robot_type is present.
    # Needed for per-robot norm_stats configs so norm_stats sees the same
    # gripper mapping as mixed training.
    force_gripper_norm: bool = False

    # Hard-off for cross-embodiment gripper unification (open=0/closed=1). Overrides both
    # force_gripper_norm and the auto-trigger on mixed robot_types. Unified configs use one
    # norm_stats set per embodiment, so each gripper keeps its raw unit; the robot_type lookup
    # table is unreliable here (wuji span 0, arx vs arx_sim, a2d_sim 0..120).
    disable_gripper_unify: bool = False

    # If True, include previous frame's observation.state via delta_timestamps
    # so transforms can compute frame-to-frame differences (e.g. base velocity).
    include_prev_state: bool = False

    dataloader_sampler: str | None = ""

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    tokenizer_path: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len, self.tokenizer_path),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str | list[str] | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    state_mask = None
    action_mask = None

    @abc.abstractmethod
    def create(self, assets_dirs, model_config) -> DataConfig:
        raise NotImplementedError

    def create_base_config(self, assets_dirs, model_config) -> DataConfig:
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=self.repo_id,
            asset_id=self.assets.asset_id,
            use_quantile_norm=True,
        )

    def _pad_norm_stats_to_canonical(self, norm_stats, robot_id):
        # Public manifests require statistics already expressed in the 64-dim model layout.
        return norm_stats

    def _apply_norm_stats_mask(self, norm_stats: dict[str, _transforms.NormStats]) -> None:
        if self.state_mask is not None and "state" in norm_stats:
            state_mask = np.asarray(self.state_mask)
            state_len = norm_stats["state"].std.shape[0]
            dims = min(state_mask.shape[-1], state_len)
            if dims > 0:
                m = state_mask[:dims]
                norm_stats["state"].std[m] = 1e6
                norm_stats["state"].mean[m] = 0
                norm_stats["state"].q01[m] = 0
                norm_stats["state"].q99[m] = 0

        if self.action_mask is not None and "actions" in norm_stats:
            action_mask = np.asarray(self.action_mask)
            action_len = norm_stats["actions"].std.shape[0]
            dims = min(action_mask.shape[-1], action_len)
            if dims > 0:
                m = action_mask[:dims]
                norm_stats["actions"].std[m] = 1e6
                norm_stats["actions"].mean[m] = 0
                norm_stats["actions"].q01[m] = 0
                norm_stats["actions"].q99[m] = 0


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LerobotUnifiedDataConfig(DataConfigFactory):
    """One dataloader for every embodiment class in the repo: WBC / multi-arm / dexterous hand.

    Unified canonical layout (62 dims)::

        [0:14]   arm        left 7 | right 7          delta
        [14:19]  waist      5                         delta
        [19:22]  base       vx, vy, yaw_rate          absolute
        [22:42]  hand_left  left effector             absolute
        [42:62]  hand_right right effector            absolute

    The layout is a *union*: each dataset contributes only the blocks it actually ships, and
    every other block is zero-filled at its canonical width. That is what lets these coexist
    in one run without any per-robot dimension bookkeeping:

    ===================  =========================  ============================
    embodiment           source fields              blocks actually populated
    ===================  =========================  ============================
    WBC (g2a_sim)        joint/waist/velocity       arm, waist, base, hand[0:1]
    multi-arm (arx etc)  joint/effector             arm, hand[0:1]
    dexterous (wuji)     joint/effector(20 each)    arm, hand[0:20]
    ===================  =========================  ============================

    A 1-DoF gripper robot's effector naturally lands in slot 0 of its 20-wide hand block and
    the remaining 19 slots stay zero -- no special-casing, because the block pads whatever
    width the dataset supplies.

    Blocks are resolved per dataset from that dataset's own ``meta/info.json``
    field_descriptions, or explicit manifest field_map entries. Different raw vector
    widths map onto the same 62 dimensions.
    """

    tokenizer_path: str | None = None
    dataset_field_maps: dict[str, dict] = dataclasses.field(default_factory=dict)
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    # Widths. arm_canonical_per_arm_dim must be >= every robot's per-arm DOF (max(6,7) = 7).
    arm_canonical_per_arm_dim: int = 7
    waist_dim: int = 5
    base_dim: int = 3
    hand_dim: int = 20

    # Raw field names. Override only if a dataset names them differently.
    joint_state_key: str = "state/joint/position"
    joint_action_key: str = "action/joint/position"
    waist_state_key: str = "state/waist/position"
    waist_action_key: str = "action/waist/position"
    left_hand_state_key: str = "state/left_effector/position"
    left_hand_action_key: str = "action/left_effector/position"
    right_hand_state_key: str = "state/right_effector/position"
    right_hand_action_key: str = "action/right_effector/position"
    # Base action is a (vx, vy, yaw_rate, ...) velocity field; state has no equivalent, so the
    # state-side block is computed from consecutive poses by ComputeBaseVelocity.
    base_action_key: str = "action/robot/velocity"
    base_action_select: Sequence[int] = (0, 1, 5)
    base_position_indices: Sequence[int] | None = None
    base_orientation_indices: Sequence[int] | None = None

    # Zero-out switches for blocks present in the data but not to be modelled.
    # mask_base_state drops the chassis from the STATE only, leaving the action supervised:
    # the state-side velocity is a two-frame finite difference at training time but is
    # identically zero at inference (single frame), so feeding it would be a train/serve
    # mismatch. The action side carries the real chassis command and stays live.
    mask_base_state: bool = True
    # Drops vy on BOTH sides, for datasets where lateral velocity is constant/degenerate and
    # quantile normalization would blow it up (q01 == q99 -> divide by ~1e-6).
    mask_vy: bool = False

    prepend_robot_type: bool = True
    # Use only the coarse info.json robot_type in the prompt prefix (see DataConfig).
    prepend_robot_type_coarse: bool = False
    prepend_robot_type_from_norm_group: bool = False
    per_robot_norm_stats: dict[str, str] | None = None
    # Explicit per-dataset stats key, parallel to repo_id. Needed when one physical robot
    # appears in two data families with DIFFERENT populated blocks (e.g. the same G2 robot
    # ctek90d body doing WBC vs fine manipulation): robot_type cannot tell them
    # apart, so they would otherwise share one stats file and corrupt each other's
    # waist/chassis statistics with structural zeros. See DataConfig.norm_group_ids.
    norm_group_ids: Sequence[str] | None = None
    # Only datasets whose norm group id is listed here populate the waist block; all others
    # get it zero-filled at full width. None = every dataset that has a waist field fills it.
    # Requires norm_group_ids (the id is what tells WBC apart from fine-manipulation data on
    # the same body).
    waist_fill_only_for: Sequence[str] | None = None
    # Same restriction for the chassis block. Without it, a robot whose
    # action/robot/velocity means something else (ARX: 4 wheel speeds) would have those
    # values read as vx/vy.
    base_fill_only_for: Sequence[str] | None = None
    output_dim: int | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    },
                    # Dex datasets ship no per-frame prompt; they rely on default_prompt.
                    optional=("prompt",),
                )
            ]
        )
    )

    action_sequence_keys: Sequence[str] = ("action",)

    @property
    def base_velocity_enabled(self) -> bool:
        # The state-side base block is computed, so a previous frame is required to derive it.
        return self.base_dim > 0 and not self.mask_base_state

    def canonical_groups(self) -> tuple[CanonicalGroup, ...]:
        """The unified layout as an ordered list of blocks."""
        arm = self.arm_canonical_per_arm_dim
        base_select = tuple(self.base_action_select)[: self.base_dim]
        base_mask_dims = (1,) if (self.mask_vy and self.base_dim > 1) else ()

        return (
            CanonicalGroup(
                name="arm_left",
                state_keys=(self.joint_state_key,),
                action_keys=(self.joint_action_key,),
                size=arm,
                delta=True,
                split="left",
            ),
            CanonicalGroup(
                name="arm_right",
                state_keys=(self.joint_state_key,),
                action_keys=(self.joint_action_key,),
                size=arm,
                delta=True,
                split="right",
            ),
            CanonicalGroup(
                name="waist",
                state_keys=(self.waist_state_key,),
                action_keys=(self.waist_action_key,),
                size=self.waist_dim,
                delta=True,
                fill_only_for=self.waist_fill_only_for,
            ),
            CanonicalGroup(
                name="base",
                # State side has no dataset field: supplied by ComputeBaseVelocity.
                state_keys=(),
                state_source="base_velocity" if self.base_velocity_enabled else None,
                action_keys=(self.base_action_key,),
                action_select=base_select,
                size=self.base_dim,
                mask_state=self.mask_base_state,
                mask_action=False,
                mask_state_dims=base_mask_dims,
                mask_action_dims=base_mask_dims,
                fill_only_for=self.base_fill_only_for,
            ),
            CanonicalGroup(
                name="hand_left",
                state_keys=(self.left_hand_state_key,),
                action_keys=(self.left_hand_action_key,),
                size=self.hand_dim,
            ),
            CanonicalGroup(
                name="hand_right",
                state_keys=(self.right_hand_state_key,),
                action_keys=(self.right_hand_action_key,),
                size=self.hand_dim,
            ),
        )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        canonical_dim = self.arm_canonical_per_arm_dim * 2 + self.waist_dim + self.base_dim + 2 * self.hand_dim
        out_dim = self.output_dim or canonical_dim
        action_dim = model_config.action_dim
        assert action_dim >= canonical_dim, (
            f"model action_dim ({action_dim}) must be >= the unified canonical dim ({canonical_dim}). "
            f"Set Pi0Config(action_dim={canonical_dim})."
        )

        groups = self.canonical_groups()
        state_mask, action_mask = _groups_to_masks(groups, action_dim)

        object.__setattr__(self, "state_mask", state_mask)
        object.__setattr__(self, "action_mask", action_mask)

        data_transforms = _transforms.Group(
            inputs=[
                go1_policy.Go1Inputs(
                    action_dim=action_dim,
                    model_type=model_config.model_type,
                    state_mask=state_mask,
                    action_mask=action_mask,
                    state_indices=None,
                    action_indices=None,
                )
            ],
            outputs=[go1_policy.Go1Outputs(output_dim=out_dim)],
        )

        if self.use_delta_joint_actions:
            delta_action_mask = _groups_delta_mask(groups)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt, tokenizer_path=self.tokenizer_path
        )(model_config)

        loaded_per_robot_ns = None
        if self.per_robot_norm_stats:
            from openpi.shared.normalize import deserialize_json

            loaded_per_robot_ns = {}
            for robot_id, ns_path in self.per_robot_norm_stats.items():
                ns = deserialize_json(pathlib.Path(ns_path).read_text())
                self._apply_norm_stats_mask(ns)
                loaded_per_robot_ns[robot_id] = ns
                logging.info(f"Loaded per-robot norm_stats for {robot_id} from {ns_path}")

        if self.base_velocity_enabled and (self.base_position_indices is None or self.base_orientation_indices is None):
            raise ValueError("Computing base state requires explicit position/orientation field indices")

        # Base velocity is derived from consecutive RAW frames, so it has to run before the
        # canonical remap strips the raw indices. create_torch_dataset does that at the dataset
        # level (see DataConfig.canonical_base_velocity); it is NOT a repack transform.
        base_velocity = (
            (tuple(self.base_position_indices), tuple(self.base_orientation_indices))
            if self.base_velocity_enabled
            else None
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            dataset_field_maps=self.dataset_field_maps,
            canonical_groups=groups,
            canonical_base_velocity=base_velocity,
            prepend_robot_type=self.prepend_robot_type,
            prepend_robot_type_coarse=self.prepend_robot_type_coarse,
            prepend_robot_type_from_norm_group=self.prepend_robot_type_from_norm_group,
            per_robot_norm_stats=loaded_per_robot_ns,
            include_prev_state=self.base_velocity_enabled,
            # Must be forwarded: the loader reads it off the DataConfig (not the factory) for
            # the prompt prefix, for fill_only_for block gating, and for the per-dataset
            # _robot_id that selects the norm stats. Leaving it None makes all three fall back.
            norm_group_ids=self.norm_group_ids,
            # Per-embodiment norm_stats absorb each gripper's own unit; never unify by robot_type.
            disable_gripper_unify=True,
        )


def _groups_to_masks(groups: Sequence[CanonicalGroup], action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim (state_mask, action_mask) over the model action_dim; padding is masked out."""
    state_mask = np.ones(action_dim, dtype=bool)
    action_mask = np.ones(action_dim, dtype=bool)
    cursor = 0
    for g in groups:
        n = min(g.size, action_dim - cursor)
        if n <= 0:
            break
        state_mask[cursor : cursor + n] = g.mask_state
        action_mask[cursor : cursor + n] = g.mask_action
        for d in g.mask_state_dims:
            if d < n:
                state_mask[cursor + d] = True
        for d in g.mask_action_dims:
            if d < n:
                action_mask[cursor + d] = True
        cursor += n
    return state_mask, action_mask


def _groups_delta_mask(groups: Sequence[CanonicalGroup]) -> tuple[bool, ...]:
    """Delta mask sized exactly to the canonical layout (not the padded model action_dim)."""
    mask: list[bool] = []
    for g in groups:
        mask.extend([g.delta] * g.size)
    return tuple(mask)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "GenieSim-four-tracks"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Explicit legal documents for model/tokenizer distribution; copied without source paths.
    license_files: tuple[str, ...] = ()

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        if not os.getenv("DEBUG_MODE", default=False) == "true":
            return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
        else:
            return (pathlib.Path(self.checkpoint_base_dir) / self.name / "debug").resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def cli() -> TrainConfig:
    from openpi.four_tracks.training.four_tracks import training_cli

    return training_cli()


def get_config(config_name: str) -> TrainConfig:
    if config_name == "debug":
        return TrainConfig(name="debug", exp_name="debug", data=FakeDataConfig(), wandb_enabled=False)
    raise ValueError(
        "Use a four-track dataset manifest with --manifest; historical experiment configs are not distributed"
    )
