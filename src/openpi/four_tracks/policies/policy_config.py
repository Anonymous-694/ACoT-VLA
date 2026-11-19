# Derived from ACoT-VLA/openpi: src/openpi/policies/policy_config.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.four_tracks.policies.policy as _policy
from openpi.four_tracks.training import checkpoints as _checkpoints
from openpi.four_tracks.training import config as _config
import openpi.four_tracks.transforms as transforms
import openpi.models.model as _model
import openpi.shared.download as download


def _load_per_robot_norm_stats(
    data_factory: _config.DataConfigFactory,
) -> dict[str, dict[str, transforms.NormStats]] | None:
    """Load and pad per-robot norm_stats from a LerobotVariableArmDataConfig."""
    per_robot = getattr(data_factory, "per_robot_norm_stats", None)
    if not per_robot:
        return None
    from openpi.shared.normalize import deserialize_json

    result = {}
    for robot_id, path in per_robot.items():
        ns = deserialize_json(pathlib.Path(path).read_text())
        ns = data_factory._pad_norm_stats_to_canonical(ns, robot_id)
        if type(data_factory).__name__ == "LerobotUnifiedDataConfig":
            # Training applies the block masks to every loaded stats set in create(); do the
            # same here so masked dims (chassis state, padding) match between train and serve.
            data_factory._apply_norm_stats_mask(ns)
        result[robot_id] = ns
    return result


def create_trained_policy(
    train_config: _config.TrainConfig | None,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    if (checkpoint_dir / "assets" / "inference.json").is_file():
        from openpi.four_tracks import checkpoint_bundle

        train_config = checkpoint_bundle.load_config(checkpoint_dir)
    if train_config is None:
        raise ValueError("checkpoint needs portable assets/inference.json; export the complete bundle first")

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    per_robot_ns = _load_per_robot_norm_stats(train_config.data)

    if norm_stats is None:
        if per_robot_ns:
            norm_stats = {}
        elif data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        else:
            logging.info("Loading norm stats: checkpoint_dir=%s, asset_id=%s", checkpoint_dir, data_config.asset_id)
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    if per_robot_ns:
        logging.info("Per-robot norm_stats loaded for: %s", sorted(per_robot_ns.keys()))
        normalize_in = transforms.PerRobotNormalize(
            norm_stats_map=per_robot_ns,
            default_norm_stats=norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        unnormalize_out = transforms.PerRobotUnnormalize(
            norm_stats_map=per_robot_ns,
            default_norm_stats=norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
    else:
        normalize_in = transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        unnormalize_out = transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)

    metadata = dict(train_config.policy_metadata or {})
    canonical = getattr(train_config.data, "arm_canonical_per_arm_dim", None)
    if canonical:
        metadata["arm_canonical_per_arm_dim"] = canonical
    if type(train_config.data).__name__ == "LerobotUnifiedDataConfig":
        # Tell Policy.post_process to leave the actions alone: this layout is 62-dim and
        # is split by serving/unified_layout, not by the canonical 30-dim rules.
        metadata["unified_layout"] = True
    if type(train_config.data).__name__ == "LerobotGo1WBCDataConfig":
        # WBC configs deliberately produce >16 action dims (waist + chassis via
        # Go1Outputs(output_dim=...)); Policy.post_process must not apply the legacy
        # 16-dim truncation meant for non-WBC deployments.
        metadata["wbc_layout"] = True

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            normalize_in,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            unnormalize_out,
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=metadata,
    )
