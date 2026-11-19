"""Portable manifests for WBC, dexterous, fine manipulation and multi-embodiment simulation."""

import argparse
import dataclasses
import json
from pathlib import Path

from openpi.four_tracks.training import dataset_v21

GROUP_TRACK = {
    "g2_ctek90d_wbc": "wbc",
    "g2_wuji": "dexterous",
    "g2_ctek90d_manip": "fine_manipulation",
    "g1op": "multi_embodiment",
    "g2op": "multi_embodiment",
    "arx": "multi_embodiment",
    "agilex": "multi_embodiment",
}


def resolve_path(root, value):
    # Manifest paths are user-supplied locations, relative to the manifest directory.
    path = Path(value).expanduser()
    return str((root / path).resolve()) if not path.is_absolute() else str(path.resolve())


def read_manifest(path, *, validate_datasets=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    datasets = manifest.get("datasets", [])
    if not datasets:
        raise ValueError("manifest.datasets must contain at least one LeRobot dataset")
    if manifest.get("sampler", ""):
        raise ValueError("The v2.1 four-track profile samples all frames; omit sampler or set it to an empty string")
    for entry in datasets:
        group = entry["norm_group"]
        if group not in GROUP_TRACK:
            raise ValueError(f"unknown norm group {group!r}")
        if entry.get("track", GROUP_TRACK[group]) != GROUP_TRACK[group]:
            raise ValueError(f"track does not match norm group {group!r}")
        entry["path"] = resolve_path(path.parent, entry["path"])
        if validate_datasets and not (Path(entry["path"]) / "meta/info.json").is_file():
            raise FileNotFoundError(f"dataset missing meta/info.json: {entry['path']}")
    if validate_datasets:
        for entry in datasets:
            info = dataset_v21.require_v21(entry["path"])
            if float(info.get("fps", 0)) != 30.0:
                raise ValueError(f"{entry['path']}: this model profile requires 30Hz data")
            field_map = entry.get("field_map", {})
            if not isinstance(field_map, dict) or set(field_map) - {"observation.state", "action"}:
                raise ValueError("field_map must contain only observation.state and action mappings")
            group = entry["norm_group"]
            for feature, prefix in (("observation.state", "state"), ("action", "action")):
                fields = dataset_v21.feature_fields(info, feature, field_map)
                joints = fields.get(f"{prefix}/joint/position", {}).get("indices", [])
                embedded = group == "agilex" and len(joints) == 14
                if len(joints) not in ((12,) if group == "arx" else (12, 14) if group == "agilex" else (14,)):
                    raise ValueError(f"{entry['path']}: invalid arm joint layout for {group}")
                for side in ("left", "right"):
                    hand = fields.get(f"{prefix}/{side}_effector/position", {}).get("indices", [])
                    if len(hand) != (20 if group == "g2_wuji" else 1) and not embedded:
                        raise ValueError(f"{entry['path']}: missing or incompatible {side} effector field")
                if group == "g2_ctek90d_wbc":
                    waist = fields.get(f"{prefix}/waist/position", {}).get("indices", [])
                    if len(waist) != 5:
                        raise ValueError(f"{entry['path']}: WBC requires five waist joints")
                    if feature == "action" and len(fields.get("action/robot/velocity", {}).get("indices", [])) not in (
                        3,
                        6,
                    ):
                        raise ValueError(f"{entry['path']}: WBC requires planar or 6D robot velocity commands")
    manifest["norm_stats"] = {
        key: resolve_path(path.parent, value) for key, value in manifest.get("norm_stats", {}).items()
    }
    if manifest.get("tokenizer_path"):
        manifest["tokenizer_path"] = resolve_path(path.parent, manifest["tokenizer_path"])
    documents = manifest.get("license_files", [])
    if not isinstance(documents, list) or any(not isinstance(p, str) for p in documents):
        raise ValueError("license_files must be a list of local document paths")
    manifest["license_files"] = [resolve_path(path.parent, p) for p in documents]
    return manifest


def config_from_manifest(path, *, require_stats=True, validate_datasets=True):
    from openpi.four_tracks import checkpoint_bundle
    from openpi.four_tracks.training.config import DataConfig
    from openpi.four_tracks.training.config import LerobotUnifiedDataConfig
    from openpi.four_tracks.training.config import TrainConfig
    from openpi.models.pi0 import Pi0Config
    from openpi.training.optimizer import AdamW
    from openpi.training.optimizer import CosineDecaySchedule

    manifest = read_manifest(path, validate_datasets=validate_datasets)
    groups = [entry["norm_group"] for entry in manifest["datasets"]]
    stats = manifest["norm_stats"]
    if require_stats:
        if set(groups) - set(stats):
            raise ValueError(f"norm_stats missing groups: {sorted(set(groups) - set(stats))}")
        for group in set(groups):
            checkpoint_bundle.read_stats(stats[group], action_dim=64)
    data = LerobotUnifiedDataConfig(
        repo_id=[entry["path"] for entry in manifest["datasets"]],
        dataset_field_maps={
            entry["path"]: entry["field_map"] for entry in manifest["datasets"] if "field_map" in entry
        },
        norm_group_ids=groups,
        per_robot_norm_stats={g: stats[g] for g in set(groups)} if require_stats else None,
        tokenizer_path=manifest.get("tokenizer_path"),
        mask_base_state=True,
        mask_vy=False,
        prepend_robot_type=True,
        prepend_robot_type_coarse=True,
        prepend_robot_type_from_norm_group=True,
        waist_fill_only_for=("g2_ctek90d_wbc",),
        base_fill_only_for=("g2_ctek90d_wbc",),
        base_config=DataConfig(
            prompt_from_task=True,
            dataloader_sampler=manifest.get("sampler", ""),
        ),
    )
    return TrainConfig(
        name="four_tracks",
        project_name="GenieSim-four-tracks",
        exp_name="run",
        model=Pi0Config(pi05=True, action_dim=64, action_horizon=50, max_token_len=350),
        data=data,
        lr_schedule=CosineDecaySchedule(warmup_steps=10000, peak_lr=5e-5, decay_steps=1000000, decay_lr=5e-5),
        optimizer=AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        batch_size=32,
        num_workers=4,
        wandb_enabled=False,
        license_files=tuple(manifest["license_files"]),
    )


def training_cli():
    from openpi.training.weight_loaders import CheckpointWeightLoader

    parser = argparse.ArgumentParser(description="Train the four simulation tracks from a portable dataset manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-params", required=True, help="pretrained pi05 params directory")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-train-steps", type=int, default=450000)
    parser.add_argument("--save-interval", type=int, default=25000)
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = config_from_manifest(args.manifest)
    if args.batch_size <= 0 or args.num_train_steps <= 0 or args.save_interval <= 0:
        parser.error("batch size, training steps and save interval must be positive")
    return dataclasses.replace(
        cfg,
        exp_name=args.exp_name,
        weight_loader=CheckpointWeightLoader(args.base_params),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        checkpoint_base_dir=args.checkpoint_base_dir,
        fsdp_devices=args.fsdp_devices,
        resume=args.resume,
    )
