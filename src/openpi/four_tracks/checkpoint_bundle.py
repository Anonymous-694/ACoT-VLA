"""Self-contained inference assets for the four simulation tracks.

Only parameters plus assets are needed at inference. Training datasets, experiment
paths, optimizer state and hardware-control code are never exported.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.shared import normalize

FORMAT_VERSION = 1
GROUPS = {"g2_ctek90d_wbc", "g2_ctek90d_manip", "g2_wuji", "g1op", "g2op", "arx", "agilex"}


def _copy_legal_text(source, destination):
    """Copy an explicitly selected UTF-8 legal document without overwriting another."""
    source, destination = Path(source), Path(destination)
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("license documents must be at most 1 MiB")
    content = source.read_bytes()
    content.decode("utf-8")
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"conflicting license document: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _attach_license_files(assets, spec, files):
    """Include supplied legal documents in the portable, checksummed asset manifest."""
    documents = set(spec.get("license_files", []))
    for source in files:
        source = Path(source)
        relative = f"licenses/{source.name}"
        _copy_legal_text(source, assets / relative)
        documents.add(relative)
        spec["sha256"][relative] = hashlib.sha256((assets / relative).read_bytes()).hexdigest()
    if documents:
        spec["license_files"] = sorted(documents)


def _preserve_legal_notices(source, destination):
    """Preserve standard legal attachments, excluding unrelated experiment files."""
    prefixes = ("LICENSE", "LICENCE", "NOTICE", "COPYING", "THIRD_PARTY_NOTICES", "THIRD_PARTY_LICENSES")
    for relative in (Path(), Path("assets")):
        directory = source / relative
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            name = path.name.upper()
            legal = any(
                name == prefix or name.startswith(tuple(prefix + sep for sep in (".", "-", "_"))) for prefix in prefixes
            )
            if legal and path.is_file() and path.suffix.lower() in ("", ".txt", ".md", ".rst"):
                # Resolve before reading so an attachment cannot point outside its source.
                _copy_legal_text(_resource(source, str(relative / path.name)), destination / relative / path.name)


def read_stats(path, *, action_dim=64):
    stats = normalize.deserialize_json(Path(path).read_text())
    if set(stats) != {"state", "actions"}:
        raise ValueError(f"{path}: expected state and actions statistics")
    for key, values in stats.items():
        arrays = {}
        for field in ("mean", "std", "q01", "q99"):
            arr = np.asarray(getattr(values, field), dtype=np.float64)
            if arr.shape != (action_dim,) or not np.isfinite(arr).all():
                raise ValueError(f"{path}: {key}.{field} must be finite shape ({action_dim},)")
            arrays[field] = arr
        if np.any(arrays["std"] < 0) or np.any(arrays["q01"] > arrays["q99"]):
            raise ValueError(f"{path}: invalid std or quantile ordering")
    return stats


def _resource(root, value):
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"bundle resource must stay inside assets: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"missing bundle resource: {value}")
    return path


def write_assets(directory, config, data_config):
    """Write the exact training normalization maps and inference configuration."""
    from openpi.four_tracks.serving import task_routing
    from openpi.four_tracks.shared import robot_config
    from openpi.shared import download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stats_map = data_config.per_robot_norm_stats
    if not stats_map or not set(stats_map) <= GROUPS:
        raise ValueError("four-track checkpoints require explicit supported per-group normalization statistics")
    paths = {}
    for group, stats in stats_map.items():
        relative = f"norm_stats/{group}/norm_stats.json"
        normalize.save((directory / relative).parent, stats)
        read_stats(directory / relative, action_dim=config.model.action_dim)
        paths[group] = relative
    table = task_routing.load()
    routes = {}
    for task, branch in table.items():
        if isinstance(branch, dict):
            branch = {robot: group for robot, group in branch.items() if group in stats_map}  # noqa: PLW2901
        if (isinstance(branch, str) and branch in stats_map) or (isinstance(branch, dict) and branch):
            routes[task] = branch
    task_routing.validate(routes, stats_map)
    if task_routing.groups(routes) != set(stats_map):
        raise ValueError("task routes do not cover all trained normalization groups")
    (directory / "task_routes.json").write_text(json.dumps(routes, indent=2))
    registry = robot_config.all_configs()
    (directory / "robot_config.json").write_text(json.dumps(registry, indent=2))
    tokenizer = getattr(config.data, "tokenizer_path", None)
    tokenizer = (
        Path(tokenizer)
        if tokenizer
        else download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    )
    shutil.copyfile(tokenizer, directory / "tokenizer.model")
    spec = {
        "format_version": FORMAT_VERSION,
        "model": dataclasses.asdict(config.model),
        "layout": "unified62",
        "fps": 30.0,
        "norm_stats": paths,
        "task_routes": "task_routes.json",
        "robot_config": "robot_config.json",
        "tokenizer": "tokenizer.model",
        "waist_groups": list(getattr(config.data, "waist_fill_only_for", ()) or ()),
        "data": {
            "use_delta_joint_actions": config.data.use_delta_joint_actions,
            "mask_base_state": config.data.mask_base_state,
            "mask_vy": config.data.mask_vy,
            "prepend_robot_type": config.data.prepend_robot_type,
            "prepend_robot_type_from_norm_group": config.data.prepend_robot_type_from_norm_group,
        },
    }
    resources = [*paths.values(), spec["task_routes"], spec["robot_config"], spec["tokenizer"]]
    spec["sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in resources}
    _attach_license_files(directory, spec, getattr(config, "license_files", ()))
    (directory / "inference.json").write_text(json.dumps(spec, indent=2))


def inspect(checkpoint_dir):
    """Validate all assets before allocating the model or touching an accelerator."""
    from openpi.four_tracks.serving import task_routing

    root = Path(checkpoint_dir).resolve()
    if not (root / "params").is_dir():
        raise FileNotFoundError(f"checkpoint lacks params directory: {root}")
    assets = root / "assets"
    spec = json.loads((assets / "inference.json").read_text())
    if spec["format_version"] != FORMAT_VERSION or spec["layout"] != "unified62":
        raise ValueError("unsupported checkpoint inference format")
    if spec["model"].get("action_dim") != 64 or not spec["model"].get("pi05"):
        raise ValueError("this release serves pi05 with action_dim=64 and unified62 output")
    if spec["fps"] != 30.0:
        raise ValueError("unified62 serving currently requires 30Hz action timestamps")
    paths = spec["norm_stats"]
    if not paths or not set(paths) <= GROUPS:
        raise ValueError("checkpoint has empty or unsupported normalization groups")
    expected = set(paths.values()) | {spec["task_routes"], spec["robot_config"], spec["tokenizer"]}
    license_files = spec.get("license_files", [])
    if not isinstance(license_files, list) or any(not isinstance(p, str) for p in license_files):
        raise ValueError("license_files must be a list of asset-relative paths")
    if len(license_files) != len(set(license_files)) or any(
        not p.startswith("licenses/")
        or ".." in Path(p).parts
        or Path(p).suffix.lower() not in ("", ".txt", ".md", ".rst")
        for p in license_files
    ):
        raise ValueError("license_files must name distinct text documents inside licenses/")
    expected.update(license_files)
    if set(spec["sha256"]) != expected:
        raise ValueError("resource hashes must cover exactly the inference resources")
    for name, digest in spec["sha256"].items():
        path = _resource(assets, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
    for name in paths.values():
        read_stats(_resource(assets, name))
    routes = task_routing.load(_resource(assets, spec["task_routes"]))
    task_routing.validate(routes, paths)
    if task_routing.groups(routes) != set(paths):
        raise ValueError("checkpoint task routes do not cover normalization groups")
    if not set(spec["waist_groups"]) <= {"g2_ctek90d_wbc"}:
        raise ValueError("only the WBC group may populate waist/chassis blocks")
    from openpi.four_tracks.shared import robot_config

    registry = robot_config.validate_registry(json.loads(_resource(assets, spec["robot_config"]).read_text()))
    registry_groups = set()
    for entry in registry.values():
        if entry.get("norm_group"):
            registry_groups.add(entry["norm_group"])
        registry_groups.update((entry.get("scenarios") or {}).values())
    if set(paths) - registry_groups:
        raise ValueError("robot registry does not cover every normalization group")
    if spec["model"]["action_horizon"] <= 0 or spec["model"]["max_token_len"] <= 0:
        raise ValueError("invalid model horizon or token length")
    return spec


def load_config(checkpoint_dir):
    from openpi.four_tracks.training.config import LerobotUnifiedDataConfig
    from openpi.four_tracks.training.config import TrainConfig
    from openpi.models.pi0 import Pi0Config

    spec = inspect(checkpoint_dir)
    assets = Path(checkpoint_dir).resolve() / "assets"
    factory = LerobotUnifiedDataConfig(
        repo_id=None,
        norm_group_ids=tuple(spec["norm_stats"]),
        per_robot_norm_stats={g: str(_resource(assets, p)) for g, p in spec["norm_stats"].items()},
        tokenizer_path=str(_resource(assets, spec["tokenizer"])),
        waist_fill_only_for=tuple(spec["waist_groups"]),
        base_fill_only_for=tuple(spec["waist_groups"]),
        **spec["data"],
    )
    return TrainConfig(
        name="four_tracks",
        model=Pi0Config(**spec["model"]),
        data=factory,
        wandb_enabled=False,
        license_files=tuple(str(_resource(assets, p)) for p in spec.get("license_files", [])),
    )


def export(source, destination, *, config=None, data_config=None, license_files=()):
    """Export params and portable assets. Destination must not already exist."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source):
        raise ValueError("export destination must not be inside the source checkpoint")
    destination.mkdir(parents=True)
    try:
        shutil.copytree(source / "params", destination / "params", symlinks=False)
        if config is None:
            spec = inspect(source)
            target = destination / "assets"
            target.mkdir()
            for relative in ["inference.json", *spec["sha256"]]:
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_resource(source / "assets", relative), path)
        else:
            write_assets(destination / "assets", config, data_config)
        _preserve_legal_notices(source, destination)
        if license_files:
            assets = destination / "assets"
            spec = json.loads((assets / "inference.json").read_text())
            _attach_license_files(assets, spec, license_files)
            (assets / "inference.json").write_text(json.dumps(spec, indent=2))
        inspect(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
