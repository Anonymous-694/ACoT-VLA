"""Validation and explicit field mappings for native LeRobot v2.1 datasets."""

import json
from pathlib import Path


def read_info(root):
    return json.loads((Path(root) / "meta/info.json").read_text())


def require_v21(root):
    """Validate the on-disk format before the native loader can download or convert it."""
    root = Path(root)
    info = read_info(root)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{root}: expected LeRobot codebase_version='v2.1', got {info.get('codebase_version')!r}")
    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(f"{root}: missing v2.1 metadata file meta/{name}")
    return info


def feature_fields(info, feature_key, field_map=None):
    """Resolve vector slices from a manifest, or optional public feature metadata.

    Standard LeRobot vectors do not prescribe robot joint semantics. A manifest
    field_map supplies those semantics without modifying the dataset's metadata.
    """
    feature = info["features"][feature_key]
    shape = feature.get("shape", [])
    if len(shape) != 1:
        raise ValueError(f"{feature_key}: expected a vector feature, got shape={shape}")
    field_map = field_map or {}
    fields = field_map.get(feature_key, feature.get("field_descriptions", {}))
    if not isinstance(fields, dict):
        raise ValueError(f"{feature_key}: field_map must map semantic field names to indices")
    result = {}
    for key, value in fields.items():
        entry = {"indices": value} if isinstance(value, list) else value
        if not isinstance(entry, dict) or not isinstance(entry.get("indices"), list):
            raise ValueError(f"{feature_key}/{key}: expected a list of indices")
        indices = entry["indices"]
        if any(type(i) is not int or i < 0 or i >= shape[0] for i in indices) or len(indices) != len(set(indices)):
            raise ValueError(f"{feature_key}/{key}: duplicate or out-of-range indices for width {shape[0]}")
        result[key] = entry
    return result
