"""Native format checks and semantic vector mappings without dataset mutation."""

import json

import pytest

from openpi.four_tracks.training import dataset_v21


def _metadata(root, version="v2.1"):
    meta = root / "meta"
    meta.mkdir()
    info = {"codebase_version": version, "features": {"action": {"shape": [6]}}}
    (meta / "info.json").write_text(json.dumps(info))
    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        (meta / name).write_text("{}\n")
    return info


def test_native_v21_validation_preserves_metadata(tmp_path):
    info = _metadata(tmp_path)
    before = {p.name: p.read_bytes() for p in (tmp_path / "meta").iterdir()}
    assert dataset_v21.require_v21(tmp_path) == info
    assert {p.name: p.read_bytes() for p in (tmp_path / "meta").iterdir()} == before


@pytest.mark.parametrize("version", ["v2.0", "v3.0", None])
def test_other_versions_fail_before_native_loader(tmp_path, version):
    _metadata(tmp_path, version)
    with pytest.raises(ValueError, match="expected LeRobot"):
        dataset_v21.require_v21(tmp_path)


@pytest.mark.parametrize("name", ["tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"])
def test_missing_standard_metadata_fails_explicitly(tmp_path, name):
    _metadata(tmp_path)
    (tmp_path / "meta" / name).unlink()
    with pytest.raises(FileNotFoundError, match=name):
        dataset_v21.require_v21(tmp_path)


def test_explicit_fields_override_optional_metadata(tmp_path):
    info = _metadata(tmp_path)
    info["features"]["action"]["field_descriptions"] = {"action/joint/position": {"indices": [0, 1]}}
    assert dataset_v21.feature_fields(info, "action")["action/joint/position"]["indices"] == [0, 1]
    explicit = {"action": {"action/joint/position": [3, 4]}}
    assert dataset_v21.feature_fields(info, "action", explicit)["action/joint/position"]["indices"] == [3, 4]
    assert info["features"]["action"]["field_descriptions"]["action/joint/position"]["indices"] == [0, 1]


@pytest.mark.parametrize("indices", [[-1], [6], [0, 0], [1.0], [True]])
def test_invalid_field_indices_are_rejected(tmp_path, indices):
    info = _metadata(tmp_path)
    with pytest.raises(ValueError, match="indices"):
        dataset_v21.feature_fields(info, "action", {"action": {"action/joint/position": indices}})


@pytest.mark.parametrize(("width", "expected"), [(3, (0, 1, 2)), (6, (0, 1, 5))])
def test_wbc_planar_and_spatial_velocity_fields(tmp_path, width, expected):
    from openpi.four_tracks.training.config import LerobotUnifiedDataConfig
    from openpi.four_tracks.training.data_loader import _resolve_group_indices

    info = _metadata(tmp_path)
    info["features"] = {"action": {"shape": [width]}, "observation.state": {"shape": [1]}}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    fields = {"action": {"action/robot/velocity": list(range(width))}}
    group = next(g for g in LerobotUnifiedDataConfig().canonical_groups() if g.name == "base")
    state, action = _resolve_group_indices(str(tmp_path), group, "g2_ctek90d_wbc", fields)
    assert state == ()
    assert action == expected
