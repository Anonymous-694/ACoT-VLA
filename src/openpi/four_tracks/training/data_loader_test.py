# Derived from ACoT-VLA/openpi: src/openpi/training/data_loader_test.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.four_tracks import transforms as _transforms
from openpi.four_tracks.training import config as _config
from openpi.four_tracks.training import data_loader as _data_loader
from openpi.models import pi0


def _write_gripper_dataset(root, values, *, robot_type="agilex_sim"):
    (root / "meta").mkdir()
    fields = {}
    for feature, field in [("observation.state", "state/joint/position"), ("action", "action/joint/position")]:
        fields[feature] = {"shape": [len(values)], "field_descriptions": {field: {"indices": list(range(len(values)))}}}
    (root / "meta" / "info.json").write_text(json.dumps({"robot_type": robot_type, "features": fields}))
    (root / "data").mkdir()
    pq.write_table(pa.table({"action": [values.tolist()]}), root / "data" / "episode.parquet")


@pytest.mark.parametrize("converted", [False, True])
def test_embedded_agilex_gripper_keeps_layout_and_scales_once(tmp_path, converted):
    values = np.arange(14, dtype=np.float64) / 10
    values[[6, 13]] = [0.04, 0.02]
    if converted:
        values[[6, 13]] *= -87.5
    _write_gripper_dataset(tmp_path, values)
    config = _config.DataConfig(
        canonical_groups=_config.LerobotUnifiedDataConfig().canonical_groups(),
    )
    transforms = _data_loader._canonical_group_transforms(str(tmp_path), config)  # noqa: SLF001
    assert sum(isinstance(t, _transforms.LinearMapDims) for t in transforms) == (0 if converted else 1)
    result = _transforms.compose(transforms)({"observation.state": values, "action": values[None, :]})
    expected = np.zeros(62)
    expected[:6] = np.arange(6) / 10
    expected[7:13] = np.arange(7, 13) / 10
    expected[[22, 42]] = [-3.5, -1.75]
    np.testing.assert_allclose(result["observation.state"], expected)
    np.testing.assert_allclose(result["action"], expected[None, :])
    np.testing.assert_allclose(values[[6, 13]], [-3.5, -1.75] if converted else [0.04, 0.02])


@pytest.mark.parametrize(("robot_type", "width"), [("agilex_sim", 12), ("arx_sim", 14)])
def test_missing_effector_nonembedded_dataset_is_not_rescaled(tmp_path, robot_type, width):
    values = np.arange(width, dtype=np.float64) / 10
    _write_gripper_dataset(tmp_path, values, robot_type=robot_type)
    assert not _data_loader._has_embedded_gripper(str(tmp_path))  # noqa: SLF001
    config = _config.DataConfig(
        canonical_groups=_config.LerobotUnifiedDataConfig().canonical_groups(),
    )
    transforms = _data_loader._canonical_group_transforms(str(tmp_path), config)  # noqa: SLF001
    assert not any(isinstance(t, _transforms.LinearMapDims) for t in transforms)
    result = _transforms.compose(transforms)({"observation.state": values, "action": values[None, :]})
    half = width // 2
    np.testing.assert_allclose(result["observation.state"][:half], values[:half])
    np.testing.assert_allclose(result["observation.state"][7 : 7 + half], values[half:])
    np.testing.assert_array_equal(result["observation.state"][22:], 0)


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
