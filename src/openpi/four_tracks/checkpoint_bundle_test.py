"""Portable assets, failure checks and real LeRobot IO for the four simulation tracks.

The optimization smoke uses a tiny NNX model, not the released pi05 network.
"""

# ruff: noqa: SLF001
import dataclasses
import importlib.util
import json
from pathlib import Path
import shutil

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import sentencepiece

from openpi.four_tracks import checkpoint_bundle as bundle
from openpi.four_tracks.training import data_loader
from openpi.four_tracks.training.four_tracks import config_from_manifest
from openpi.models.model import BaseModel
from openpi.shared import normalize
from openpi.training import utils


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    root = tmp_path_factory.mktemp("tokenizer")
    prefix = str(root / "tokenizer")
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["current robot is g2_ctek90d_wbc pick the object 0 1 2 3 4 5 6 7 8 9"] * 20),
        model_prefix=prefix,
        vocab_size=64,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return Path(prefix + ".model")


def _stats(root, group):
    ns = normalize.NormStats(mean=np.zeros(64), std=np.ones(64), q01=-np.ones(64), q99=np.ones(64))
    normalize.save(root / group, {"state": ns, "actions": dataclasses.replace(ns)})
    return str(root / group / "norm_stats.json")


def _manifest(tmp_path, tokenizer, groups=("g2_ctek90d_wbc", "g2_ctek90d_manip", "g2_wuji", "arx")):
    stats = {g: _stats(tmp_path / "norm_stats", g) for g in groups}
    data = {
        "datasets": [{"path": "unavailable_training_data/" + g, "norm_group": g} for g in groups],
        "norm_stats": stats,
        "tokenizer_path": str(tokenizer),
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(data))
    return path


def _checkpoint(tmp_path, tokenizer):
    cfg = config_from_manifest(_manifest(tmp_path, tokenizer), validate_datasets=False)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    source = tmp_path / "source"
    (source / "params").mkdir(parents=True)
    (source / "params" / "fixture-only").write_text("asset tests do not contain model weights")
    bundle.write_assets(source / "assets", cfg, dc)
    return source


def test_complete_assets_survive_relocation_without_training_paths(tmp_path, tokenizer):
    source = _checkpoint(tmp_path, tokenizer)
    destination = bundle.export(source, tmp_path / "moved")
    shutil.rmtree(source)
    shutil.rmtree(tmp_path / "norm_stats")
    (tmp_path / "training.json").unlink()
    cfg = bundle.load_config(destination)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert cfg.data.repo_id is None
    assert set(dc.per_robot_norm_stats) == {"g2_ctek90d_wbc", "g2_ctek90d_manip", "g2_wuji", "arx"}
    assert cfg.model.action_dim == 64
    assert cfg.data.tokenizer_path.startswith(str(destination))
    assert dc.norm_stats is None
    assert len(cfg.data.canonical_groups()) == 6


@pytest.mark.parametrize("damage", ["missing", "modified", "nan", "unknown_group", "escape"])
def test_invalid_assets_fail_before_model_load(tmp_path, tokenizer, damage):
    checkpoint = _checkpoint(tmp_path, tokenizer)
    assets = checkpoint / "assets"
    path = assets / "inference.json"
    spec = json.loads(path.read_text())
    rel = next(iter(spec["norm_stats"].values()))
    if damage == "missing":
        (assets / rel).unlink()
    elif damage == "modified":
        (assets / rel).write_text("{}")
    elif damage == "nan":
        from hashlib import sha256

        d = json.loads((assets / rel).read_text())
        d["norm_stats"]["state"]["q01"][0] = float("nan")
        (assets / rel).write_text(json.dumps(d))
        spec["sha256"][rel] = sha256((assets / rel).read_bytes()).hexdigest()
    elif damage == "unknown_group":
        spec["norm_stats"]["not-a-group"] = spec["norm_stats"].pop("arx")
    else:
        spec["tokenizer"] = "../../outside.model"
        spec["sha256"][spec["tokenizer"]] = spec["sha256"].pop("tokenizer.model")
    path.write_text(json.dumps(spec))
    with pytest.raises((ValueError, FileNotFoundError)):
        bundle.inspect(checkpoint)


def test_export_preserves_legal_notices_without_copying_experiment_files(tmp_path, tokenizer):
    source = _checkpoint(tmp_path, tokenizer)
    documents = {
        "LICENSE": "Model license terms",
        "assets/NOTICE.txt": "Tokenizer attribution and use restrictions",
        "assets/LICENSE-tokenizer.md": "Tokenizer license terms",
    }
    for name, text in documents.items():
        (source / name).write_text(text)
    (source / "private-experiment.json").write_text("not a release document")
    destination = bundle.export(source, tmp_path / "exported")
    second = bundle.export(destination, tmp_path / "exported-again")
    for root in (destination, second):
        for name, text in documents.items():
            assert (root / name).read_text() == text
        assert not (root / "private-experiment.json").exists()


def test_manifest_legal_documents_survive_training_export_and_relocation(tmp_path, tokenizer):
    manifest = _manifest(tmp_path, tokenizer)
    document = tmp_path / "MODEL_LICENSE.txt"
    document.write_text("Model attribution, terms and use restrictions")
    data = json.loads(manifest.read_text())
    data["license_files"] = [document.name]
    manifest.write_text(json.dumps(data))
    config = config_from_manifest(manifest, validate_datasets=False)
    assert config.license_files == (str(document),)
    source = tmp_path / "source"
    (source / "params").mkdir(parents=True)
    bundle.write_assets(source / "assets", config, config.data.create(config.assets_dirs, config.model))
    spec = bundle.inspect(source)
    assert spec["license_files"] == ["licenses/MODEL_LICENSE.txt"]
    assert str(tmp_path) not in (source / "assets/inference.json").read_text()
    notice = tmp_path / "TOKENIZER_NOTICE.txt"
    notice.write_text("Tokenizer attribution")
    destination = bundle.export(source, tmp_path / "exported", license_files=[notice])
    document.unlink()
    notice.unlink()
    shutil.rmtree(source)
    moved = bundle.export(destination, tmp_path / "relocated")
    config = bundle.load_config(moved)
    assert len(config.license_files) == 2
    assert all(Path(p).is_relative_to(moved / "assets/licenses") for p in config.license_files)
    (moved / "assets/licenses/MODEL_LICENSE.txt").write_text("changed terms")
    with pytest.raises(ValueError, match="checksum"):
        bundle.inspect(moved)


def test_export_rejects_conflicting_legal_documents_and_escaping_symlinks(tmp_path, tokenizer):
    source = _checkpoint(tmp_path, tokenizer)
    first = tmp_path / "first/LICENSE.txt"
    second = tmp_path / "second/LICENSE.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("First license")
    second.write_text("Different license")
    with pytest.raises(ValueError, match="conflicting license"):
        bundle.export(source, tmp_path / "conflict", license_files=[first, second])
    assert not (tmp_path / "conflict").exists()
    (source / "LICENSE").symlink_to(first)
    with pytest.raises(ValueError, match="stay inside"):
        bundle.export(source, tmp_path / "escape")
    assert not (tmp_path / "escape").exists()


def _write_lerobot(root, group, *, use_videos=False):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dex = group == "g2_wuji"
    arm = 6 if group == "arx" else 7
    hand = 20 if dex else 1
    width = arm * 2 + hand * 2 + (5 if group == "g2_ctek90d_wbc" else 0)
    action_width = width + (3 if group == "g2_ctek90d_wbc" else 0)
    features = {
        "observation.state": {"dtype": "float32", "shape": (width,), "names": None},
        "action": {"dtype": "float32", "shape": (action_width,), "names": None},
        **{
            f"observation.images.{key}": {
                "dtype": "video" if use_videos else "image",
                "shape": (32, 32, 3),
                "names": ["height", "width", "channels"],
            }
            for key in ("top_head", "hand_left", "hand_right")
        },
    }
    ds = LeRobotDataset.create(
        repo_id=str(root),
        root=root,
        fps=30,
        robot_type="arx_sim" if arm == 6 else "g2a_sim",
        features=features,
        use_videos=use_videos,
    )
    for i in range(8):
        state = np.linspace(0.01, 0.2, width, dtype=np.float32)
        action = np.linspace(0.01, 0.2, action_width, dtype=np.float32) + np.float32(i * 0.001)
        ds.add_frame(
            {
                "observation.state": state,
                "action": action,
                "task": "pick the object",
                **{
                    f"observation.images.{key}": np.full((32, 32, 3), i, dtype=np.uint8)
                    for key in ("top_head", "hand_left", "hand_right")
                },
            }
        )
    ds.save_episode()
    p = root / "meta/info.json"
    info = json.loads(p.read_text())
    field_map = {}
    for feature, prefix in [("observation.state", "state"), ("action", "action")]:
        fields = {
            f"{prefix}/joint/position": {"indices": list(range(2 * arm))},
            f"{prefix}/left_effector/position": {"indices": list(range(2 * arm, 2 * arm + hand))},
            f"{prefix}/right_effector/position": {"indices": list(range(2 * arm + hand, 2 * arm + 2 * hand))},
        }
        if group == "g2_ctek90d_wbc":
            fields[f"{prefix}/waist/position"] = {"indices": list(range(16, 21))}
            if feature == "action":
                fields["action/robot/velocity"] = {"indices": list(range(21, 24))}
        field_map[feature] = fields
    assert info["codebase_version"] == "v2.1"
    assert all("field_descriptions" not in feature for feature in info["features"].values())
    return field_map


def test_real_lerobot_training_checkpoint_and_rpc_with_tiny_model(tmp_path, tokenizer, monkeypatch):
    groups = ("g2_ctek90d_wbc", "g2_ctek90d_manip", "g2_wuji", "arx")
    manifest_path = _manifest(tmp_path, tokenizer, groups)
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["datasets"]:
        root = tmp_path / item["path"]
        item["field_map"] = _write_lerobot(root, item["norm_group"], use_videos=item["norm_group"] == "g2_ctek90d_wbc")
    manifest_path.write_text(json.dumps(manifest))
    stats_spec = importlib.util.spec_from_file_location(
        "four_track_stats", Path(__file__).parents[3] / "scripts/compute_four_track_norm_stats.py"
    )
    stats_module = importlib.util.module_from_spec(stats_spec)
    stats_spec.loader.exec_module(stats_module)
    manifest["norm_stats"] = stats_module.compute(manifest_path, tmp_path / "computed_stats", max_frames=4)
    manifest_path.write_text(json.dumps(manifest))
    cfg = config_from_manifest(manifest_path)
    cfg = dataclasses.replace(cfg, batch_size=32, num_workers=2)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    from lerobot.common.datasets import lerobot_dataset

    native_metadata_loader = lerobot_dataset.LeRobotDatasetMetadata.load_metadata
    native_aggregate = lerobot_dataset.aggregate_stats
    raw = data_loader.create_torch_dataset(dc, cfg.model)
    assert lerobot_dataset.LeRobotDatasetMetadata.load_metadata is native_metadata_loader
    assert lerobot_dataset.aggregate_stats is native_aggregate
    assert len(raw._datasets) == 4
    for sub, group in zip(raw._datasets, groups, strict=True):
        sample = sub[0]
        assert sample["observation.state"].shape == (62,)
        assert sample["_robot_id"] == group
        assert sample["prompt"].startswith(f"current robot is {group}, ")
    loader = data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)
    batch = next(iter(loader))
    assert batch[0].state.shape == (32, 64)
    assert batch[1].shape == (32, 50, 64)
    assert np.isfinite(np.asarray(batch[1])).all()

    class Tiny(BaseModel):
        def __init__(self):
            super().__init__(64, 50, 350)
            self.kernel = nnx.Param(jnp.full((64, 64), 0.01))

        def compute_loss(self, rng, observation, actions, *, train=True):
            prediction = observation.state @ self.kernel.value
            return jnp.mean(jnp.square(prediction[:, None, :] - actions), axis=-1)

        def sample_actions(self, rng, observation):
            return jnp.repeat((observation.state @ self.kernel.value)[:, None, :], 50, axis=1)

    model = Tiny()
    tx = optax.adam(1e-3)
    params = nnx.state(model)
    state = utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params),
        ema_decay=None,
        ema_params=None,
    )
    spec = importlib.util.spec_from_file_location(
        "four_track_train", Path(__file__).parents[3] / "scripts/train_four_tracks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    updated, metrics = module.train_step(cfg, jax.random.key(0), state, batch)
    assert int(updated.step) == 1
    assert np.isfinite(float(metrics["loss"]))
    assert float(metrics["grad_norm"]) > 0

    # Exercise the real Orbax checkpoint callback, not only a hand-built assets fixture.
    import asyncio
    import io

    from openpi_client import msgpack_numpy
    from PIL import Image
    import websockets

    from openpi.four_tracks.policies import policy_config
    from openpi.four_tracks.serving import task_routing
    from openpi.four_tracks.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.four_tracks.training import checkpoints
    from openpi.models import model as model_lib
    from openpi.models import pi0

    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "trained", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, updated, loader, 0, config=cfg)
        manager.wait_until_finished()
    finally:
        manager.close()
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "trained", keep_period=None, overwrite=False, resume=True
    )
    try:
        assert resuming
        resumed = checkpoints.restore_state(manager, updated, loader, step=0)
        assert int(resumed.step) == 1
    finally:
        manager.close()
    saved = tmp_path / "trained" / "0"
    restored = model_lib.restore_params(saved / "params", restore_type=np.ndarray)
    np.testing.assert_allclose(restored["kernel"], np.asarray(updated.params["kernel"].value))
    portable = bundle.export(saved, tmp_path / "portable")

    # Only this test substitutes the network architecture: all IO, transforms and RPC are real.
    def load_tiny(self, params):
        net = Tiny()
        net.kernel.value = jnp.asarray(params["kernel"], dtype=jnp.float32)
        return net

    monkeypatch.setattr(pi0.Pi0Config, "load", load_tiny)
    shutil.rmtree(tmp_path / "unavailable_training_data")
    shutil.rmtree(tmp_path / "computed_stats")
    shutil.rmtree(tmp_path / "norm_stats")
    policy = policy_config.create_trained_policy(None, portable)
    server = WebsocketPolicyServer(
        policy,
        prepend_robot_type=True,
        arm_canonical_per_arm_dim=7,
        scenario="wbc",
        scenario_routing=True,
        unified_layout=True,
        prefix_from_norm_group=True,
        task_routes=task_routing.load(portable / "assets/task_routes.json"),
        waist_groups=("g2_ctek90d_wbc",),
    )
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="JPEG")

    class Socket:
        remote_address = ("cpu-fixture", 0)

        def __init__(self, request):
            self.request = request
            self.responses = []
            self.done = False

        async def recv(self):
            if self.done:
                raise websockets.ConnectionClosedOK(None, None)
            self.done = True
            return msgpack_numpy.Packer().pack(self.request)

        async def send(self, message):
            self.responses.append(msgpack_numpy.unpackb(message))

    cases = [
        ("clean_the_blackboard", "g2a_sim", 7, False, True),
        ("insert_building_block", "g2a_sim", 7, False, False),
        ("stack_bowls_wuji", "g2a_sim", 7, True, False),
        ("hang_cups_on_cup_tree_arxone", "arx_sim", 6, False, False),
    ]
    for task, robot, dof, dex, wbc in cases:
        request = {
            "method": "infer",
            "params": {
                "task_name": task,
                "robot_type": robot,
                "prompt": "pick the object",
                "images": {
                    key: {"encoding": "JPEG", "image_data": image.getvalue()}
                    for key in ("head", "hand_left", "hand_right")
                },
                "states": {
                    "arm_joint_states": [0.1] * (2 * dof),
                    "gripper_states": [0.0] * (40 if dex else 2),
                    "waist_joint_states": [0.5] * 5,
                },
            },
        }
        socket = Socket(request)
        asyncio.run(server._handler(socket))
        response = socket.responses[0]
        assert not response.get("error"), response
        result = response["result"]
        assert np.asarray(result["left_arm"]["values"]).shape == (50, dof)
        assert np.asarray(result["left_effector"]).shape == (50, 20 if dex else 1)
        assert ("waist" in result) == wbc
        assert ("wheels" in result) == wbc
        for key in ("left_effector", "right_effector"):
            assert np.isfinite(np.asarray(result[key])).all()


def test_unknown_group_cannot_silently_skip_normalization():
    from openpi.four_tracks import transforms

    stats = normalize.NormStats(mean=np.zeros(2), std=np.ones(2))
    transform = transforms.PerRobotNormalize({"arx": {"state": stats}})
    with pytest.raises(ValueError, match="normalization group"):
        transform({"state": np.ones(2), "_robot_id": "wrong"})
