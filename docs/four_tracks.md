# Four-track simulation training and inference

π0.5/JAX training and simulation policy serving for exactly four tracks:

| Track | Norm groups | Reference task routes |
|---|---|---:|
| Whole-body control (WBC) | `g2_ctek90d_wbc` | 8 |
| Dexterous manipulation | `g2_wuji` | 10 |
| Fine manipulation | `g2_ctek90d_manip` | 8 |
| Multi-embodiment | `g1op`, `g2op`, `arx`, `agilex` | 24 |

This extension adds simulation policy serving only. It contains no hardware-control entrypoint. Existing upstream models, configuration names, examples and serving/client protocols remain available through their original entrypoints.

The four-track entrypoints use the upstream π0.5 model, optimizer and FSDP implementation with their own data adapters, normalization and simulation protocol. Use the original `scripts/train.py` and `scripts/serve_policy.py` for existing ACoT experiments. The new four-track server uses `scripts/four_track_inference_client.py`; the original client retains its metadata handshake and raw-observation protocol.

The common model uses `action_dim=64`, 50-step chunks, 350 prompt tokens and a 62-dimensional output layout:

```
0:7 left arm | 7:14 right arm | 14:19 waist | 19:22 base velocity
22:42 left effector | 42:62 right effector
```

Six-DOF arms are padded independently. Scalar grippers occupy slots 22 and 42; the dexterous hand occupies all 20 slots per side. Only the WBC group fills/emits waist and chassis blocks. Base state is masked; action velocities are integrated at 30 Hz.

## Install

Use Python 3.11, FFmpeg, and an NVIDIA environment compatible with the pinned JAX CUDA12 runtime for full π0.5 training/inference. Training memory depends on batch size, frozen parameters and FSDP; this source does not claim that full training fits one 24GB GPU.

```
GIT_LFS_SKIP_SMUDGE=1 uv sync --group four-tracks --frozen
```

The LeRobot dependency is pinned to a public GitHub revision. `openpi-client` is provided in this source workspace. No private registry or credentials are required. CPU tests can be run with `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''`.

## Prepare training data

The five manifests in `configs/` target native **LeRobot v2.1** datasets. The public [GenieSim dataset](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset) publishes four independent task-suite archives:

| Public category | Manifest |
|---|---|
| `whole_body_coordination` | `configs/wbc.json` |
| `dexterous_manipulation` | `configs/dexterous.json` |
| `precision_manipulation` | `configs/fine_manipulation.json` |
| `cross_embodiment_transfer` | `configs/multi_embodiment.json` |

Download every numbered part for each selected category, verify its `SHA256SUMS`, and concatenate the parts in order before extraction. The archive includes the category directory. For example, from the directory containing the whole-body parts:

```bash
sha256sum -c SHA256SUMS
mkdir -p /path/to/openpi/data
cat whole_body_coordination.tar.gz.part* | tar -xz -C /path/to/openpi/data
```

The templates expect `data/<category>/<task>/meta/info.json`, matching the public extraction layout. Each task is a separate dataset with `codebase_version: "v2.1"`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, per-episode Parquet data and image/video features. The loader uses the native LeRobot reader. It validates the version before loading and does not convert datasets or replace LeRobot metadata/statistics functions.

All datasets must use 30 Hz and three camera streams (`observation.images.top_head`, `observation.images.hand_left`, `observation.images.hand_right`). Task prompts come from the standard `task_index` / `meta/tasks.jsonl` mapping. Training samples all frames; omit `sampler` or set it to `""`. No segment annotations or extra per-episode metadata are required.

Robot vector semantics come from optional `features[feature].field_descriptions` in `meta/info.json`, or from an explicit `datasets[].field_map` in the manifest. Standard LeRobot does not prescribe joint semantics; supply a field map when these descriptions are absent. This leaves standard metadata untouched. Each map uses `state/...` names for `observation.state` and `action/...` for `action`; see [field-map example](../configs/README.md).

Arm joints must be ordered left then right (7 per arm, or 6 for ARX/Agilex). Scalar grippers require one value per side; the dexterous group requires 20 ordered joints per hand (thumb through little finger). WBC requires five waist joints and either three planar velocity commands `(vx, vy, yaw_rate)` or six linear/angular commands (selected as indices 0, 1, 5). Raw vector widths may differ across datasets: each dataset is remapped separately before concatenation and per-group normalization.

All manifest paths are resolved relative to the manifest file. Replace normalization-statistics and tokenizer paths with your own local resources. The source archive contains no demonstrations or weights; paths shown as `/path/to/...` are placeholders.

Provide the matching PaliGemma tokenizer locally, set `tokenizer_path`, and provide the pretrained π0.5 `params` directory when training. To obtain the tokenizer using the existing public resource downloader:

```
uv run --group four-tracks --frozen python scripts/prepare_four_track_tokenizer.py --output assets/tokenizer.model
```

Compute each normalization group on its own training data, then put the printed paths into the manifest's `norm_stats` mapping:

```
uv run --group four-tracks --frozen python scripts/compute_four_track_norm_stats.py --manifest configs/all.json --output norm_stats
```

The statistics include the same canonical remap and delta actions as training and retain each embodiment's gripper units. Never substitute statistics from a different checkpoint/data version.

## Train

```
uv run --group four-tracks --frozen python scripts/train_four_tracks.py \
  --manifest configs/all.json \
  --base-params /path/to/pretrained/pi05/params \
  --exp-name combined \
  --batch-size 32 --num-workers 4 --num-train-steps 450000
```

Use `configs/wbc.json`, `configs/dexterous.json`, `configs/fine_manipulation.json` or `configs/multi_embodiment.json` for independent track training. `--fsdp-devices` selects sharding; `--resume` restores a training checkpoint. W&B is disabled by default. Every newly saved checkpoint contains per-group statistics, the local tokenizer, routing/robot registries and a portable inference spec; optimizer state is not needed for serving.

## Export a checkpoint

For a newly trained checkpoint:

```
uv run --group four-tracks --frozen python scripts/export_four_track_checkpoint.py \
  --source checkpoints/four_tracks/combined/449999 --output released_ckpt
```

For an older checkpoint that has only parameters, additionally pass `--manifest configs/all.json` after supplying its exact matching statistics and tokenizer. Exporting an old checkpoint does not read training data, but the manifest must describe the normalization groups and model profile of those weights. A different model architecture is not supported by this four-track profile.

The complete portable package is:

```
released_ckpt/
  params/
  assets/
    inference.json
    norm_stats/<group>/norm_stats.json
    tokenizer.model
    task_routes.json
    robot_config.json
    licenses/                 # explicitly supplied legal documents
```

Export dereferences source file links and excludes optimizer state, training data and experiment paths. Keep the entire package together. Inference validates resource hashes, dimensions, finite statistics and group coverage before loading the model.

## License and attribution materials

The software uses the repository's Apache-2.0 license. The modified four-track files identify their public [ACoT-VLA/openpi source](https://github.com/Anonymous-694/ACoT-VLA/tree/bf16d89ab05c4c4dbd8ca83e50e51f85658fe090) and modifications in their headers. Dataset, tokenizer and model assets retain their applicable separate terms; the software license does not grant rights to those resources. Check the public dataset's license and the applicable [PaliGemma/Gemma terms](https://gemma.ai/resources/terms/) for the resources you distribute.

Supply the actual license, notice and use-restriction texts for your model/tokenizer. To include them in every training checkpoint, add a top-level manifest entry (paths are relative to the manifest):

```json
"license_files": ["../legal/MODEL_LICENSE.txt", "../legal/TOKENIZER_NOTICE.txt"]
```

The supplied UTF-8 documents are copied to `assets/licenses/` and checksummed, without recording their original paths. Inference and re-export use only these bundled copies. You can also attach them during export:

```bash
uv run --group four-tracks --frozen python scripts/export_four_track_checkpoint.py \
  --source /path/to/checkpoint --output released_ckpt \
  --license-file legal/MODEL_LICENSE.txt --license-file legal/TOKENIZER_NOTICE.txt
```

Export additionally preserves standard `LICENSE`, `LICENCE`, `NOTICE`, `COPYING` and `THIRD_PARTY_NOTICES` text files at the source checkpoint root and directly inside `assets/` (including `.txt`, `.md`, `.rst` and named variants). Other documents require explicit `license_files` / `--license-file` selection. Legal documents must have distinct basenames if their contents differ and be UTF-8 text of at most 1 MiB each. Conflicting documents and source attachments that link outside the source are rejected. The exporter preserves supplied terms; it does not determine redistribution rights or create missing asset permissions.

## Run inference

```
uv run --group four-tracks --frozen python scripts/serve_four_tracks.py --checkpoint released_ckpt --validate-only
uv run --group four-tracks --frozen python scripts/serve_four_tracks.py --checkpoint released_ckpt --host 0.0.0.0 --port 9000
```

The first command validates assets only; it does not prove the parameters match or run a model forward pass. The second restores the actual parameters. The checkpoint may be moved to any directory and requires no original training datasets or private paths. Its tokenizer is loaded locally, so model startup requires no tokenizer download.

Use `scripts/four_track_inference_client.py` to send a sample simulation request. `task_name` selects the normalization group; ambiguous shared task names additionally use `robot_type`. Unknown tasks/groups and incompatible hand widths are rejected. The output contains 50-step `left_arm`/`right_arm` JOINT_ABS trajectories, one- or 20-dimensional effectors, and WBC-only waist/wheels. A valid response is a policy prediction, not a guarantee of task success.

```
uv run --group four-tracks --frozen python scripts/four_track_inference_client.py --host localhost --port 9000 \
  --task clean_the_blackboard --robot g2a_sim --prompt "Clean the blackboard." --head head.jpg \
  --left-camera left.jpg --right-camera right.jpg --states states.json
```

`states.json` contains `arm_joint_states` (left then right, 12 or 14 total), `gripper_states` (two scalars or 40 hand joints), and WBC `waist_joint_states` (five joints). Requests use the existing msgpack JSON-RPC-style `infer` envelope. Unexpected failures return a short message, error code and error ID; full tracebacks remain in the server log. No physical hardware is connected by this client.

## Validation

CPU regressions cover real LeRobot file IO across all four tracks, canonical/group/prompt alignment, a tiny NNX optimization update, Orbax checkpoint saving/restoring, relocation without training data, asset rejection and simulation RPC with the tiny model. The tiny model test does not validate the full pretrained π0.5 network, real track datasets, convergence or benchmark success. Those require a separate run with the actual release weights/data on appropriate hardware before release acceptance.

```
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' uv run --group four-tracks --frozen pytest \
  src/openpi/four_tracks packages/openpi-client/src \
  -k 'not tokenize and not with_real_dataset'
```
