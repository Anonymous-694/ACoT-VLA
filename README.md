<div align="center">

# ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models

**π₀ / π₀.₅ baseline for the AgiBot World Challenge — Reasoning to Action track**

[![arXiv](https://img.shields.io/badge/arXiv-2601.11404-b31b1b.svg)](https://arxiv.org/pdf/2601.11404v2)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow.svg)](https://huggingface.co/papers/2601.11404)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[**🏆 Challenge Baseline**](#challenge-baseline) ·
[**📰 News**](#news) ·
[**✨ About ACoT-VLA**](#about-acot) ·
[**📊 Benchmarks**](#benchmarks) ·
[**🚀 Get Started**](#get-started)

</div>

---

> This repository is the official **π₀ / π₀.₅ baseline** for the **[AgiBot World Challenge](https://agibot-world.com/challenge2026) — Reasoning to Action track**.
> It ships everything needed to reproduce the baseline end to end: dataset download,
> `pi0` / `pi05` training configs, and a tunnel-mode inference agent that connects to the challenge
> gateway. It is also the **official implementation of [ACoT-VLA](https://arxiv.org/abs/2601.11404v2)**
> (see [About ACoT-VLA](#about-acot) below).

---

<a id="challenge-baseline"></a>

## 🏆 AgiBot World Challenge Baseline (π₀ / π₀.₅)

This section walks through the complete leaderboard workflow. The released checkpoints let you skip
training and go straight to inference + submission.

<div align="center">

`Install`&nbsp; ──▸ &nbsp;`Download data`&nbsp; ──▸ &nbsp;`Train pi0 / pi05`&nbsp; ──▸ &nbsp;`Deploy tunnel inference`&nbsp; ──▸ &nbsp;`Submit a job`

</div>

### Board ↔ config ↔ checkpoint ↔ dataset map

The challenge is scored on four **boards**. Each board has matched `pi0` / `pi05` training configs, a
released checkpoint, and a source dataset suite:

| Board | Dataset suite | `pi05` train config | `pi0` train config | Released checkpoint (`checkpoints/<name>`) |
|-------|---------------|---------------------|--------------------|--------------------------------------------|
| `instruction` | `instruction` | `pi05_genie_sim_instruction_and_robust_20260526` | `pi0_genie_sim_instruction_and_robust_20260526` | `instruction_and_robust_pi05` |
| `robust`      | `instruction` | `pi05_genie_sim_instruction_and_robust_20260526` | `pi0_genie_sim_instruction_and_robust_20260526` | `instruction_and_robust_pi05` |
| `spatial`     | `instruction`† | `pi05_genie_sim_spatial_20260528` | `pi0_genie_sim_spatial_20260518` | `spatial_pi05` |
| `manip`       | `manipulation` | `pi05_genie_sim_manip_20260613` | `pi0_genie_sim_manip_20260526` | `manipulation_pi05` |

> `instruction` and `robust` share one checkpoint/config. A `sim2real` suite +
> `pi05_genie_sim_s2r_20260615` config are also provided for the sim2real boards. The ACoT submission
> config lives at `src/openpi/training/config.py` → `acot_icra_simulation_challenge_reasoning_to_action`.
>
> † Downloadable suites are `instruction` / `manipulation` / `sim2real`. The `spatial` config reads
> from spatial task folders — match the config's `repo_id` task names to the downloaded data; if a
> task is missing from a single suite, fetch the full dataset (`./scripts/download_dataset.sh` with
> no args). Always treat the `repo_id` task list in `config.py` as the source of truth.

<a id="install"></a>

### Step 1 · Install

We use **uv** to manage the Python environment.

```bash
git clone https://github.com/AgibotTech/ACoT-VLA.git
cd ACoT-VLA
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Step 2 · Download training data

Training data is hosted on **ModelScope** (`agibot_world/GenieSim3.0-Dataset`) in **LeRobot v2.1**
format and pulled with `scripts/download_dataset.sh`. Install the `modelscope` CLI once:

```bash
pip install modelscope
```

```bash
# Signature: ./scripts/download_dataset.sh [SUITE] [LOCAL_DIR]   (LOCAL_DIR default: ./data/)
./scripts/download_dataset.sh instruction          # -> ./data/instruction/
./scripts/download_dataset.sh manipulation         # -> ./data/manipulation/
./scripts/download_dataset.sh sim2real             # -> ./data/sim2real/
./scripts/download_dataset.sh                       # all three suites (large)
```

Each suite lands at `<LOCAL_DIR>/<suite>/` as a set of per-task LeRobot v2.1 datasets. Downloads are
resumable (re-running continues rather than restarting).

**Point the config at your data.** The `repo_id` lists in `src/openpi/training/config.py` use
internal absolute paths (`/mnt/public/...`). After downloading, edit the `repo_id` of the config you
plan to train so each entry points at the matching `./data/<suite>/<task>` directory.

### Step 3 · Train `pi0` / `pi05`

Pick the config for your board from the table above — `pi05_*` configs train a π₀.₅ policy, `pi0_*`
configs train a π₀ policy. Compute normalization stats, then launch training:

```bash
CONFIG=pi05_genie_sim_instruction_and_robust_20260526   # or any pi0_*/pi05_* config above
EXP=my_run

# 1. Compute normalization statistics for the config
uv run scripts/compute_norm_stats.py --config-name $CONFIG

# 2. Train (writes to checkpoints/<CONFIG>/<EXP>/<step>/)
bash scripts/train.sh $CONFIG $EXP
```

Swap `CONFIG` to a `pi0_*` name to train the π₀ variant instead (e.g.
`pi0_genie_sim_instruction_and_robust_20260526`). Defaults: batch size 256, 50k steps, checkpoints
every 5k steps (set `DEBUG_MODE=true` for a tiny smoke-test run).

**Or skip training and use the released checkpoints.** Download a baseline checkpoint straight into
`./checkpoints/` with `scripts/download_checkpoint.sh`:

```bash
# Signature: ./scripts/download_checkpoint.sh [NAME] [LOCAL_DIR]   (LOCAL_DIR default: ./checkpoints/)
./scripts/download_checkpoint.sh instruction_and_robust_pi05   # -> ./checkpoints/instruction_and_robust_pi05/
./scripts/download_checkpoint.sh spatial_pi05
./scripts/download_checkpoint.sh manipulation_pi05
```

Each lands at `./checkpoints/<name>/` with `params/`, `assets/` and `_CHECKPOINT_METADATA` directly
inside (no step subdir) — exactly the layout `tunnel.sh` expects.

### Step 4 · Deploy tunnel inference

Challenge inference runs as a **tunnel agent** (`scripts/tunnel_agent.py`, launched via
`scripts/tunnel.sh`) that reverse-dials the platform **gateway** over WebSocket and serves the
QUEUED → WARMUP → RUNNING → DRAINING job lifecycle. The on-wire payload stays
`msgpack_numpy`-encoded obs/action dicts, identical to the old `serve_policy.py` server, so the
simulator side is unchanged.

```bash
export CHALLENGE_TOKEN=<your challenge JWT>          # login credential
PI05_BOARD=instruction \
  ./scripts/tunnel.sh <CUDA_VISIBLE_DEVICES> <JOB_UUID> <GATEWAY_URL>
```

`tunnel.sh` selects the config + checkpoint from `PI05_BOARD` (`instruction` | `robust` | `spatial`
| `manip`) per the table above, so for the released checkpoints you only set the board. Useful knobs:

| Variable | Purpose |
|----------|---------|
| `PI05_BOARD` | Board to serve — picks config + `checkpoints/<...>` dir (default `instruction`) |
| `CHALLENGE_TOKEN` | Challenge JWT (required) |
| `PI05_CONFIG` / `PI05_CKPT_DIR` | Override the auto-selected config / checkpoint path (e.g. a self-trained `checkpoints/<config>/<exp>/<step>`) |
| `PI05_PARALLELISM` | Number of services packed on one GPU; sets `XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9 / N` |

**Packing multiple services on one GPU:** a single `pi05` service uses **< 8 GB**, so a 24 GB card
(e.g. RTX 4090) fits **at most 3**. Pass the *same* GPU index and tell each how many share the card:

```bash
N=3
for uuid in "${JOB_UUIDS[@]}"; do
  PI05_PARALLELISM=$N PI05_BOARD=instruction ./scripts/tunnel.sh 0 "$uuid" "$GATEWAY_URL" &
done
```

> Sanity-check before submitting: the agent log should reach `state -> RUNNING`, after which obs
> frames stream in and actions stream back.

### Step 5 · Submit a challenge job

1. **Get a token** — sign in on the
   [challenge quick-start](https://agibot-world.com/challenge2026/reasoning2action/quick-start) and
   copy your JWT into `CHALLENGE_TOKEN`.
2. **Create a job** — `POST /api/challenge/job` (board = the board you want to score). The response
   returns a **`uuid`** (your `JOB_UUID`) and the **tunnel/gateway endpoint** (`GATEWAY_URL`, of the
   form `wss://<host>/api/challenge/tunnel`).
3. **Launch the agent** against that job (Step 4):

   ```bash
   export CHALLENGE_TOKEN=<jwt>
   PI05_BOARD=<board> ./scripts/tunnel.sh 0 "$JOB_UUID" "$GATEWAY_URL"
   ```

   The board you pass **must match** the job's board so the right checkpoint is served.
4. **Track the result** — poll the job status / per-task scores via the `/api/challenge/*` endpoints
   until it reaches a terminal state, then read your score on the leaderboard.

> `tunnel.sh` / `tunnel_agent.py` replace the old host/port `serve_policy.py` entrypoint for challenge
> submission. `serve_policy.py` / `server.sh` remain available for **local** WebSocket eval.

---

<a id="news"></a>

## 📰 News

- 🚀🚀 **The [test server](https://agibot-world.com/challenge2026/reasoning2action/quick-start) of the AgiBot World Challenge is available now.**
- 🔥🔥 The minimal version of training code for the [AgiBot World Challenge](https://agibot-world.com/challenge2026) - Reasoning to Action track has been released.
- 🚀🚀 The training datasets of the [AgiBot World Challenge - Reasoning to Action track](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026/tree/main/Reasoning2Action-Sim) have been released.

---

<a id="about-acot"></a>

## ✨ About ACoT-VLA

This is the **official implementation** of [**ACoT-VLA**](https://arxiv.org/abs/2601.11404v2), a novel paradigm designed to bridge the fundamental semantic-kinematic gap in modern robotic policies. By shifting the locus of reasoning from perception to action, ACoT-VLA enables robots to "think" in the language of actions.

### 🌟 Overview

Existing VLA models often rely on indirect reasoning like sub-task prediction (language) or goal image synthesis (vision), which lack the granular information required for precise execution. We posit that the most effective form of reasoning is one that **deliberates directly in the action space**.

**Key Components:**

* **Explicit Action Reasoner (EAR):** A light-weight Transformer that synthesizes coarse-grained motion trajectories to provide direct motion cues.

* **Implicit Action Reasoner (IAR):** Extracts latent action priors from the internal representations of the VLM backbone using cross-attention modeling.

* **Action Chain-of-Thought (ACoT):** Together, EAR and IAR co-form an Action Chain-of-Thought, a reasoning paradigm where the deliberative process is formulated as structured action intents, enabling grounded and long-horizon policy learning.

<p align="center">
  <img src="docs/framework.png" alt="ACoT-VLA framework" width="90%">
</p>

---

<a id="benchmarks"></a>

## 📊 Performance Benchmarks

ACoT-VLA achieves state-of-the-art performance on multiple simulation benchmarks and exhibits superior robustness under distribution shifts.

### 1. LIBERO Benchmark

ACoT-VLA demonstrates significant improvements, particularly in the **LIBERO-Long** suite, by reducing ambiguity in mapping observations to actions.

| Method | Spatial | Object | Goal | Long | **Avg.** |
| --- | --- | --- | --- | --- | --- |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **ACoT-VLA (Frozen)** | **99.4** | **99.6** | 98.8 | 96.0 | **98.5** |
| **ACoT-VLA** | 98.6 | 99.0 | **99.4** | **97.0** | **98.5** |

> *Note: Models are trained on the LIBERO dataset. "Frozen" indicates the LLM backbone is frozen during training. All metrics are average success rates (%). The best results are highlighted in **bold**.*

### 2. LIBERO-Plus Robustness Evaluation

ACoT-VLA shows pronounced robustness under challenging perturbations like camera-viewpoint shifts and sensor noise.

| Setting | Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Zero-Shot** | $\pi_0^*$ | 61.0 | 40.8 | 63.5 | 89.3 | 84.1 | 80.1 | 76.4 | 69.4 |
| | $\pi_{0.5}^*$ | **75.8** | 79.4 | 83.3 | 95.5 | 95.0 | **89.6** | 87.0 | 85.7 |
| | **ACoT-VLA (Frozen)** | 68.9 | 80.3 | 84.1 | 95.6 | 93.1 | 81.5 | **88.3** | 83.6 |
| | **ACoT-VLA** | 72.6 | **82.6** | **87.5** | **97.7** | **96.5** | 87.8 | 88.1 | **86.6** |
| **SFT** | $\pi_0$ (Frozen) | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| | $\pi_{0.5}$ (Frozen) | 70.3 | 41.7 | **81.1** | **97.3** | 94.6 | 71.8 | 84.9 | 75.7 |
| | **ACoT-VLA (Frozen)** | 91.2 | 62.5 | 80.3 | 95.1 | 91.5 | 88.3 | 84.9 | 84.1 |
| | **ACoT-VLA** | **96.6** | **70.4** | 79.7 | 95.1 | **97.1** | **95.9** | **85.0** | **88.0** |

> *Note: Methods under **Zero-Shot** are trained on LIBERO and directly evaluated on LIBERO-Plus. **SFT** (Supervised Fine-Tuning) denotes models trained on the LIBERO-Plus training set. An asterisk (\*) denotes results reproduced using officially released checkpoints. "Frozen" indicates the LLM backbone is frozen during training. The best results are highlighted in **bold**.*

### 3. VLABench

Our method delivers substantial gains in unseen-texture tracks and complex tabletop scenarios. Comparison based on **Intention Score (IS)** and **Progress Score (PS)**.

| Method | In-dist. (IS/PS) | Category (IS/PS) | Commonsense (IS/PS) | Instruction (IS/PS) | Texture (IS/PS) | **Avg. (IS/PS)** |
| --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ (Frozen) | 67.8 / 62.7 | 44.0 / 33.6 | 54.9 / **43.0** | **58.0** / 38.7 | 50.6 / 42.5 | 55.0 / 44.1 |
| $\pi_{0.5}$ (Frozen) | 75.0 / 60.8 | 49.6 / 35.3 | **57.5** / 41.6 | 57.1 / 30.3 | 62.0 / 47.4 | 60.2 / 43.1 |
| **ACoT-VLA (Frozen)** | **79.8 / 66.1** | **54.1 / 38.9** | 52.3 / 37.8 | 56.8 / **39.6** | **74.6 / 54.6** | **63.5 / 47.4** |

> *Note: "Frozen" indicates that the LLM backbone is frozen during training. The best results are highlighted in **bold**.*

---

<a id="get-started"></a>

## 🚀 Get Started (ACoT-VLA on LIBERO / VLABench)

For the AgiBot World Challenge baseline, follow the [Challenge Baseline pipeline](#challenge-baseline) above. The steps below cover the ACoT-VLA paper experiments.

### 1. Installation

Same as challenge Step 1 — see [Install](#install).

### 2. Dataset Preparation

Datasets are processed into the **LeRobot format**.

```bash
python examples/libero/convert_libero_data_to_lerobot.py
```

### 3. Training & Inference

Follow the standardized pipeline to compute normalization statistics and launch training.

```bash
# Compute stats
uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>

# Start training
bash scripts/train.sh <CONFIG_NAME> <EXP_NAME>

# Launch a local policy server (host/port WebSocket, for LIBERO/VLABench-style eval)
bash scripts/server.sh <GPU_ID> <PORT>
```

---

## 📅 TODO List

* [x] Release core EAR and IAR training modules.
* [x] Release inference code.
* [x] Training configurations for **LIBERO**, **LIBERO-Plus**, and **VLABench**.
* [x] Official **AgiBot World Challenge** baseline (data download, `pi0`/`pi05` configs, tunnel inference).
* [x] Release baseline model checkpoints (`instruction_and_robust_pi05` / `spatial_pi05` / `manipulation_pi05`).
* [ ] Add training configurations for **CALVIN**.
* [ ] Add training configurations for **RoboCasa**.

---

## 📜 Citation

```bibtex
@article{zhong2026acot,
  title={ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models},
  author={Zhong, Linqing and Liu, Yi and Wei, Yifei and Xiong, Ziyu and Yao, Maoqing and Liu, Si and Ren, Guanghui},
  journal={arXiv preprint arXiv:2601.11404},
  year={2026}
}
```

## 🙏 Acknowledgements

This repo is built upon the [OpenPI](https://github.com/Physical-Intelligence/openpi) framework. We sincerely thank the authors for their contributions to the community.
