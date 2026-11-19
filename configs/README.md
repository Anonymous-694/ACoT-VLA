# Four-track manifest paths

The JSON files are runnable configuration templates after you supply the required resources. JSON does not support comments, so the path placeholders are documented here.

| Key | Example | Supply |
|---|---|---|
| `datasets[].path` | `../data/<category>/<task>` | A native LeRobot v2.1 dataset root containing `meta/info.json` |
| `norm_stats.<group>` | `../norm_stats/<group>/norm_stats.json` | Statistics computed from the matching training data |
| `tokenizer_path` | `../assets/tokenizer.model` | The PaliGemma tokenizer used by the model |

Relative paths are resolved against the JSON file's directory, not the shell's current directory. With the provided manifests in `configs/`, `../data/` refers to the repository's `data/` directory. You may edit these entries to use your own absolute or relative paths. Keep local datasets, checkpoints and generated assets out of version control.

The task names, normalization groups and track labels describe the four simulation tracks. Preserve their correspondence when changing dataset locations. Choose `wbc.json`, `dexterous.json`, `fine_manipulation.json` or `multi_embodiment.json` for a single track, or `all.json` for joint training.

`--base-params /path/to/pretrained/pi05/params` in the four-track training command is also a placeholder: replace it with your own pretrained parameter directory. Inference uses the resources inside the exported checkpoint's `assets/` directory and does not need training-data paths.

The dataset templates match the four public archive category directories. Keep the category layer when extracting into `data/`. All frames are sampled (`"sampler": ""`); prompts use the standard task-index mapping.

If `meta/info.json` has no semantic `field_descriptions`, add `field_map` to each dataset entry. The following example assumes 14 arm joints followed by two scalar grippers in both vectors; inspect your dataset before assigning indices:

```json
{
  "path": "../data/precision_manipulation/example_task",
  "norm_group": "g2_ctek90d_manip",
  "track": "fine_manipulation",
  "field_map": {
    "observation.state": {
      "state/joint/position": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
      "state/left_effector/position": [14],
      "state/right_effector/position": [15]
    },
    "action": {
      "action/joint/position": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
      "action/left_effector/position": [14],
      "action/right_effector/position": [15]
    }
  }
}
```

A feature mapping replaces that feature's metadata descriptions, so include all its required fields. Values may also be objects such as `{"indices": [0, 1], "joint_names": ["joint_a", "joint_b"]}`. Indices are checked against the raw vector width. WBC adds `state/waist/position`, `action/waist/position` and `action/robot/velocity`; dexterous hands each require 20 indices. These mappings are training-only and are not copied into portable inference assets.

Optional top-level `license_files` lists local model/tokenizer license and notice text files. Paths are resolved relative to the manifest, and training copies the documents into each checkpoint's `assets/licenses/` with hashes. Original document paths are not stored in the portable inference spec. Supply the applicable terms for your own model resources; see [license materials](../docs/four_tracks.md#license-and-attribution-materials).
