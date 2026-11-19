"""Compute independent normalization statistics for the four simulation tracks."""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from openpi.four_tracks.training import data_loader
from openpi.four_tracks.training.four_tracks import config_from_manifest
from openpi.shared import normalize


def compute(manifest, output, max_frames=None):
    config = config_from_manifest(manifest, require_stats=False)
    factory = config.data
    results = {}
    for group in sorted(set(factory.norm_group_ids)):
        indices = [i for i, value in enumerate(factory.norm_group_ids) if value == group]
        one = dataclasses.replace(
            factory,
            repo_id=[factory.repo_id[i] for i in indices],
            norm_group_ids=[group] * len(indices),
            per_robot_norm_stats=None,
        )
        dc = one.create(config.assets_dirs, config.model)
        dataset = data_loader.create_torch_dataset(dc, config.model)
        nodes = list(getattr(dataset, "_datasets", [dataset]))
        keys = set()
        for node in nodes:
            while hasattr(node, "_dataset"):
                node = node._dataset  # noqa: PLW2901, SLF001
            features = node.meta.info["features"]
            for key in list(features):
                if features[key].get("dtype") == "video":
                    keys.add(key)
                    del features[key]

        def placeholder(item, keys=tuple(keys)):
            return {**item, **{key: np.zeros((224, 224, 3), dtype=np.uint8) for key in keys}}

        dataset = data_loader.TransformedDataset(
            dataset, [placeholder, *dc.repack_transforms.inputs, *dc.data_transforms.inputs]
        )
        running = {key: normalize.RunningStats() for key in ("state", "actions")}
        count = len(dataset) if max_frames is None else min(max_frames, len(dataset))
        if count <= 0:
            raise ValueError(f"no frames for {group}")
        for index in range(count):
            item = dataset[index]
            for key, accumulator in running.items():
                values = np.asarray(item[key])
                accumulator.update(values.reshape(-1, values.shape[-1]))
        normalize.save(Path(output) / group, {key: value.get_statistics() for key, value in running.items()})
        results[group] = str((Path(output) / group / "norm_stats.json").resolve())
    return results


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    print(json.dumps(compute(args.manifest, args.output, args.max_frames), indent=2))
