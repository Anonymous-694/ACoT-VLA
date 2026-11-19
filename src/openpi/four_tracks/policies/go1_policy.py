# Derived from ACoT-VLA/openpi: src/openpi/policies/go1_policy.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

"""Policy transforms for the Go1 robot."""

from collections.abc import Sequence
import copy
import dataclasses
import json
import logging
import pathlib
from typing import ClassVar

import numpy as np
import torch

import openpi.four_tracks.transforms as transforms
import openpi.models.model as _model


def load_field_indices(
    repo_id: str | list[str], feature_key: str, field_keys: Sequence[str], *, field_map=None
) -> list[int]:
    if isinstance(repo_id, list):
        repo_id = repo_id[0]
    info_path = pathlib.Path(repo_id) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})
    feature = features.get(feature_key)
    if feature is None:
        raise KeyError(f"Feature '{feature_key}' not found in info.json")

    from openpi.four_tracks.training.dataset_v21 import feature_fields

    field_descriptions = feature_fields(info, feature_key, field_map)
    indices: list[int] = []
    for key in field_keys:
        if key not in field_descriptions:
            raise KeyError(f"Field '{key}' not found in field_descriptions of '{feature_key}'")
        indices.extend(field_descriptions[key]["indices"])
    logging.info("Loaded %d indices for %s from %s: fields=%s", len(indices), feature_key, info_path, field_keys)
    return indices


@dataclasses.dataclass(frozen=True)
class Go1Inputs(transforms.DataTransformFn):
    """Inputs for the Go1 policy.
    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [N] (variable dimension, remapped via state_indices)
    - actions: [action_horizon, M] (variable dimension, remapped via action_indices)
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    state_indices: tuple[int, ...] | None = None
    action_indices: tuple[int, ...] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")
    rename_map = {"top_head": "base_0_rgb", "hand_left": "left_wrist_0_rgb", "hand_right": "right_wrist_0_rgb"}

    def _remap_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_indices is not None:
            size = state.shape[-1]
            if size > max(self.state_indices):
                state = state[list(self.state_indices)]
        if self.state_mask is not None:
            if len(state) > len(self.state_mask):
                state = state[: len(self.state_mask)]
            state[self.state_mask[: len(state)]] = 0
        return state

    def _remap_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.action_indices is not None and actions.shape[1] > max(self.action_indices):
            actions = actions[:, list(self.action_indices)]
        if self.action_mask is not None:
            if actions.shape[1] > len(self.action_mask):
                actions = actions[:, : len(self.action_mask)]
            actions[:, self.action_mask[: actions.shape[1]]] = 0
        return actions

    def __call__(self, data: dict) -> dict:
        state = copy.deepcopy(data["state"])
        state = self._remap_state(state)
        state = transforms.pad_to_dim(state, self.action_dim)
        state = state.squeeze()

        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if "actions" in data:
            actions = data["actions"]
            actions = self._remap_actions(actions)
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions.squeeze()

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "_robot_id" in data:
            inputs["_robot_id"] = data["_robot_id"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go1Outputs(transforms.DataTransformFn):
    """Outputs for the Go1 policy."""

    output_dim: int = 22

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_dim])}
