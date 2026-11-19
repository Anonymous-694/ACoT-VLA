# Derived from ACoT-VLA/openpi: src/openpi/policies/policy.py (public base bf16d89).
# Modified for four-track simulation training and inference.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.four_tracks import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        robot_id = inputs.get("_robot_id")
        inputs = self._input_transform(inputs)
        # _robot_id is routing metadata for PerRobotNormalize (already captured above for
        # the output side); drop it here so plain Normalize configs don't carry a string
        # key into the all-numeric jnp.asarray conversion below.
        inputs.pop("_robot_id", None)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        # Use a fixed RNG every call so identical payloads produce identical outputs.
        # We intentionally do NOT split self._rng across calls; advancing it would make
        # consecutive infers non-reproducible even with the same observation.
        sample_rng = self._rng
        outputs = {"state": inputs["state"]}
        result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)
        else:
            outputs["actions"] = result
        # outputs["actions"] = inputs["actions"]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        if robot_id is not None:
            outputs["_robot_id"] = robot_id
        outputs = self._output_transform(outputs)
        outputs.pop("_robot_id", None)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return self.post_process(obs, outputs)

    def post_process(self, obs: dict, outputs: dict) -> dict:
        actions = np.asarray(outputs["actions"])
        if actions.ndim != 2 or actions.shape[-1] != 62:
            raise ValueError(f"four-track policies must return (horizon,62), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy returned non-finite actions")
        return outputs


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
