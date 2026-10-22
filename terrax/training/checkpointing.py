# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for checkpointing training runs."""

from collections.abc import Sequence
import dataclasses
import json
from typing import Any

from absl import logging
import fiddle
from fiddle.experimental import serialization
from flax import nnx
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as ocp_managers
from terrax.core import api
from terrax.core import checkpointing as model_checkpointing
from terrax.core import parallelism

# pylint: disable=logging-fstring-interpolation


@dataclasses.dataclass
class LastNSteps(ocp_managers.PreservationPolicy):
  """Ensures sufficient checkpoints are preserved for lookback restoration."""

  lookback_steps: int

  def should_preserve(
      self,
      checkpoints: Sequence[ocp_managers.PolicyCheckpointInfo],
      *,
      context: ocp_managers.PreservationContext,
  ) -> Sequence[bool]:
    latest_step = max(ckpt.step for ckpt in checkpoints)
    result = [
        latest_step - ckpt.step <= self.lookback_steps for ckpt in checkpoints
    ]
    return result


_MODEL_CONFIG_KEY = 'fiddle_model_config'
_STEP_FORMAT_FIXED_LENGTH = 6


def _checkpoint_manager_options_with_fixed_step_format(
    options: ocp.CheckpointManagerOptions | None = None,
) -> ocp.CheckpointManagerOptions:
  """Returns a CheckpointManagerOptions with fixed step format."""
  if options is None:
    options = ocp.CheckpointManagerOptions()
  if options.step_format_fixed_length is not None:
    raise ValueError('cannot customize step_format_fixed_length')
  return dataclasses.replace(
      options, step_format_fixed_length=_STEP_FORMAT_FIXED_LENGTH
  )


def training_manager(
    directory: str,
    options: ocp.CheckpointManagerOptions,
    model_config_str: str,
    metadata: dict[str, Any] | None = None,
) -> ocp.CheckpointManager:
  """Returns a CheckpointManager for training."""
  model_config_dict = model_checkpointing.without_paths(
      json.loads(model_config_str)
  )
  options = _checkpoint_manager_options_with_fixed_step_format(options)
  if metadata is None:
    metadata = {}
  if _MODEL_CONFIG_KEY in metadata:
    raise ValueError(f'{_MODEL_CONFIG_KEY!r} already set in metadata')
  metadata[_MODEL_CONFIG_KEY] = model_config_dict
  return ocp.CheckpointManager(directory, options=options, metadata=metadata)


def read_only_manager(directory: str) -> ocp.CheckpointManager:
  """Returns a CheckpointManager that is read-only."""
  options = _checkpoint_manager_options_with_fixed_step_format()
  options.read_only = True
  return ocp.CheckpointManager(directory, options=options)


def load_fiddle_json(json_str: str) -> fiddle.Config:
  """Loads a Fiddle config from a JSON string."""
  # Use replacements to map from old to new module names, e.g.,
  # 'google3.research.simulation.neuralgcm.something': 'neuralgcm.something'
  replacements = {}
  for k, v in replacements.items():
    json_str = json_str.replace(k, v)
  return serialization.load_json(json_str)


def untrained_model_from_training_dir(
    checkpoints_dir: str,
    mesh: parallelism.MeshUpdates = None,
    attrs_to_update: Sequence[str] = ('spmd_mesh',),
    *,
    eager: bool = False,
) -> api.Model:
  """Loads a Model from training directory metadata without restoring weights.

  Args:
    checkpoints_dir: Directory containing training checkpoints and metadata.
    mesh: Optional `parallelism.Mesh` (or mapping from tag types to `Mesh`
      instances) specifying the device mesh and partition schemas to use when
      constructing the model. If `None`, builds the model with its saved
      configuration (without sharding).
    attrs_to_update: `Mesh` attribute names to copy from `mesh` into the saved
      model config when `mesh` is provided. Defaults to `('spmd_mesh',)`.
    eager: Whether to build the model eagerly instead of inside `nnx.jit`. See
      `api.Model.from_fiddle_config`.

  Returns:
    An initialized `api.Model` instance before checkpoint restoration.
  """
  logging.info('loading model metadata')
  with read_only_manager(checkpoints_dir) as manager:
    model_config_dict = manager.metadata().custom_metadata[_MODEL_CONFIG_KEY]
  # TODO(shoyer): avoid unnecessary serialization to/from JSON. This may require
  # refactoring fiddle.experiment.serialization.
  logging.info('loading model config into fiddle')
  model_config = load_fiddle_json(json.dumps(model_config_dict))
  logging.info('building model')
  return api.Model.from_fiddle_config(
      model_config,
      mesh=mesh,
      attrs_to_update=attrs_to_update,
      eager=eager,
  )


def model_from_training_checkpoint(
    checkpoints_dir: str,
    step: int | None = None,
    use_ema_params: bool = True,
    mesh: parallelism.MeshUpdates = None,
    attrs_to_update: Sequence[str] = ('spmd_mesh',),
    *,
    eager: bool = False,
) -> api.Model:
  """Loads a Model and its parameters from a training checkpoint.

  Args:
    checkpoints_dir: Directory containing training checkpoints.
    step: Training step to restore. If `None`, restores the latest step.
    use_ema_params: Whether to restore EMA parameters (`ema_params`) or raw
      parameters (`params`).
    mesh: Optional `parallelism.Mesh` (or mapping from tag types to `Mesh`
      instances) specifying the device mesh and partition schemas for the
      restored model and parameters. If `None`, the model and its parameters are
      loaded onto a single local device regardless of the accelerator topology
      used during training.
    attrs_to_update: `Mesh` attribute names to copy from `mesh` into the saved
      model config when `mesh` is provided. Defaults to `('spmd_mesh',)`.
    eager: Whether to build the model eagerly instead of inside `nnx.jit`. See
      `api.Model.from_fiddle_config`.

  Returns:
    The restored `api.Model` instance.
  """
  logging.info(f'Loading checkpoint for step {step} from {checkpoints_dir}')
  model = untrained_model_from_training_dir(
      checkpoints_dir,
      mesh=mesh,
      attrs_to_update=attrs_to_update,
      eager=eager,
  )
  state = model_checkpointing.split_model_state_for_saving(model)

  param_key = 'ema_params' if use_ema_params else 'params'
  restore_kwargs = {
      'metadata': ocp.args.JsonRestore(),
      param_key: ocp.args.StandardRestore(state.params),
  }
  has_non_params = bool(jax.tree.leaves(state.non_params))
  if has_non_params:
    restore_kwargs['non_params'] = ocp.args.StandardRestore(state.non_params)
  restore_args = ocp.args.Composite(**restore_kwargs)

  with read_only_manager(checkpoints_dir) as manager:
    if step is None:
      step = manager.latest_step()
      if step is None:
        raise ValueError(f'Found no checkpoints in {checkpoints_dir}')
    logging.info(f'Loading checkpoint for step {step} from {checkpoints_dir}')
    ckpt = manager.restore(step, args=restore_args)

  params = ckpt[param_key]
  nnx.update(model, params)
  if has_non_params:
    nnx.update(model, ckpt.non_params)
  return model


def export_training_to_model_checkpoint(
    train_checkpoints_dir: str,
    model_checkpoint_path: str,
    train_step: int | None = None,
    use_ema_params: bool = True,
    *,
    eager: bool = False,
) -> None:
  """Loads a training checkpoint and saves it as a standalone model checkpoint.

  Args:
    train_checkpoints_dir: The directory containing training checkpoints.
    model_checkpoint_path: The path to save the model checkpoint to.
    train_step: The step of the training checkpoint to load. If None, the latest
      checkpoint will be loaded.
    use_ema_params: Whether to use the EMA params or the non-EMA params.
    eager: Whether to build the model eagerly instead of inside `nnx.jit`. See
      `api.Model.from_fiddle_config`.
  """
  model = model_from_training_checkpoint(
      train_checkpoints_dir,
      step=train_step,
      use_ema_params=use_ema_params,
      eager=eager,
  )
  model_checkpointing.save_checkpoint(model, model_checkpoint_path)
