# Copyright 2024 Google LLC
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
"""Helper utilities for checkpointing Fiddle configs used in NeuralGCM."""

import dataclasses
import json
from typing import Any

from etils import epath
import fiddle as fdl
from fiddle.experimental import serialization
from flax import nnx
import jax
import orbax.checkpoint as ocp
from terrax.core import api
from terrax.core import parallelism
from terrax.core import typing

ParamInfo = ocp.type_handlers.ParamInfo
Metadata = ocp.metadata.value.Metadata


@dataclasses.dataclass(frozen=True)
class _SplitState:
  params: nnx.State
  non_params: nnx.State


def split_model_state_for_saving(model: nnx.Module) -> _SplitState:
  """Extracts model state to save from an nnx.Module."""
  params, _, non_params = nnx.state(
      model, nnx.Param, (typing.SimulationVariable, typing.DynamicInput), ...
  )
  return _SplitState(params=params, non_params=non_params)


def without_paths(json_data: Any) -> Any:
  """Returns a copy of `json_data` with all `paths` entries removed.

  Fiddle annotates each serialized object with every path that reaches it from
  the root. Deserialization ignores these annotations, but for large configs
  they can account for over 99% of the serialized bytes.

  Args:
    json_data: parsed JSON data of a serialized Fiddle config, e.g.
      `json.loads(serialization.dump_json(config))`. Dicts and lists are
      traversed recursively; all other values are returned unchanged.

  Returns:
    The same JSON data with every `paths` key dropped from nested dicts.
  """
  if isinstance(json_data, dict):
    items = json_data.items()
    return {k: without_paths(v) for k, v in items if k != 'paths'}
  if isinstance(json_data, list):
    return [without_paths(v) for v in json_data]
  return json_data


_STATE_KEY = 'state'
_CONFIG_KEY = 'fiddle_config'


def load_model_checkpoint(
    path: str | epath.PathLike,
    spmd_mesh_updates: (
        dict[parallelism.TagOrMeshType, jax.sharding.Mesh | None] | None
    ) = None,
    array_partitions_updates: (
        dict[parallelism.TagOrMeshType, parallelism.ArrayPartitions] | None
    ) = None,
    field_partitions_updates: (
        dict[parallelism.TagOrMeshType, parallelism.FieldPartitions] | None
    ) = None,
) -> api.Model:
  """Loads a Model from a checkpoint."""
  checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())

  # Create model from checkpoint metadata.
  config_args = ocp.args.Composite(**{_CONFIG_KEY: ocp.args.JsonRestore()})
  model_config_dict = checkpointer.restore(path, config_args)[_CONFIG_KEY]
  model_config = serialization.load_json(json.dumps(model_config_dict))
  model = api.Model.from_fiddle_config(
      model_config,
      spmd_mesh_updates=spmd_mesh_updates,
      array_partitions_updates=array_partitions_updates,
      field_partitions_updates=field_partitions_updates,
  )

  # Set model parameters from checkpoint.
  state_tuple = split_model_state_for_saving(model)
  state = nnx.merge_state(state_tuple.params, state_tuple.non_params)
  state_args = ocp.args.Composite(**{_STATE_KEY: ocp.args.PyTreeRestore(state)})
  restored = checkpointer.restore(path, state_args)[_STATE_KEY]
  nnx.update(model, restored)
  return model


def save_checkpoint(
    model: api.Model,
    path: str | epath.PathLike,
    fiddle_config: fdl.Config[api.Model] | None = None,
):
  """Saves model to a checkpoint."""
  if fiddle_config is None:
    fiddle_config = model.fiddle_config

  if not isinstance(fiddle_config, fdl.Config):
    raise TypeError(f'must supply a fiddle.Config, got {fiddle_config=}')

  state_tuple = split_model_state_for_saving(model)
  state = nnx.merge_state(state_tuple.params, state_tuple.non_params)
  serialized_config = serialization.dump_json(fiddle_config)
  model_config_dict = without_paths(json.loads(serialized_config))
  args = ocp.args.Composite(**{
      _STATE_KEY: ocp.args.PyTreeSave(state),
      _CONFIG_KEY: ocp.args.JsonSave(model_config_dict),
  })
  checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
  checkpointer.save(path, args)
