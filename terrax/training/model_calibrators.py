# Copyright 2026 Google LLC
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
"""Model calibration utilities."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, ClassVar, TypeAlias

import coordax as cx
from flax import nnx
import jax
from terrax.core import api
from terrax.core import checkpointing
from terrax.core import data_specs
from terrax.core import module_utils
from terrax.core import parallelism
from terrax.core import transforms
from terrax.training import checkpointing as training_checkpointing
from terrax.training import data_loading
import xarray

Model: TypeAlias = api.Model
DataLoader: TypeAlias = data_loading.DataLoader
TrainSchedule: TypeAlias = Any  # Avoid circular import with trainer.py


def _get_component(model: Any, key: str | None) -> Any:
  """Returns the model component at a '.'-separated `key` path.

  Each part of `key` is looked up as a key of a mapping or as an attribute
  otherwise. Mapping keys that contain '.' cannot be addressed.

  Args:
    model: Model to retrieve the component from.
    key: Path to the component. If None or empty, returns `model`.

  Raises:
    KeyError: If a mapping along the path does not contain the next part.
  """
  if not key:
    return model
  component = model
  for part in key.split('.'):
    if isinstance(component, Mapping):
      if part not in component:
        raise KeyError(f'{part!r} of {key=} not found in {list(component)}.')
      component = component[part]
    else:
      component = getattr(component, part)
  return component


def _canonicalize_component_keys(
    component_key: str | Sequence[str] | None,
) -> tuple[str | None, ...]:
  """Returns `component_key` as a tuple of keys accepted by `_get_component`."""
  if component_key is None or isinstance(component_key, str):
    return (component_key,)
  return tuple(component_key)


def _collect_all_node_paths(
    root: Any, target_types: type[Any] | tuple[Any, ...]
) -> dict[tuple[Any, ...], Any]:
  """Returns a mapping from all alias paths to nodes matching `target_types`.

  Unlike `nnx.iter_graph`, which yields each shared submodule only once at its
  first-encountered path, this helper expands duplicate paths from
  `nnx.find_duplicates` so that variables and submodules inside shared parent
  modules are indexed under every valid path from `root`.

  Args:
    root: Root NNX module or pytree node to traverse.
    target_types: Type or tuple of types to collect.

  Returns:
    Dictionary mapping each valid path tuple from `root` to the matching node.
  """
  all_paths = dict(nnx.iter_graph(root))
  duplicate_groups = nnx.find_duplicates(root)
  changed = True
  while changed:
    changed = False
    for primary_path, *alias_paths in reversed(duplicate_groups):
      n = len(primary_path)
      entries = [
          (path[n:], node)
          for path, node in all_paths.items()
          if path[:n] == primary_path
      ]
      for alias_path in alias_paths:
        if alias_path[:n] == primary_path:
          continue
        for suffix, node in entries:
          full_path = (*alias_path, *suffix)
          if full_path not in all_paths:
            all_paths[full_path] = node
            changed = True
  return {
      path: node
      for path, node in all_paths.items()
      if isinstance(node, target_types)
  }


class ModelCalibrator(abc.ABC):
  """Abstract base class for model calibrators.

  Calibrators are expected to update existing model variables in place (e.g.
  via `set_value`) rather than add or replace variables, which would drop
  metadata set by other calibrators (e.g. tags set by `SetVariableTags`), and
  to keep the placement of updated values (see `parallelism.shard_like`) so
  that calibration does not change the sharding of the model.

  `RolloutTrainer` runs calibrators in two groups, each in the listed order:
  calibrators with `run_on_restore=True` run when the trainer is constructed,
  all others run only when training starts from scratch, i.e. not when
  restoring from a checkpoint.

  Attributes:
    run_on_restore: Whether this calibrator should also apply when training is
      restored from a checkpoint. Set to True for calibrators that configure
      model metadata that is not saved in checkpoints (e.g. variable tags).
  """

  run_on_restore: ClassVar[bool] = False

  @abc.abstractmethod
  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    """Calibrates the model."""
    raise NotImplementedError()


@dataclasses.dataclass
class UpdateSubmodulesFromXarray(ModelCalibrator):
  """Updates submodules from Xarray datasets.

  For each `(cls, suffix)` key in `update_specs`, calls
  `module.update_from_xarray(dataset, **kwargs)` on every module of type `cls`
  whose path from the model root ends with `suffix`, where any path of a shared
  module counts. Implementations of `update_from_xarray` are expected to update
  their variables in place and keep their placement, e.g. via
  `var.set_value(parallelism.shard_like(value, var.get_value()))`.
  """

  # pylint: disable=g-bare-generic
  update_specs: dict[tuple[type, str], xarray.Dataset]
  kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  raise_on_no_update: bool = True
  # pylint: enable=g-bare-generic

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    del data_loader, train_schedule  # Unused.

    def path_to_str(path):
      return '.'.join(str(p) for p in path)

    for (cls, suffix), dataset in self.update_specs.items():
      path_to_module = _collect_all_node_paths(model, cls)
      updated_ids = set()
      for path, module in path_to_module.items():
        if id(module) not in updated_ids and path_to_str(path).endswith(suffix):
          module.update_from_xarray(dataset, **self.kwargs)
          updated_ids.add(id(module))

      if not updated_ids and self.raise_on_no_update:
        raise ValueError(
            f'No module of type {cls} with suffix "{suffix}" found for update.'
        )

    return model


def _has_collected_stats(stream_norm: transforms.StreamNorm) -> bool:
  """Returns whether `stream_norm` has accumulated statistics for any field."""
  counters = stream_norm.stream_norm.counters.get_value()
  return any(bool((count.data > 0).any()) for count in counters.values())


@dataclasses.dataclass
class CollectNormalizationStats(ModelCalibrator):
  """Collects normalization statistics by running the model on data.

  By default, statistics of all `StreamNorm` modules in the model are updated.
  `component_key` restricts updates to `StreamNorm` modules of the given model
  component(s), and `skip_initialized` skips modules that already have
  statistics (e.g. loaded from a checkpoint).
  """

  train_stage_idx: int
  rng_seed: int
  time_start: str
  time_end: str
  batch_count: int
  component_key: str | Sequence[str] | None = None
  skip_initialized: bool = False

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    stream_norms = []
    for key in _canonicalize_component_keys(self.component_key):
      component = _get_component(model, key)
      for stream_norm in module_utils.retrieve_subclass_modules(
          component, transforms.StreamNorm
      ):
        if not any(stream_norm is seen for seen in stream_norms):
          stream_norms.append(stream_norm)

    if self.skip_initialized:
      stream_norms = [s for s in stream_norms if not _has_collected_stats(s)]

    if not stream_norms:  # No stats to collect, can return early.
      return model

    for stream_norm in stream_norms:
      stream_norm.update_stats = True

    rng = jax.random.key(self.rng_seed)

    def unroll_fn(model, inputs, dynamic_data, rng, queries):
      model.reset_diagnostic_state()
      model.initialize_random_processes(cx.field(rng))
      model.update_dynamic_inputs(dynamic_data)
      model.assimilate(inputs)
      model.advance()
      observed = model.observe(queries)
      return observed

    unroll_fn = nnx.jit(unroll_fn)

    train_stage = train_schedule.stages[self.train_stage_idx]
    data_iter = data_loader.build_eval_inputs(
        train_stage.inputs_spec,
        train_stage.dynamic_inputs_spec,
        dataset_time_slice=(self.time_start, self.time_end),
        batch_size_per_device=None,
        time_sample_offset=train_stage.time_sample_offset,
        batch_count=self.batch_count,
    )

    # Helper to select a single last time slice in trajectory.
    sel_td_in_f = lambda f: f.isel(timedelta=-1)
    sel_timedelta = lambda t: jax.tree.map(sel_td_in_f, t, is_leaf=cx.is_field)

    for i, (inputs, dynamic_data) in enumerate(data_iter):
      init_slice = data_loading.sel_init_fields(inputs)
      # For queries we need a single time slice of input data.
      queries = data_specs.construct_query(
          sel_timedelta(init_slice), train_stage.queries_spec
      )
      result = unroll_fn(
          model, inputs, dynamic_data, jax.random.fold_in(rng, i), queries
      )
      jax.block_until_ready(result)

    # Set update_stats to False to avoid updating stats during training, and
    # reset any diagnostic state accumulated during calibration steps.
    for stream_norm in stream_norms:
      stream_norm.update_stats = False
    model.reset_diagnostic_state()

    return model


@dataclasses.dataclass
class LoadModelComponentParams(ModelCalibrator):
  """Loads parameters for a specific model component from a checkpoint.

  The checkpoint model is materialized on the device mesh of `model` and loaded
  values are placed like the parameters they replace, so the sharding of
  `model` is unchanged.
  """

  component_key: str | None
  ckpt_path_or_dir: str
  ckpt_step: int | None = None  # only used for training checkpoints.
  param_types: tuple[Any, ...] = (nnx.Param,)
  is_training_checkpoint: bool = False
  load_subset: bool = False

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    del data_loader, train_schedule  # Unused.
    if self.is_training_checkpoint:
      component = training_checkpointing.model_from_training_checkpoint(
          checkpoints_dir=self.ckpt_path_or_dir,
          step=self.ckpt_step,
          mesh=model.mesh,
      )
    else:
      component = checkpointing.load_model_checkpoint(
          path=self.ckpt_path_or_dir,
          mesh=model.mesh,
      )

    key = self.component_key
    to_update = _get_component(model, key)

    if self.load_subset:
      loaded_vars = _collect_all_node_paths(component, self.param_types)
      target_vars = _collect_all_node_paths(to_update, self.param_types)
      # Iterate in `to_update` traversal order so that a target variable that
      # is shared between several paths deterministically takes the value at
      # its first-visited path.
      common_paths = [path for path in target_vars if path in loaded_vars]
      if not common_paths:
        raise ValueError(
            f'No parameters of types {self.param_types} in checkpoint'
            f' {self.ckpt_path_or_dir!r} match the structure of component'
            f' {key!r}.'
        )
      updated_ids = set()
      for path in common_paths:
        target_var = target_vars[path]
        if id(target_var) not in updated_ids:
          loaded_value = loaded_vars[path].get_value()
          target_var.set_value(
              parallelism.shard_like(loaded_value, target_var.get_value())
          )
          updated_ids.add(id(target_var))
    else:
      # Compare and update using pure dicts of values so existing variable
      # metadata (e.g. tags set by `SetVariableTags`) is preserved.
      loaded_state = nnx.state(component, self.param_types)
      target_state = nnx.state(to_update, self.param_types)
      loaded_values = nnx.to_pure_dict(loaded_state)
      target_values = nnx.to_pure_dict(target_state)
      loaded_struct = jax.tree.structure(loaded_values)
      target_struct = jax.tree.structure(target_values)
      if loaded_struct != target_struct:
        raise ValueError(
            f'Parameter structures do not match for component {key}.\n'
            f'Loaded structure: {loaded_struct}\n'
            f'Target structure: {target_struct}'
        )
      nnx.update(
          to_update, parallelism.shard_like(loaded_values, target_values)
      )
    return model


@dataclasses.dataclass
class SetVariableTags(ModelCalibrator):
  """Sets or clears (`tag=None`) metadata tags on matching component variables.

  Tags can also be set directly in model code (e.g. via
  `nnx.Param(value, tag=...)`), in which case they are part of the model
  definition and persist in model checkpoints. This calibrator instead sets
  tags for a specific training experiment, e.g. to select trainable parameters
  via `RolloutTrainer.trainable_parameters_filter` (such as `nnx.WithTag(...)`
  or `nnx.Not(nnx.WithTag(...))`).

  Each variable holds at most one tag, so tags set by later calibrators replace
  earlier ones. Flax also uses the `tag` metadata to identify RNG streams, hence
  `variable_filter` should not select `nnx.RngState` variables.

  Attributes:
    component_key: Key(s) of model components whose variables are updated, see
      `_get_component`. If None, updates variables of the whole model.
    tag: Tag to set on matching variables. If None, removes existing tags.
    variable_filter: Filter that selects variables to update. Path-based filters
      are matched against all paths of a variable from the component, including
      aliases through shared submodules.
    run_on_restore: True, since tags are not saved in checkpoints and hence need
      to be set whenever the trainer is constructed.
  """

  component_key: str | Sequence[str] | None
  tag: str | None
  variable_filter: nnx.filterlib.Filter = nnx.Param
  run_on_restore: ClassVar[bool] = True

  def __call__(
      self,
      model: Model,
      data_loader: DataLoader,
      train_schedule: TrainSchedule,
  ) -> Model:
    del data_loader, train_schedule  # Unused.
    is_selected = nnx.filterlib.to_predicate(self.variable_filter)
    for key in _canonicalize_component_keys(self.component_key):
      component = _get_component(model, key)
      # Index variables under all of their paths, so that path-based filters
      # also match variables of shared submodules reached via secondary
      # aliases. A variable is selected if any of its paths matches.
      path_to_variable = _collect_all_node_paths(component, nnx.Variable)
      variables = {
          id(node): node
          for path, node in path_to_variable.items()
          if is_selected(path, node)
      }
      if not variables:
        raise ValueError(
            f'No variables matching {self.variable_filter=} found for'
            f' component {key!r}.'
        )
      for variable in variables.values():
        if self.tag is None:
          if variable.has_metadata('tag'):
            variable.del_metadata('tag')
        else:
          variable.set_metadata(tag=self.tag)
    return model
