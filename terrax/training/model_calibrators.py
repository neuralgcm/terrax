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
from terrax.core import checkpointing
from terrax.core import data_specs
from terrax.core import module_utils
from terrax.core import transforms
from terrax.training import checkpointing as training_checkpointing
from terrax.training import data_loading
import xarray

Model: TypeAlias = nnx.Module  # Avoid circular import with api.py
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


class ModelCalibrator(abc.ABC):
  """Abstract base class for model calibrators.

  Calibrators are expected to update existing model variables in place (e.g.
  via `set_value`) rather than add or replace variables, which would drop
  metadata set by other calibrators (e.g. tags set by `SetVariableTags`).

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
  """Updates submodules from Xarray datasets."""

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

    # Find duplicates to handle multiple paths to the same object.
    duplicate_paths_groups = nnx.graph.find_duplicates(model)
    # Maps a path string to a set of all equivalent path strings (aliases).
    path_to_aliases = {}

    def path_to_str(path):
      return '.'.join(str(p) for p in path)

    for group in duplicate_paths_groups:
      # group is a list of paths (tuples of keys).
      aliases = {path_to_str(path) for path in group}
      for path in group:
        path_to_aliases[path_to_str(path)] = aliases

    for (cls, suffix), dataset in self.update_specs.items():
      modules, paths = module_utils.retrieve_subclass_modules(
          model, cls, return_paths=True
      )

      updated = False
      for module, path in zip(modules, paths):
        full_path = path_to_str(path)
        candidates = path_to_aliases.get(full_path, {full_path})

        if any(candidate.endswith(suffix) for candidate in candidates):
          module.update_from_xarray(dataset, **self.kwargs)
          updated = True

      if not updated and self.raise_on_no_update:
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

    # Set update_stats to False to avoid updating stats during training.
    for stream_norm in stream_norms:
      stream_norm.update_stats = False

    return model


@dataclasses.dataclass
class LoadModelComponentParams(ModelCalibrator):
  """Loads parameters for a specific model component from a checkpoint."""

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
    # TODO(dkochkov): Consider if we need to provide spmd_mesh updates. For
    # standard model parameters this should not be necessary, since we tend to
    # store them in a non-padded format, but this might be safer or useful in
    # other cases.
    if self.is_training_checkpoint:
      component = training_checkpointing.model_from_training_checkpoint(
          checkpoints_dir=self.ckpt_path_or_dir, step=self.ckpt_step
      )
    else:
      component = checkpointing.load_model_checkpoint(
          path=self.ckpt_path_or_dir
      )

    loaded_state = nnx.state(component, self.param_types)
    key = self.component_key
    to_update = _get_component(model, key)
    target_state = nnx.state(to_update, self.param_types)

    if self.load_subset:
      loaded_vars = {
          path: node
          for path, node in nnx.graph.iter_graph(component)
          if isinstance(node, self.param_types)
      }
      # Expanding loaded_vars to include duplicate aliases.
      for group in nnx.graph.find_duplicates(component):
        canonical = next((p for p in group if p in loaded_vars), None)
        if canonical is not None:
          val = loaded_vars[canonical]
          for p in group:
            loaded_vars[p] = val
      target_vars = {
          path: node
          for path, node in nnx.graph.iter_graph(to_update)
          if isinstance(node, self.param_types)
      }
      # Expanding target_vars to include duplicate aliases symmetrically.
      # This ensures that if module has extra references that affect the
      # traversal order in `to_update`, we still identify such duplicates.
      for group in nnx.graph.find_duplicates(to_update):
        canonical = next((p for p in group if p in target_vars), None)
        if canonical is not None:
          val = target_vars[canonical]
          for p in group:
            target_vars[p] = val

      common_keys = set(loaded_vars.keys()).intersection(target_vars.keys())
      for path in common_keys:
        target_vars[path].set_value(loaded_vars[path].get_value())
    else:
      # Compare and update using pure dicts of values so existing variable
      # metadata (e.g. tags set by `SetVariableTags`) is preserved.
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
      nnx.update(to_update, loaded_values)
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
    variable_filter: Filter that selects variables to update.
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
      variables = [
          node
          for path, node in nnx.iter_graph(component)
          if isinstance(node, nnx.Variable) and is_selected(path, node)
      ]
      if not variables:
        raise ValueError(
            f'No variables matching {self.variable_filter=} found for'
            f' component {key!r}.'
        )
      for variable in variables:
        if self.tag is None:
          if variable.has_metadata('tag'):
            variable.del_metadata('tag')
        else:
          variable.set_metadata(tag=self.tag)
    return model
