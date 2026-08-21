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
"""High performance inference API for NeuralGCM models."""

import contextlib
import dataclasses
import functools
import logging
import math
import pickle
from typing import Any, TypeVar
import uuid

import coordax as cx
import dask.array
from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from terrax.core import api
from terrax.core import data_specs
from terrax.core import typing
from terrax.core import xarray_utils
from terrax.inference import dynamic_inputs as dynamic_inputs_lib
from terrax.inference import streaming
from terrax.inference import timing
import xarray

# pylint: disable=logging-fstring-interpolation

NestedData = dict[str, xarray.Dataset]  # eventually, use xarray.DataTree


def _queries_to_dummy_datatree(queries: typing.Queries) -> xarray.DataTree:
  """Create a DataTree of dummy dask.array objects, matching the given queries."""
  outputs = {}
  for group, sub_query in queries.items():
    ds = xarray.Dataset()
    for var, c in sub_query.items():
      if isinstance(c, data_specs.FieldInQuerySpec):
        c = c.spec
      coords = c.coordinate.to_xarray() if cx.is_field(c) else c.to_xarray()  # pyrefly: ignore[missing-attribute]
      ds[var] = xarray.DataArray(
          # TODO(shoyer): include dtype as part of the query spec?
          data=dask.array.zeros(c.shape, np.float32),  # pyrefly: ignore[missing-attribute]
          dims=c.dims,  # pyrefly: ignore[missing-attribute]
          coords=coords,
      )
    outputs[group] = ds
  return xarray.DataTree.from_dict(outputs)


# TODO(shoyer): Add these methods upstream onto xarray.DataTree.


def _datatree_expand_dims(
    tree: xarray.DataTree, **kwargs: int | list[Any] | np.ndarray
) -> xarray.DataTree:
  """Expand dimensions in a DataTree."""
  return tree.map_over_datasets(lambda ds: ds.expand_dims(**kwargs))


def _datatree_rename(
    tree: xarray.DataTree, renames: dict[str, str]
) -> xarray.DataTree:
  """Rename variables and coordinates in a DataTree."""
  return tree.map_over_datasets(
      lambda ds: ds.rename({k: v for k, v in renames.items() if k in ds})
  )


def _coordinate_to_root(tree: xarray.DataTree, name: str) -> xarray.DataTree:
  """Move a coordinate to the root of a DataTree."""
  tree = tree.copy()
  for node in tree.subtree:
    if name in node.coords:
      coord = node.coords[name]
      del node.coords[name]
      if name in tree.coords:
        if not tree.coords[name].equals(coord):
          raise ValueError(
              f'Coordinate {name=} {tree.coords[name]=} != {coord=}'
          )
      else:
        tree.coords[name] = coord
  return tree


def device_put_to_cpu(x: typing.Pytree) -> typing.Pytree:
  """Non-blocking transfer of JAX arrays to CPU."""
  cpu_device = jax.devices('cpu')[0]
  return jax.device_put(x, cpu_device)


class BadStateError(Exception):
  """Error raised when an invalid state value (i.e., NaN) is encountered."""


@jax.jit
def _any_nans(x: typing.Pytree) -> jax.Array:
  return jnp.any(jnp.array([jnp.isnan(x).any() for x in jax.tree.leaves(x)]))


def check_pytree_for_bad_state(state: typing.Pytree, msg: str) -> None:
  """Check a pytree for bad state (i.e., NaN).

  This computation is always performed on the CPU, to avoid blocking computation
  on accelerators.

  Args:
    state: pytree to check.
    msg: message to include in the exception if a bad state is found.

  Raises:
    BadStateError: if a bad state is found.
  """
  if _any_nans(device_put_to_cpu(state)):
    raise BadStateError(f'bad state found: {msg}')


T = TypeVar('T')


def _tmp_file_path(path: str) -> epath.Path:
  return epath.Path(path).with_suffix(f'.{uuid.uuid4().hex}.tmp')


@contextlib.contextmanager
def _atomic_write(path):
  tmp_path = _tmp_file_path(path)
  try:
    with tmp_path.open('wb') as f:
      yield f
    tmp_path.replace(path)
  except Exception:  # pylint: disable=broad-except
    tmp_path.unlink(missing_ok=True)
    raise


@dataclasses.dataclass
class InferenceRunner:
  """High performance inference runner for NeuralGCM models."""

  model: api.InferenceModel | None
  inputs: NestedData
  dynamic_inputs: dynamic_inputs_lib.DynamicInputs
  init_times: npt.NDArray[np.datetime64] | pd.DatetimeIndex
  ensemble_size: int | None  # ensemble_size=None for deterministic models
  ensemble_batch_size: int = dataclasses.field(default=1, kw_only=True)
  output_path: str
  output_query: typing.Queries
  dynamic_query_inputs: dynamic_inputs_lib.DynamicInputs | None = (
      dataclasses.field(default=None, kw_only=True)
  )
  output_freq: np.timedelta64
  output_duration: np.timedelta64
  unroll_duration: np.timedelta64
  write_duration: np.timedelta64
  checkpoint_duration: np.timedelta64
  zarr_attrs: dict[str, Any] = dataclasses.field(default_factory=dict)
  zarr_chunks: dict[str, int] = dataclasses.field(default_factory=dict)
  zarr_shards: dict[str, int] | None = None
  bad_state_strategy: str = 'raise'
  max_nan_retries: int = 3  # only used when bad_state_strategy='retry'.
  check_dynamic_inputs_data: bool = False
  random_seed: int = 0
  _dynamic_queries_read_spec: dict[str, dict[str, cx.Coordinate]] = (
      dataclasses.field(default_factory=dict, init=False)
  )

  def __post_init__(self):
    if self.ensemble_size is not None and self.ensemble_batch_size > 1:
      if self.ensemble_size % self.ensemble_batch_size != 0:
        raise ValueError(
            f'ensemble_size ({self.ensemble_size}) must be divisible by '
            f'ensemble_batch_size ({self.ensemble_batch_size})'
        )
    if self.output_duration % self.output_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.output_duration=} must be a multiple of {self.output_freq=}'
      )
    if self.unroll_duration % self.output_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.unroll_duration=} must be a multiple of {self.output_freq=}'
      )
    if self.write_duration % self.output_freq != np.timedelta64(0):
      raise ValueError(
          f'{self.write_duration=} must be a multiple of {self.output_freq=}'
      )
    if self.max_steps_per_write % self.steps_per_unroll != 0:
      raise ValueError(
          f'write_duration in steps ({self.max_steps_per_write}) must be a '
          f'multiple of unroll_duration in steps ({self.steps_per_unroll})'
      )
    if self.max_steps_per_write % self.zarr_chunks['lead_time'] != 0:
      raise ValueError(
          f'write_duration in steps ({self.max_steps_per_write}) must be a'
          " multiple of zarr_chunks['lead_time']"
          f" ({self.zarr_chunks['lead_time']})"
      )
    if self.checkpoint_duration % self.write_duration != np.timedelta64(0):
      raise ValueError(
          f'{self.checkpoint_duration=} must be a multiple of'
          f' {self.write_duration=}'
      )
    if self.bad_state_strategy not in (
        'raise', 'impute_nan', 'ignore', 'retry',
    ):
      raise ValueError(f'Unknown bad_state_strategy: {self.bad_state_strategy}')
    if self.bad_state_strategy == 'retry' and self.ensemble_size is None:
      raise ValueError(
          "bad_state_strategy='retry' requires ensemble_size to be set, "
          'because deterministic models have no RNG to vary across retries.'
      )
    if isinstance(self.init_times, pd.DatetimeIndex):
      self.init_times = self.init_times.to_numpy()
    for key, dataset in self.inputs.items():
      time_index: pd.Index = dataset.indexes['time']
      if (time_index.get_indexer(self.init_times) == -1).any():
        raise ValueError(
            f'inputs {key!r} does not contain all init_times:\n'
            f'inputs={self.inputs}\ninit_times={self.init_times}'
        )

    # Pre-compute specs for reading dynamic queries from the data.
    dynamic_queries_read_spec = {}
    for ds_key, sub_query in self.output_query.items():
      ds_specs = {}
      for var_name, spec in sub_query.items():
        # we only need read specs for FieldInQuery.
        if isinstance(spec, data_specs.FieldInQuerySpec):
          var_spec = spec.spec
          # Set any timedelta to enable reading arbitrary lead times.
          if isinstance(var_spec, cx.Coordinate):
            var_spec = data_specs.CoordSpec.with_any_timedelta(var_spec)
          elif isinstance(var_spec, data_specs.CoordSpec):
            var_spec = data_specs.CoordSpec.with_any_timedelta(
                var_spec.coord, var_spec.dim_match_rules
            )
          else:
            raise ValueError(
                f'Unsupported spec type for dynamic query {ds_key} {var_name}:'
                f' {type(var_spec)}'
            )
          ds_specs[var_name] = var_spec
      if ds_specs:
        dynamic_queries_read_spec[ds_key] = ds_specs
    self._dynamic_queries_read_spec = dynamic_queries_read_spec

  @property
  def total_steps(self) -> int:
    return math.ceil(self.output_duration / self.output_freq)

  @property
  def steps_per_unroll(self) -> int:
    return self.unroll_duration // self.output_freq  # pyrefly: ignore[bad-return]

  @property
  def max_steps_per_write(self) -> int:
    return self.write_duration // self.output_freq  # pyrefly: ignore[bad-return]

  @property
  def steps_per_checkpoint(self) -> int:
    return self.checkpoint_duration // self.output_freq  # pyrefly: ignore[bad-return]

  def _checkpoints_path(self) -> epath.Path:
    # names beginning with __ are reserved for Zarr v3 extensions:
    # https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#node-names
    return epath.Path(self.output_path) / '__inference_checkpoints'

  def _get_array_encoding(
      self, array: xarray.DataArray
  ) -> dict[str, tuple[int, ...]]:
    """Return encoding for a DataArray."""
    chunks = tuple(
        self.zarr_chunks.get(dim, size) for dim, size in array.sizes.items()  # pyrefly: ignore[no-matching-overload]
    )
    # TODO(shoyer): Remove fill_value after Xarray defaults to NaN:
    # https://github.com/pydata/xarray/pull/10757
    encoding = {'chunks': chunks, 'fill_value': np.nan}
    zarr_shards = self.zarr_shards
    if zarr_shards is not None:
      encoding['shards'] = tuple(
          zarr_shards.get(dim, chunksize)  # pyrefly: ignore[no-matching-overload]
          for dim, chunksize in zip(array.dims, chunks)
      )
    return encoding  # pyrefly: ignore[bad-return]

  def setup(self) -> None:
    """Call once to setup metadata in output Zarr(s)."""
    checkpoints_path = self._checkpoints_path()
    if checkpoints_path.exists():
      logging.info('InferenceRunner.setup() already called, skipping')
      return

    # Our strategy here is to create a DataTree where all variables are empty
    # dask.array objects, similar xarray_beam.make_template()
    sample_queries = self.output_query
    if self.dynamic_query_inputs is not None:
      # If using streaming dynamic queries, we fetch metadata & finalize specs.
      first_time = self.init_times[0].astype('datetime64[ns]')
      q_stream = self.dynamic_query_inputs.get_forecast(first_time)
      q_data = q_stream.get_data(np.timedelta64(0), self.output_freq)
      q_data = {k: v.isel(time=0, drop=True) for k, v in q_data.items()}
      c_types = data_specs.get_nested_coord_types(
          self._dynamic_queries_read_spec
      )
      coord_sources = xarray_utils.extract_xr_source_coords(q_data, c_types)
      sample_queries = data_specs.finalize_nested_queries_spec(
          self.output_query, coord_sources
      )
    template = _queries_to_dummy_datatree(sample_queries)
    lead_times = np.arange(0, self.output_duration, self.output_freq)
    expanded_dims = {}
    if self.ensemble_size is not None:
      expanded_dims['realization'] = np.arange(self.ensemble_size)
    expanded_dims['init_time'] = self.init_times.astype('datetime64[ns]')
    expanded_dims['lead_time'] = lead_times.astype('timedelta64[ns]')

    # The use of expand_dims() here replicates
    # xarray_beam.replace_template_dims().
    template = _datatree_expand_dims(template, **expanded_dims)
    template.attrs.update(self.zarr_attrs)

    # TODO(shoyer): Determine if there's anything we need to do to work around
    # idempotency issues in Zarr, or if using Zarr v3 is sufficient:
    # https://github.com/zarr-developers/zarr-python/issues/1435

    encoding = {}
    for node in template.subtree:
      encoding[node.path] = {
          k: self._get_array_encoding(v) for k, v in node.data_vars.items()
      }

    logging.info(
        f'writing template to {self.output_path}:\n{encoding=}\n{template}'
    )
    template.to_zarr(
        self.output_path,
        compute=False,
        encoding=encoding,
        mode='w',  # override if exists, to ensure robustness to preemptions
        zarr_format=3,
        consolidated=False,
    )

    # Add directory for inference checkpoints. The existance of this directory
    # is used as a marker to indicate that setup() has been called.
    checkpoints_path.mkdir(exist_ok=True, mode=0o775)
    logging.info('InferenceRunner.setup() complete')

  def _decompose_task_id(
      self, task_id: int
  ) -> tuple[int, int | None, int | None]:
    """Decompose a task_id to init_time_index and realization range."""
    if self.ensemble_size is None:
      return task_id, None, None
    ens_total, ens_batch = self.ensemble_size, self.ensemble_batch_size
    ensemble_batches = math.ceil(ens_total / ens_batch)
    init_time_index = task_id // ensemble_batches
    batch_index = task_id % ensemble_batches
    realization_start = batch_index * ens_batch
    realization_end = realization_start + ens_batch
    return (
        init_time_index,
        realization_start,
        realization_end,
    )

  @property
  def task_count(self) -> int:
    """Total number of simulation tasks."""
    if self.ensemble_size is None:
      return len(self.init_times)
    ens_total, ens_batch = self.ensemble_size, self.ensemble_batch_size
    ensemble_batches = math.ceil(ens_total / ens_batch)
    return len(self.init_times) * ensemble_batches

  def _task_path(self, task_id: int) -> epath.Path:
    return self._checkpoints_path() / f'task_{task_id}.pkl'

  def _make_rng(
      self,
      task_id: int,
      init_time_index: int,
      realization_start: int | None,
      attempt: int = 0,
  ) -> cx.Field | None:
    """Construct RNG for a given task, with optional retry offset.

    On retry attempts, the RNG seed is offset by `attempt * task_count` to
    guarantee a unique but deterministic key for each retry while avoiding
    collisions with other task_ids.

    Args:
      task_id: integer task ID.
      init_time_index: index into init_times for this task.
      realization_start: starting realization index, or None for deterministic.
      attempt: retry attempt number (0 for the first try).

    Returns:
      A Field wrapping the JAX PRNGKey(s), or None for deterministic models.
    """
    if self.ensemble_size is None:
      return None
    base_key = jax.random.key(self.random_seed)
    total_realizations = len(self.init_times) * self.ensemble_size
    retry_offset = attempt * total_realizations
    if self.ensemble_batch_size == 1:
      return cx.field(jax.random.fold_in(base_key, task_id + retry_offset))
    # Use individual task_ids for each realization in the batch.
    # This matches the RNG sequence of the unbatched execution setup.
    task_ids = (
        realization_start  # pyrefly: ignore[unsupported-operation]
        + np.arange(self.ensemble_batch_size)
        + init_time_index * self.ensemble_size
        + retry_offset
    )
    keys = jax.vmap(jax.random.fold_in, (None, 0))(base_key, task_ids)
    return cx.field(keys)

  def run(self, task_id: int) -> None:
    """Simulate a single task, saving outputs to the Zarr file.

    Each task corresponds to an initial condition and ensemble RNG.

    Args:
      task_id: integer task ID to run, from the range [0, task_count).

    Raises:
      SomeException: if setup() has not been completed first.
    """
    if self.model is None:
      raise ValueError('model must be set before calling InferenceRunner.run()')

    max_attempts = (
        self.max_nan_retries + 1
        if self.bad_state_strategy == 'retry'
        else 1
    )
    for attempt in range(max_attempts):
      try:
        self._run_attempt(task_id, attempt=attempt)
        return  # success
      except BadStateError as error:
        if self.bad_state_strategy == 'raise':
          error.add_note(f'{task_id=} failed')
          raise
        elif self.bad_state_strategy == 'retry':
          # Clean up any partial checkpoint before retrying.
          self._task_path(task_id).unlink(missing_ok=True)
          if attempt + 1 < max_attempts:
            logging.warning(
                f'Bad state on {task_id=} (attempt {attempt + 1}'
                f'/{max_attempts}), retrying with new RNG: {error}'
            )
          else:
            logging.warning(
                f'Bad state on {task_id=}, all {max_attempts} attempts'
                f' exhausted. Imputing NaN for remainder: {error}'
            )
        else:
          # 'impute_nan': stop and leave remainder as NaN.
          logging.warning(
              f'Bad state encountered, imputing NaN for remainder of'
              f' {task_id=}: {error}'
          )
          return

  def _run_attempt(self, task_id: int, *, attempt: int = 0) -> None:
    """Execute a single simulation attempt for a task.

    Args:
      task_id: integer task ID to run.
      attempt: retry attempt number (0 for the first try). Used to offset the
        RNG seed so each retry uses a different random trajectory.

    Raises:
      BadStateError: if NaN is detected in state or outputs.
    """
    assert self.model is not None  # checked in run()
    init_time_index, realization_start, _ = (
        self._decompose_task_id(task_id)
    )

    init_time = self.init_times[init_time_index].astype('datetime64[ns]')
    dynamic_inputs_forecast = self.dynamic_inputs.get_forecast(init_time)
    dynamic_queries_stream = (
        self.dynamic_query_inputs.get_forecast(init_time)
        if self.dynamic_query_inputs is not None
        else None
    )

    # Initialize state from checkpoint or from scratch.
    task_path = self._task_path(task_id)
    if task_path.exists():
      logging.info(f'state checkpoint already exists for task {task_id=}')
      with task_path.open('rb') as f:
        state, commited_steps = pickle.load(f)
    else:
      logging.info(f'no state checkpoint found for task {task_id=}')
      rng = self._make_rng(task_id, init_time_index, realization_start, attempt)

      # TODO(shoyer): Can we get the number of time-steps to include in model
      # inputs from the model (or at least an InferenceRunner property) instead
      # of hard-coding it here?
      selected_inputs = {
          k: v.sel(time=[init_time]) for k, v in self.inputs.items()
      }
      input_obs = xarray_utils.model_inputs_from_xarray(
          selected_inputs, self.model
      )
      dynamic_inputs = xarray_utils.model_dynamic_inputs_from_xarray(
          dynamic_inputs_forecast.get_data(
              np.timedelta64(0), self.model.timestep  # pyrefly: ignore[bad-argument-type]
          ),
          model=self.model,
      )
      logging.info('assimilating initial state')

      if self.ensemble_size is not None and self.ensemble_batch_size > 1:
        vmap_assimilate = jax.vmap(
            self.model.assimilate, in_axes=(None, None, 0)
        )
        state = vmap_assimilate(input_obs, dynamic_inputs, rng)
      else:
        state = self.model.assimilate(input_obs, dynamic_inputs, rng)

      if self.bad_state_strategy != 'ignore':
        check_pytree_for_bad_state(state, f'initial state for {task_id=}')
      commited_steps = 0

    # Unroll simulation forward in time.

    def get_dynamic_inputs_and_queries(output_step):
      lead_start = output_step * self.output_freq
      lead_stop = (output_step + self.steps_per_unroll) * self.output_freq
      logging.info(f'getting dynamic inputs for {output_step=}')
      xarray_inputs = dynamic_inputs_forecast.get_data(lead_start, lead_stop)
      logging.info(f'xarray_inputs: {xarray_inputs}')
      data = xarray_utils.model_dynamic_inputs_from_xarray(
          xarray_inputs, self.model
      )
      if (
          self.check_dynamic_inputs_data
          and self.bad_state_strategy != 'ignore'
      ):
        check_pytree_for_bad_state(data, f'dynamic inputs at {output_step=}')
      keys = {k: list(v) for k, v in data.items()}
      logging.info(f'dynamic inputs ready: {keys}')

      query_fields = {}
      if dynamic_queries_stream is not None:
        queries_data = dynamic_queries_stream.get_data(
            lead_start, lead_stop + self.output_freq
        )
        queries_data = xarray_utils.ensure_timedelta_axis(queries_data)
        query_fields = xarray_utils.read_from_xarray(
            queries_data, self._dynamic_queries_read_spec, strict_matches=False  # pyrefly: ignore[bad-argument-type]
        )
      queries = data_specs.construct_query(query_fields, self.output_query)  # pyrefly: ignore[bad-argument-type]
      return data, queries

    @timing.Timed
    def unroll(state, dynamic_inputs, query_slice):
      unroll_fn = functools.partial(
          api.unroll_from_advance,
          timedelta=self.output_freq,
          steps=self.steps_per_unroll,
          queries=query_slice,
          dynamic_inputs=dynamic_inputs,
          prepend_init=True,
          trim_last=True,
      )
      if self.ensemble_size is not None and self.ensemble_batch_size > 1:
        unroll_fn = jax.vmap(unroll_fn, in_axes=(None, 0))
      return unroll_fn(self.model, state)

    dynamic_inputs_task = streaming.SingleTaskExecutor(
        get_dynamic_inputs_and_queries
    )
    commit_chunk_task = streaming.SingleTaskExecutor(self._commit_chunk)

    dynamic_inputs_task.submit(commited_steps)
    write_step_timer = timing.Timer()

    for steps_written in range(
        commited_steps, self.total_steps, self.max_steps_per_write
    ):
      write_step_timer.begin_step()
      output_buffer = []
      chunk_end_step = min(
          steps_written + self.max_steps_per_write, self.total_steps
      )

      for step_start in range(
          steps_written, chunk_end_step, self.steps_per_unroll
      ):
        dynamic_inputs, query_slice = dynamic_inputs_task.get()
        if step_start + self.steps_per_unroll < self.total_steps:
          dynamic_inputs_task.submit(step_start + self.steps_per_unroll)

        logging.info(f'{dynamic_inputs=}')
        state, trajectory_slice = unroll(state, dynamic_inputs, query_slice)
        output_buffer.append(device_put_to_cpu(trajectory_slice))

      commit_chunk_task.wait()
      commit_chunk_task.submit(state, output_buffer, task_id, steps_written)
      write_step_timer.finish_step()

      logging.info(
          f'write step took {write_step_timer.last:.3g}s '
          f'({unroll.timer.total:.3g}s for compute, '
          f'{dynamic_inputs_task.timer.total:.3g}s for dynamic inputs, '
          f'{commit_chunk_task.timer.last:.3g}s for commit chunk)'
      )
      dynamic_inputs_task.timer.reset_total()
      unroll.timer.reset_total()

    commit_chunk_task.wait()

  def _commit_chunk(
      self,
      state: typing.Pytree,
      output_buffer: list[typing.Pytree],
      task_id: int,
      steps_written: int,
  ) -> None:
    """Write outputs to disk, checking state if necessary."""
    logging.info(f'committing chunk for {task_id=} and {steps_written=}')
    self._write_zarr_chunk(output_buffer, task_id, steps_written)
    self._maybe_update_state_checkpoint(state, task_id, steps_written)
    logging.info('commit chunk complete')

  def _write_zarr_chunk(
      self, output_buffer: list[typing.Pytree], task_id: int, steps_written: int
  ) -> None:
    """Write Zarr chunk to disk."""
    if self.model is None:
      raise ValueError('model must be set before calling InferenceRunner.run()')
    init_time_index, realization_start, realization_end = (
        self._decompose_task_id(task_id)
    )

    logging.info('preparing DataTree for writing to zarr')
    trees = []
    for trajectory_slice in output_buffer:
      if self.ensemble_size is not None and self.ensemble_batch_size > 1:
        r_axis = cx.LabeledAxis(
            'realization', np.arange(realization_start, realization_end)  # pyrefly: ignore[no-matching-overload]
        )
        trajectory_slice = cx.tag(trajectory_slice, r_axis)

      outputs = {
          k: xarray_utils.fields_to_xarray(ds)
          for k, ds in trajectory_slice.items()
      }
      outputs_tree = xarray.DataTree.from_dict(outputs)

      outputs_tree = _datatree_rename(outputs_tree, {'timedelta': 'lead_time'})
      trees.append(outputs_tree)

    tree = xarray.map_over_datasets(  # pyrefly: ignore[no-matching-overload]
        lambda *xs: xarray.concat(xs, dim='lead_time'), *trees
    )
    tree = _coordinate_to_root(tree, 'lead_time')
    if self.ensemble_size is not None and self.ensemble_batch_size > 1:
      tree = _coordinate_to_root(tree, 'realization')
    num_steps_in_chunk = tree.sizes['lead_time']

    init_time = self.init_times[init_time_index].astype('datetime64[ns]')
    tree = _datatree_expand_dims(tree, init_time=[init_time])

    region = {'init_time': slice(init_time_index, init_time_index + 1)}
    region['lead_time'] = slice(
        steps_written, steps_written + num_steps_in_chunk
    )
    if realization_start is not None:
      if self.ensemble_batch_size == 1:
        tree = _datatree_expand_dims(tree, realization=[realization_start])
      region['realization'] = slice(realization_start, realization_end)  # pyrefly: ignore[bad-assignment]
    # Remove variables that don't have an init_time dimension. These shouldn't
    # be written to disk again.
    for node in tree.subtree:
      for name, variable in node.variables.items():
        if not any(dim in variable.dims for dim in region):
          del node[name]  # pyrefly: ignore[unsupported-operation]

    logging.info('setting up zarr outputs')
    delayed = tree.chunk().to_zarr(
        self.output_path, mode='r+', region=region, compute=False,
        consolidated=False,
    )

    logging.info(f'writing {tree.nbytes/1e6:.1f} MB to zarr')
    delayed.compute(num_workers=128)
    if self.bad_state_strategy != 'ignore':
      check_pytree_for_bad_state(
          output_buffer, f'unroll outputs at {steps_written=}'
      )

  def _maybe_update_state_checkpoint(
      self, state: typing.Pytree, task_id: int, steps_written: int
  ) -> None:
    """Update the state checkpoint, if necessary."""
    if steps_written + self.max_steps_per_write >= self.total_steps:
      logging.info('task complete, removing temporary state checkpoint')
      self._task_path(task_id).unlink(missing_ok=True)
    elif steps_written % self.steps_per_checkpoint == 0:
      logging.info('writing state checkpoint')
      task_path = self._task_path(task_id)
      state = device_put_to_cpu(state)
      saved_step = steps_written + self.max_steps_per_write
      with _atomic_write(task_path) as f:
        contents = (state, saved_step)
        pickle.dump(contents, f, protocol=pickle.HIGHEST_PROTOCOL)
      # Check after dumping to pickle file to aid in debugging.
      if self.bad_state_strategy != 'ignore':
        check_pytree_for_bad_state(state, f'unroll state at {saved_step=}')
    else:
      logging.info('no state checkpoint update for this step')
