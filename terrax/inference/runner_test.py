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

import functools
import operator
import os
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from fiddle.experimental import auto_config
from flax import nnx
import jax
import numpy as np
from terrax.core import api
from terrax.core import data_specs
from terrax.core import dynamic_io
from terrax.core import random_processes
from terrax.core import typing
from terrax.inference import dynamic_inputs as dynamic_inputs_lib
from terrax.inference import runner as runnerlib
import xarray


@nnx.dataclass
class MockModel(api.Model):
  """A mock Model for testing."""

  input_specs: dict[str, dict[str, cx.Coordinate]]
  dynamic_input_specs: dict[str, dict[str, cx.Coordinate]]
  dynamic_input_slice: dynamic_io.DynamicInputSlice = nnx.data()
  assimilation_noise: random_processes.RandomProcessModule | None = nnx.data()

  def __post_init__(self):
    self.prognostics = typing.Prognostic({
        k: cx.field(np.zeros(v.coord.shape), v.coord)
        for k, v in self.inputs_spec['state'].items()
    })

  @property
  def inputs_spec(
      self,
  ) -> dict[str, dict[str, data_specs.CoordSpec]]:
    make_spec = data_specs.CoordSpec.with_any_timedelta
    return jax.tree.map(make_spec, self.input_specs, is_leaf=cx.is_coord)

  @property
  def dynamic_inputs_spec(
      self,
  ) -> dict[str, dict[str, cx.Coordinate]]:
    make_spec = data_specs.CoordSpec.with_any_timedelta
    return jax.tree.map(
        make_spec, self.dynamic_input_specs, is_leaf=cx.is_coord
    )

  @property
  def timestep(self) -> np.timedelta64:
    return np.timedelta64(1, 'h')

  def assimilate(self, observations: typing.Observation) -> None:
    state = jax.tree.map(
        # TODO(shoyer): create a .isel() method on Field?
        cx.cmap(lambda x: x[-1]),
        cx.untag(observations['state'], 'timedelta'),
        is_leaf=cx.is_field,
    )
    if self.assimilation_noise is not None:
      noise = self.assimilation_noise.state_values()
      state['foo'] = state['foo'] + noise
    self.prognostics.set_value(state)

  def advance(self) -> None:
    prognostics = self.prognostics.get_value()
    time = prognostics.pop('time')
    sliced_inputs = self.dynamic_input_slice(time)
    next_prognostics = jax.tree.map(
        operator.add, prognostics, sliced_inputs, is_leaf=cx.is_field
    )
    next_prognostics['time'] = time + self.timestep
    self.prognostics.set_value(next_prognostics)

  def observe(self, queries: typing.Queries) -> typing.Observation:
    prognostics = self.prognostics.get_value()
    return {
        ds_key: {
            k: v if cx.is_field(v) else prognostics[k]
            for k, v in sub_query.items()
        }
        for ds_key, sub_query in queries.items()
    }


@auto_config.auto_config
def construct_model(
    assimilation_noise: random_processes.RandomProcessModule | None = None,
) -> MockModel:
  """Constructs a MockModel advancing 'foo' and 'bar' by fixed increments."""
  return MockModel(
      input_specs={
          'state': {
              'foo': cx.Scalar(),
              'bar': cx.LabeledAxis('x', np.array([0.1, 0.2, 0.3])),
              'time': cx.Scalar(),
          }
      },
      dynamic_input_specs={
          'data': {
              'foo': cx.Scalar(),
              'bar': cx.Scalar(),
              'time': cx.Scalar(),
          }
      },
      dynamic_input_slice=dynamic_io.DynamicInputSlice(
          keys_to_coords={'foo': cx.Scalar(), 'bar': cx.Scalar()},
          observation_key='data',
      ),
      assimilation_noise=assimilation_noise,
  )


class RunnerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='deterministic',
          ensemble_size=None,
          ensemble_batch_size=1,
      ),
      dict(testcase_name='ensemble', ensemble_size=2, ensemble_batch_size=1),
      dict(
          testcase_name='ensemble_batched_exact',
          ensemble_size=4,
          ensemble_batch_size=2,
      ),
      dict(
          testcase_name='static_field_query',
          ensemble_size=None,
          ensemble_batch_size=1,
          field_in_queries_coord=cx.LabeledAxis('loc', np.array([1, 2])),
          use_dynamic_query_inputs=False,
      ),
      dict(
          testcase_name='dynamic_field_query',
          ensemble_size=None,
          ensemble_batch_size=1,
          field_in_queries_coord=cx.LabeledAxis('loc', np.array([1, 2])),
          use_dynamic_query_inputs=True,
      ),
  )
  def test_inference_runner(
      self,
      ensemble_size,
      ensemble_batch_size,
      field_in_queries_coord: cx.Coordinate | None = None,
      use_dynamic_query_inputs: bool = False,
  ):
    if 'GITHUB_ACTIONS' in os.environ:
      self.skipTest(
          'TODO(dkochkov): Fiddle on PyPI lacks support for bound method '
          'serialization.'
      )
    if ensemble_size is not None:
      assimilation_noise = random_processes.UniformUncorrelated.construct(
          minval=-0.1, maxval=0.1, coord=cx.Scalar(), rngs=nnx.Rngs(0)
      )
    else:
      assimilation_noise = None

    module_model = construct_model(assimilation_noise)
    model = api.InferenceModel.from_model_api(
        module_model, construct_model.as_buildable(assimilation_noise)
    )
    init_times = np.array(
        [np.datetime64('2025-01-01'), np.datetime64('2025-01-02')]
    )
    out_freq_in_h = 6
    out_duration_in_h = 48
    output_freq = np.timedelta64(out_freq_in_h, 'h')
    output_duration = np.timedelta64(out_duration_in_h, 'h')
    one_h = np.timedelta64(1, 'h')
    inputs = {
        'state': xarray.Dataset(
            {
                'foo': (('time',), np.array([0.0, 10.0])),
                'bar': (('time', 'x'), np.array(2 * [[1.0, 2.0, 3.0]])),
            },
            coords={'time': init_times, 'x': np.array([0.1, 0.2, 0.3])},
        )
    }
    delta = xarray.Dataset({'foo': 1.0, 'bar': 2.0})
    dynamic_inputs = dynamic_inputs_lib.Persistence(
        full_data={'data': delta.expand_dims(time=init_times)},
        climatology=None,
        update_freq=np.timedelta64(6, 'h'),
    )
    output_path = self.create_tempdir().full_path
    output_query = {
        'state': {
            'foo': cx.Scalar(),
            'bar': cx.LabeledAxis('x', np.array([0.1, 0.2, 0.3])),
        }
    }
    query_ds = None  # make pytype happy.
    dynamic_query_inputs = None
    if field_in_queries_coord is not None:
      buz_spec = data_specs.FieldInQuerySpec(field_in_queries_coord)
      if use_dynamic_query_inputs:
        output_query['state']['buz'] = buz_spec
        lead_times = one_h * np.arange(
            0, out_duration_in_h + out_freq_in_h, out_freq_in_h
        )  # contains values for all lead times, including last (not written).
        all_times = np.unique(
            np.concatenate([t0 + lead_times for t0 in init_times])
        )
        hours = (all_times - all_times[0]) / one_h
        query_data = hours[:, None] * np.ones(buz_spec.spec.shape)
        query_ds = xarray.Dataset(
            {'buz': (('time',) + buz_spec.spec.dims, query_data)},
            coords={'time': all_times} | buz_spec.spec.to_xarray(),
        )
        dynamic_query_inputs = dynamic_inputs_lib.Prescribed(
            full_data={'state': query_ds},
            climatology=None,
            update_freq=output_freq,  # update for each output step.
        )
      else:
        output_query['state']['buz'] = cx.field(
            np.ones(buz_spec.spec.shape), buz_spec.spec
        )

    zarr_chunks = {'lead_time': 4, 'init_time': 1}
    if ensemble_size is not None:
      zarr_chunks['realization'] = 1
    runner = runnerlib.InferenceRunner(
        model=model,
        inputs=inputs,
        dynamic_inputs=dynamic_inputs,
        dynamic_query_inputs=dynamic_query_inputs,
        init_times=init_times,
        ensemble_size=ensemble_size,
        ensemble_batch_size=ensemble_batch_size,
        output_path=output_path,
        output_query=output_query,
        output_freq=output_freq,
        output_duration=output_duration,
        zarr_chunks=zarr_chunks,
        write_duration=np.timedelta64(24, 'h'),
        unroll_duration=np.timedelta64(12, 'h'),
        checkpoint_duration=np.timedelta64(24, 'h'),
    )
    runner.setup()

    if ensemble_size is None:
      expected_task_count = len(init_times)
    else:
      expected_task_count = len(init_times) * (
          ensemble_size // ensemble_batch_size
      )
    self.assertEqual(runner.task_count, expected_task_count)

    expected_lead_times = np.arange(0, out_duration_in_h, out_freq_in_h) * one_h
    nans = functools.partial(np.full, fill_value=np.nan)
    # `lead_time` lives on each output group, since groups may be written at
    # different output frequencies.
    coords = {'init_time': init_times.astype('datetime64[ns]')}
    dims = ('init_time', 'lead_time')
    shape = (len(init_times), len(expected_lead_times))
    if ensemble_size is not None:
      coords['realization'] = np.arange(ensemble_size)
      dims = ('realization',) + dims
      shape = (ensemble_size,) + shape

    child_coords = {
        'x': np.array([0.1, 0.2, 0.3]),
        'lead_time': expected_lead_times.astype('timedelta64[ns]'),
    }
    child_node_dict = {
        'foo': (dims, nans(shape)),
        'bar': (dims + ('x',), nans(shape + (3,))),
    }
    if field_in_queries_coord is not None:
      buz_dims = dims + field_in_queries_coord.dims
      buz_shape = shape + field_in_queries_coord.shape
      child_node_dict['buz'] = (buz_dims, nans(buz_shape))
      child_coords.update(field_in_queries_coord.to_xarray())

    root_node = xarray.Dataset(coords=coords)
    child_node = xarray.Dataset(child_node_dict, coords=child_coords)
    expected = xarray.DataTree.from_dict({'/': root_node, '/state': child_node})
    actual = xarray.open_datatree(output_path, engine='zarr')
    xarray.testing.assert_equal(actual, expected)

    for task_id in range(runner.task_count):
      runner.run(task_id)

    h = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0])
    expected_foo = np.stack([h, 10 + h])
    expected_bar = np.stack(
        2 * [np.stack([2 * h + 1, 2 * h + 2, 2 * h + 3], axis=1)]
    )
    if ensemble_size is not None:
      expected_foo = np.stack([expected_foo] * ensemble_size, axis=0)
      expected_bar = np.stack([expected_bar] * ensemble_size, axis=0)

    child_node_dict = {
        'foo': (dims, expected_foo),
        'bar': (dims + ('x',), expected_bar),
    }
    if field_in_queries_coord is not None:
      buz_dims = dims + field_in_queries_coord.dims
      buz_shape = shape + field_in_queries_coord.shape
      if use_dynamic_query_inputs:
        assert query_ds is not None
        expected_buz_list = [
            query_ds['buz'].sel(time=t + expected_lead_times).to_numpy()
            for t in init_times
        ]
        expected_buz = np.stack(expected_buz_list, axis=0)
        if ensemble_size is not None:
          expected_buz = np.stack([expected_buz] * ensemble_size, axis=0)
      else:
        expected_buz = np.ones(buz_shape)
      child_node_dict['buz'] = (buz_dims, expected_buz)
      child_coords.update(field_in_queries_coord.to_xarray())

    child_node = xarray.Dataset(child_node_dict, coords=child_coords)
    expected = xarray.DataTree.from_dict({'/': root_node, '/state': child_node})
    actual = xarray.open_datatree(output_path, engine='zarr')
    # round() removes initialization noise, which is between -0.1 and 0.1
    xarray.testing.assert_equal(actual.round(), expected)

    if ensemble_size is not None:
      # different ensemble members have different RNGs
      actual_foo = actual.state.foo.isel(init_time=0)
      first_realization = actual_foo.sel(realization=0)
      second_realization = actual_foo.sel(realization=1)
      self.assertFalse(
          np.allclose(first_realization, second_realization, atol=0.001),
          msg=f'{first_realization=}, {second_realization=}',
      )
      # different initialization also have different RNGs
      second_init = actual.state.foo.isel(init_time=1).sel(realization=0)
      self.assertFalse(
          np.allclose(first_realization, second_init, atol=0.001),
          msg=f'{first_realization=}, {second_init=}',
      )

    with self.assertRaisesRegex(ValueError, 'does not contain all init_times'):
      runnerlib.InferenceRunner(
          model=model,
          inputs=inputs,
          dynamic_inputs=dynamic_inputs,
          init_times=np.append(init_times, [np.datetime64('2025-01-03')]),
          ensemble_size=ensemble_size,
          output_path=output_path,
          output_query=output_query,
          output_freq=np.timedelta64(6, 'h'),
          output_duration=np.timedelta64(48, 'h'),
          zarr_chunks=zarr_chunks,
          write_duration=np.timedelta64(24, 'h'),
          unroll_duration=np.timedelta64(12, 'h'),
          checkpoint_duration=np.timedelta64(24, 'h'),
      )

  def test_inference_runner_with_output_freq_per_dataset_key(self):
    module_model = construct_model()
    model = api.InferenceModel.from_model_api(
        module_model, construct_model.as_buildable()
    )
    init_times = np.array(
        [np.datetime64('2025-01-01'), np.datetime64('2025-01-02')]
    )
    one_h = np.timedelta64(1, 'h')
    out_duration_in_h = 48
    inputs = {
        'state': xarray.Dataset(
            {
                'foo': (('time',), np.array([0.0, 10.0])),
                'bar': (('time', 'x'), np.array(2 * [[1.0, 2.0, 3.0]])),
            },
            coords={'time': init_times, 'x': np.array([0.1, 0.2, 0.3])},
        )
    }
    delta = xarray.Dataset({'foo': 1.0, 'bar': 2.0})
    dynamic_inputs = dynamic_inputs_lib.Persistence(
        full_data={'data': delta.expand_dims(time=init_times)},
        climatology=None,
        update_freq=np.timedelta64(6, 'h'),
    )
    output_path = self.create_tempdir().full_path
    runner = runnerlib.InferenceRunner(
        model=model,
        inputs=inputs,
        dynamic_inputs=dynamic_inputs,
        init_times=init_times,
        ensemble_size=None,
        output_path=output_path,
        output_query={
            'fast': {'foo': cx.Scalar()},
            'slow': {'bar': cx.LabeledAxis('x', np.array([0.1, 0.2, 0.3]))},
        },
        output_freq={
            'fast': np.timedelta64(6, 'h'),
            'slow': np.timedelta64(12, 'h'),
        },
        output_duration=np.timedelta64(out_duration_in_h, 'h'),
        zarr_chunks={'lead_time': 2, 'init_time': 1},
        write_duration=np.timedelta64(24, 'h'),
        unroll_duration=np.timedelta64(12, 'h'),
        checkpoint_duration=np.timedelta64(24, 'h'),
    )
    runner.setup()
    for task_id in range(runner.task_count):
      runner.run(task_id)

    actual = xarray.open_datatree(output_path, engine='zarr')
    fast = actual['fast'].to_dataset()
    slow = actual['slow'].to_dataset()

    # Each group is written on its own lead_time axis.
    fast_h = np.arange(0, out_duration_in_h, 6)
    slow_h = np.arange(0, out_duration_in_h, 12)
    np.testing.assert_array_equal(
        fast['lead_time'].to_numpy(),
        (fast_h * one_h).astype('timedelta64[ns]'),
    )
    np.testing.assert_array_equal(
        slow['lead_time'].to_numpy(),
        (slow_h * one_h).astype('timedelta64[ns]'),
    )

    # 'foo' grows by 1 per hour, starting from its value at each init_time.
    expected_foo = np.stack([fast_h, 10 + fast_h]).astype(float)
    np.testing.assert_allclose(fast['foo'].to_numpy(), expected_foo, rtol=1e-5)
    # 'bar' grows by 2 per hour, starting from [1, 2, 3] at each init_time.
    expected_bar = np.stack(
        2 * [np.stack([2 * slow_h + 1, 2 * slow_h + 2, 2 * slow_h + 3], -1)]
    ).astype(float)
    np.testing.assert_allclose(slow['bar'].to_numpy(), expected_bar, rtol=1e-5)


def make_runner(**kwargs) -> runnerlib.InferenceRunner:
  """Builds a valid InferenceRunner, overriding defaults with `kwargs`."""
  init_times = np.array([np.datetime64('2025-01-01')])
  inputs = {
      'state': xarray.Dataset(
          {'foo': (('time',), np.array([0.0]))}, coords={'time': init_times}
      )
  }
  dynamic_inputs = dynamic_inputs_lib.Persistence(
      full_data={
          'data': xarray.Dataset({'foo': 1.0}).expand_dims(time=init_times)
      },
      climatology=None,
      update_freq=np.timedelta64(6, 'h'),
  )
  defaults = dict(
      model=None,
      inputs=inputs,
      dynamic_inputs=dynamic_inputs,
      init_times=init_times,
      ensemble_size=None,
      output_path='',
      output_query={'a': {'foo': cx.Scalar()}, 'b': {'foo': cx.Scalar()}},
      output_freq=np.timedelta64(6, 'h'),
      output_duration=np.timedelta64(48, 'h'),
      unroll_duration=np.timedelta64(12, 'h'),
      write_duration=np.timedelta64(24, 'h'),
      checkpoint_duration=np.timedelta64(24, 'h'),
      zarr_chunks={'lead_time': 2, 'init_time': 1},
  )
  return runnerlib.InferenceRunner(**(defaults | kwargs))


class OutputFreqTest(parameterized.TestCase):

  def test_single_freq_applies_to_every_query_key(self):
    six_h = np.timedelta64(6, 'h')
    runner = make_runner(output_freq=six_h)
    self.assertEqual(runner.output_freqs, {'a': six_h, 'b': six_h})
    self.assertEqual(runner.unique_output_freqs, (six_h,))
    self.assertEqual(runner.finest_output_freq, six_h)
    self.assertEqual(runner.coarsest_output_freq, six_h)
    # A single frequency means the runner's steps and the outermost scan's
    # steps coincide.
    self.assertEqual(runner.steps_per_unroll, 2)
    self.assertEqual(runner.scan_steps_per_unroll, 2)

  def test_freq_per_key_is_ordered_finest_to_coarsest(self):
    six_h, twelve_h = np.timedelta64(6, 'h'), np.timedelta64(12, 'h')
    runner = make_runner(output_freq={'a': twelve_h, 'b': six_h})
    self.assertEqual(runner.output_freqs, {'a': twelve_h, 'b': six_h})
    self.assertEqual(runner.unique_output_freqs, (six_h, twelve_h))
    self.assertEqual(runner.finest_output_freq, six_h)
    self.assertEqual(runner.coarsest_output_freq, twelve_h)
    # Steps are counted at the finest frequency, while the outermost scan of
    # `api.unroll_from_advance` counts steps at the coarsest one.
    self.assertEqual(runner.steps_per_unroll, 2)
    self.assertEqual(runner.scan_steps_per_unroll, 1)

  def test_missing_query_key_raises(self):
    with self.assertRaisesRegex(ValueError, 'exactly the dataset keys'):
      make_runner(output_freq={'a': np.timedelta64(6, 'h')})

  def test_unknown_freq_key_raises(self):
    with self.assertRaisesRegex(ValueError, 'exactly the dataset keys'):
      make_runner(
          output_freq={
              'a': np.timedelta64(6, 'h'),
              'b': np.timedelta64(6, 'h'),
              'c': np.timedelta64(6, 'h'),
          }
      )

  def test_incongruent_freqs_raise(self):
    with self.assertRaisesRegex(ValueError, 'must be congruent'):
      make_runner(
          output_freq={
              'a': np.timedelta64(6, 'h'),
              'b': np.timedelta64(8, 'h'),
          }
      )

  def test_empty_output_query_raises(self):
    with self.assertRaisesRegex(ValueError, 'output_query must not be empty'):
      make_runner(output_query={})


class SplitQueriesByFrequencyTest(absltest.TestCase):

  def test_single_frequency_yields_one_group(self):
    six_h = np.timedelta64(6, 'h')
    queries = {'a': {'foo': cx.Scalar()}, 'b': {'bar': cx.Scalar()}}
    actual = runnerlib.split_queries_by_frequency(
        queries, {'a': six_h, 'b': six_h}, (six_h,)
    )
    self.assertEqual(actual, (queries,))

  def test_queries_are_grouped_by_frequency(self):
    six_h, twelve_h = np.timedelta64(6, 'h'), np.timedelta64(12, 'h')
    queries = {
        'a': {'foo': cx.Scalar()},
        'b': {'bar': cx.Scalar()},
        'c': {'baz': cx.Scalar()},
    }
    output_freqs = {'a': twelve_h, 'b': six_h, 'c': twelve_h}
    actual = runnerlib.split_queries_by_frequency(
        queries, output_freqs, (six_h, twelve_h)
    )
    expected = (
        {'b': queries['b']},
        {'a': queries['a'], 'c': queries['c']},
    )
    self.assertEqual(actual, expected)


class AtomicWriteTest(absltest.TestCase):

  def test_successful(self):
    path = pathlib.Path(self.create_tempdir()) / 'data'
    with runnerlib._atomic_write(path) as f:
      f.write(b'abc')
    self.assertTrue(path.exists())
    self.assertEqual(path.read_bytes(), b'abc')
    self.assertEqual(os.listdir(path.parent), ['data'])  # no temp files

  def test_incomplete(self):
    path = pathlib.Path(self.create_tempdir()) / 'data'
    with self.assertRaises(RuntimeError):
      with runnerlib._atomic_write(path) as f:
        f.write(b'a')
        raise RuntimeError
    self.assertFalse(path.exists())
    self.assertEqual(os.listdir(path.parent), [])  # no temp files


if __name__ == '__main__':
  absltest.main()
