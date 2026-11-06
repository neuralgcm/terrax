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

"""Tests that feature transforms produce outputs with expected structure."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax_datetime as jdt
import numpy as np
from terrax.core import coordinates
from terrax.core import dynamic_io
from terrax.core import feature_transforms
from terrax.core import observation_operators
from terrax.core import orographies
from terrax.core import pytree_utils
from terrax.core import random_processes
from terrax.core import spherical_harmonics
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units


class FeatureTransformsTest(parameterized.TestCase):
  """Tests feature transforms."""

  def _test_feature_module(
      self,
      feature_module: typing.Transform,
      inputs: typing.Pytree,
  ):
    features = feature_module(inputs)
    expected = pytree_utils.shape_structure(features)
    input_shapes = pytree_utils.shape_structure(inputs)
    actual = feature_module.output_shapes(input_shapes)
    chex.assert_trees_all_equal(actual, expected)

  @parameterized.named_parameters(
      dict(testcase_name='t21', grid=coordinates.LonLatGrid.T21()),
      dict(testcase_name='tl31', grid=coordinates.LonLatGrid.TL31()),
  )
  def test_radiation_features(self, grid):
    radiation_features = feature_transforms.RadiationFeatures(grid=grid)
    self._test_feature_module(
        radiation_features,
        {'time': cx.field(jdt.to_datetime('2025-01-09T15:00'))},
    )

  def test_latitude_features(self):
    grid = coordinates.LonLatGrid.T21()
    latitude_features = feature_transforms.LatitudeFeatures(grid=grid)
    self._test_feature_module(latitude_features, None)

  def test_orography_features(self):
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    orography = orographies.ModalOrography(
        ylm_map=ylm_map,
        rngs=None,
    )
    orography_features = feature_transforms.OrographyFeatures(
        orography_module=orography,
    )
    self._test_feature_module(orography_features, None)

  def test_orography_with_grads_features(self):
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    orography = orographies.ModalOrography(
        ylm_map=ylm_map,
        rngs=None,
    )
    orography_features = feature_transforms.OrographyWithGradsFeatures(
        orography_module=orography,
        compute_gradients_transform=transforms.ToModalWithDerivatives(
            ylm_map, attenuations=[2.0]
        ),
    )
    self._test_feature_module(orography_features, None)

  def test_observation_value_features(self):
    grid = coordinates.LonLatGrid.T21()
    levels = coordinates.PressureLevels([100, 500, 850])
    coord = cx.coords.compose(levels, grid)
    fields = {'temperature': cx.field(np.ones(coord.shape), coord)}
    operator = observation_operators.DataObservationOperator(fields)
    level_subset = coordinates.PressureLevels([500])
    query = {'temperature': cx.coords.compose(level_subset, grid)}
    observation_features = feature_transforms.ObservationValueFeatures(
        operator=operator, query=query
    )
    features = observation_features({})
    self.assertEqual(list(features.keys()), ['temperature'])
    self.assertEqual(
        features['temperature'].coordinate,
        cx.coords.compose(level_subset, grid),
    )

  def test_dynamic_input_features(self):
    grid = coordinates.LonLatGrid.T21()
    dynamic_input = dynamic_io.DynamicInputSlice(
        keys_to_coords={'a': grid, 'b': grid, 'c': grid},
        observation_key='abc',
    )
    expand_dims = lambda x: np.expand_dims(x, axis=(1, 2))
    data = {
        'abc': {
            'a': expand_dims(np.arange(2)) * np.ones(grid.shape),
            'b': expand_dims(np.arange(2)) * np.zeros(grid.shape),
            'c': expand_dims(np.arange(2)) * np.ones(grid.shape),
        }
    }
    timedelta = coordinates.TimeDelta(np.arange(2, dtype='timedelta64[h]'))
    grid_trajectory = cx.coords.compose(timedelta, grid)
    time = jdt.to_datetime('2000-01-01') + jdt.to_timedelta(
        12, 'h'
    ) * np.arange(timedelta.shape[0])
    in_data = jax.tree.map(lambda x: cx.field(x, grid_trajectory), data)
    in_data['abc']['time'] = cx.field(time, timedelta)
    dynamic_input.update_dynamic_inputs(in_data)
    with self.subTest('two_keys'):
      dynamic_input_features = feature_transforms.DynamicInputFeatures(
          ('a', 'b'), dynamic_input
      )
      self._test_feature_module(
          dynamic_input_features,
          {'time': cx.field(jdt.to_datetime('2000-01-01T06'))},
      )

    with self.subTest('with_time_offsets'):
      offsets = [np.timedelta64(0, 'h'), np.timedelta64(12, 'h')]
      window = cx.LabeledAxis('window', np.array([0, 12]))
      dynamic_input_features_with_offsets = (
          feature_transforms.DynamicInputFeatures(
              ('a', 'b'),
              dynamic_input,
              time_offsets=offsets,
              window_axis=window,
          )
      )
      self._test_feature_module(
          dynamic_input_features_with_offsets,
          {'time': cx.field(jdt.to_datetime('2000-01-01T00'))},
      )
      features = dynamic_input_features_with_offsets(
          {'time': cx.field(jdt.to_datetime('2000-01-01T00'))}
      )
      self.assertSameElements(['a', 'b'], list(features.keys()))
      init_time = cx.field(jdt.to_datetime('2000-01-01T00'))
      for offset in offsets:
        expected = dynamic_input(init_time + cx.field(jdt.to_timedelta(offset)))
        hours = int(offset / np.timedelta64(1, 'h'))
        for k in ('a', 'b'):
          actual = features[k].sel({'window': hours})
          cx.testing.assert_fields_allclose(
              actual, expected[k], rtol=1e-6, atol=1e-6
          )

    with self.subTest('missing_window_axis_raises'):
      with self.assertRaisesRegex(ValueError, '`window_axis` is required'):
        feature_transforms.DynamicInputFeatures(
            ('a',), dynamic_input, time_offsets=[np.timedelta64(0, 'h')]
        )({'time': cx.field(jdt.to_datetime('2000-01-01T00'))})

    with self.subTest('mismatched_window_axis_raises'):
      with self.assertRaisesRegex(ValueError, 'must be one dimensional'):
        feature_transforms.DynamicInputFeatures(
            ('a',),
            dynamic_input,
            time_offsets=[np.timedelta64(0, 'h')],
            window_axis=cx.SizedAxis('window', 2),
        )({'time': cx.field(jdt.to_datetime('2000-01-01T00'))})

  def test_dynamic_input_features_inder_jit(self):
    grid = coordinates.LonLatGrid.T21()
    dynamic_input = dynamic_io.DynamicInputSlice(
        keys_to_coords={'a': grid, 'b': grid, 'c': grid},
        observation_key='abc',
    )
    expand_dims = lambda x: np.expand_dims(x, axis=(1, 2))
    data = {
        'abc': {
            'a': expand_dims(np.arange(2)) * np.ones(grid.shape),
            'b': expand_dims(np.arange(2)) * np.zeros(grid.shape),
            'c': expand_dims(np.arange(2)) * np.ones(grid.shape),
        }
    }
    timedelta = coordinates.TimeDelta(np.arange(2, dtype='timedelta64[h]'))
    grid_trajectory = cx.coords.compose(timedelta, grid)
    time = jdt.to_datetime('2000-01-01') + jdt.to_timedelta(
        12, 'h'
    ) * np.arange(timedelta.shape[0])
    in_data = jax.tree.map(lambda x: cx.field(x, grid_trajectory), data)
    in_data['abc']['time'] = cx.field(time, timedelta)

    @nnx.jit
    def run(module, inputs, dynamic_inputs):
      module.dynamic_input_module.update_dynamic_inputs(dynamic_inputs)
      return module(inputs)

    dynamic_input_features = feature_transforms.DynamicInputFeatures(
        ('a', 'b'), dynamic_input
    )
    run(
        dynamic_input_features,
        {'time': cx.field(jdt.to_datetime('2000-01-01T06'))},
        in_data,
    )

  def test_coord_features(self):
    z = cx.SizedAxis('z', 8)
    grid = coordinates.LonLatGrid.T21()
    coords = {
        'surface_embedings': cx.coords.compose(z, grid),
        'land_sea_mask': grid,
    }
    coord_features = feature_transforms.ParamFeatures(coords, rngs=nnx.Rngs(1))
    self._test_feature_module(coord_features, None)

  def test_param_features_update_from_xarray_keeps_sharding(self):
    grid = coordinates.LonLatGrid.T21()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',)),
        jax.sharding.PartitionSpec(),
    )
    param_features = feature_transforms.ParamFeatures(
        {'land_sea_mask': grid}, rngs=nnx.Rngs(1)
    )
    feature = param_features.features['land_sea_mask']
    feature.set_value(jax.device_put(feature.get_value(), sharding))
    dataset = (
        cx.field(np.ones(grid.shape), grid)
        .to_xarray()
        .assign_attrs(units='dimensionless')
        .to_dataset(name='land_sea_mask')
    )

    param_features.update_from_xarray(dataset, sim_units=units.DEFAULT_UNITS)

    np.testing.assert_array_equal(feature.get_value(), np.ones(grid.shape))
    self.assertEqual(feature.get_value().sharding, sharding)

  @parameterized.named_parameters(
      dict(
          testcase_name='T21_grid',
          ylm_map=spherical_harmonics.FixedYlmMapping(
              lon_lat_grid=coordinates.LonLatGrid.T21(),
              ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
          ),
      ),
  )
  def test_randomness_features(self, ylm_map):
    with self.subTest('gaussian_random_field'):
      random_process = random_processes.GaussianRandomField.construct(
          ylm_map=ylm_map,
          dt=1.0,
          sim_units=units.DEFAULT_UNITS,
          correlation_time=1.0,
          correlation_length=1.0,
          variance=1.0,
          rngs=nnx.Rngs(0),
      )
      random_process.unconditional_sample(jax.random.key(0))
      randomness_features = feature_transforms.RandomnessFeatures(
          random_process=random_process,
          coord=ylm_map.lon_lat_grid,
      )
      self._test_feature_module(randomness_features, None)

    with self.subTest('batched_gaussian_random_fields'):
      random_process = random_processes.VectorizedGaussianRandomField.construct(
          ylm_map=ylm_map,
          dt=1.0,
          sim_units=units.DEFAULT_UNITS,
          axis=cx.SizedAxis('grf', 2),
          correlation_times=[1.0, 2.0],
          correlation_lengths=[0.6, 0.9],
          variances=[1.0, 1.0],
          rngs=nnx.Rngs(0),
      )
      random_process.unconditional_sample(jax.random.key(0))
      randomness_features = feature_transforms.RandomnessFeatures(
          random_process=random_process,
          coord=ylm_map.lon_lat_grid,
      )
      self._test_feature_module(randomness_features, None)

  def test_randomness_features_like_input(self):
    random_process = random_processes.NormalUncorrelated.construct(
        rngs=nnx.Rngs(0)
    )
    random_process.unconditional_sample(jax.random.key(0))
    sample_axis = cx.SizedAxis('noise', 4)
    randomness_features = feature_transforms.RandomnessFeatures(
        random_process=random_process,
        like='longitude',
        feature_name='point_randomness',
        sample_axis=sample_axis,
    )
    stations = cx.SizedAxis('station', 3)
    inputs = {'longitude': cx.field(np.array([1.0, 2.0, 3.0]), stations)}
    self._test_feature_module(randomness_features, inputs)
    draw = randomness_features(inputs)['point_randomness']
    self.assertEqual(draw.coordinate, cx.coords.compose(stations, sample_axis))
    # Draws are deterministic for a fixed state and change once advanced.
    same_draw = randomness_features(inputs)['point_randomness']
    np.testing.assert_array_equal(draw.data, same_draw.data)
    random_process.advance()
    advanced_draw = randomness_features(inputs)['point_randomness']
    self.assertFalse(np.allclose(draw.data, advanced_draw.data))
    # The same transform serves templates of a different size.
    more_stations = cx.SizedAxis('station', 5)
    more_inputs = {'longitude': cx.field(np.arange(5.0), more_stations)}
    self.assertEqual(
        randomness_features(more_inputs)['point_randomness'].coordinate,
        cx.coords.compose(more_stations, sample_axis),
    )
    # Without a sample axis values live on the template coordinate.
    scalar_features = feature_transforms.RandomnessFeatures(
        random_process=random_process, like='longitude'
    )
    self.assertEqual(
        scalar_features(inputs)['randomness'].coordinate, stations
    )
    with self.assertRaisesRegex(ValueError, 'Exactly one of'):
      feature_transforms.RandomnessFeatures(
          random_process=random_process, coord=stations, like='longitude'
      )
    with self.assertRaisesRegex(ValueError, 'Exactly one of'):
      feature_transforms.RandomnessFeatures(random_process=random_process)

  def test_climatology_features(self):
    grid = coordinates.LonLatGrid.TL31()
    climatology_features = feature_transforms.ClimatologyFeatures(
        coords={'sea_surface_temperature': grid},
        rngs=nnx.Rngs(0),
    )
    self._test_feature_module(
        climatology_features,
        {'time': cx.field(jdt.to_datetime('2025-01-09T15:00'))},
    )

  def test_point_cyclic_covariates(self):
    station = cx.SizedAxis('station', 4)
    inputs = {
        'longitude': cx.field(np.array([0.0, 90.0, 180.0, -90.0]), station),
        'latitude': cx.field(np.array([-45.0, 0.0, 45.0, 60.0]), station),
        'time': cx.field(jdt.to_datetime('2025-06-21T12:00')),
        'time_offset': cx.field(np.array([0.0, 3600.0, 0.0, 0.0]), station),
    }

    with self.subTest('default_axis'):
      transform = feature_transforms.PointCyclicCovariates()
      self._test_feature_module(transform, inputs)
      out = transform(inputs)['cyclic_covariates']
      expected_axis = cx.SizedAxis('cyclic_covariates_at_loc', 10)
      self.assertEqual(
          out.coordinate, cx.coords.compose(station, expected_axis)
      )
      self.assertTrue(np.all(np.abs(out.data) <= 1.0 + 1e-6))
      untagged = out.untag(station, expected_axis).data
      lat_rad = np.deg2rad(inputs['latitude'].data)
      np.testing.assert_allclose(untagged[:, 0], np.sin(lat_rad), atol=1e-6)
      np.testing.assert_allclose(untagged[:, 1], np.cos(lat_rad), atol=1e-6)
      # Local solar time at 12:00 UTC is noon at lon=0 and midnight at
      # lon=180 (both without time offset): first diurnal harmonic flips sign.
      np.testing.assert_allclose(untagged[0, 2:4], -untagged[2, 2:4], atol=1e-5)
      # Harmonics of day-of-year are shared by stations with equal offsets.
      np.testing.assert_allclose(untagged[0, 6:], untagged[3, 6:], atol=1e-6)

    with self.subTest('without_latitude_and_time_offset'):
      transform = feature_transforms.PointCyclicCovariates(
          num_harmonics=1, include_latitude=False, axis='custom_cyclic'
      )
      no_offset_inputs = {k: v for k, v in inputs.items() if k != 'time_offset'}
      out = transform(no_offset_inputs)['cyclic_covariates']
      expected_axis = cx.SizedAxis('custom_cyclic', 4)
      self.assertEqual(
          out.coordinate, cx.coords.compose(station, expected_axis)
      )

    with self.subTest('missing_time_raises'):
      transform = feature_transforms.PointCyclicCovariates()
      no_time_inputs = {k: v for k, v in inputs.items() if k != 'time'}
      with self.assertRaisesRegex(ValueError, 'missing required keys'):
        transform(no_time_inputs)

    with self.subTest('invalid_axis_shape_raises'):
      with self.assertRaisesRegex(ValueError, 'must have shape'):
        feature_transforms.PointCyclicCovariates(
            axis=cx.SizedAxis('bad_axis', 8)
        )

    with self.subTest('no_covariates_raises'):
      with self.assertRaisesRegex(ValueError, 'produces no covariates'):
        feature_transforms.PointCyclicCovariates(
            num_harmonics=0, include_latitude=False
        )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
