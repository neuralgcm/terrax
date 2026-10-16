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

"""Tests for interpolation routines."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import numpy as np
from terrax.core import coordinates
from terrax.core import interpolators


class InterpolatorsTest(parameterized.TestCase):
  """Tests interpolator modules."""

  @parameterized.named_parameters(
      dict(
          testcase_name='spherical_coords',
          target_grid=coordinates.SphericalHarmonicGrid.T21(),
          input_coords=coordinates.SphericalHarmonicGrid.TL31(),
      ),
  )
  def test_spectral_regridder(self, target_grid, input_coords):
    regridder = interpolators.SpectralRegridder(target_grid)
    inputs = cx.field(np.ones(input_coords.shape), input_coords)
    outputs = regridder(inputs)
    output_coords = cx.get_coordinate(outputs)
    self.assertEqual(output_coords, target_grid)

  @parameterized.named_parameters(
      dict(
          testcase_name='TL63_to_T21',
          input_grid=coordinates.LonLatGrid.TL63(),
          extra_coords=[],
          target_grid=coordinates.LonLatGrid.T21(),
      ),
  )
  def test_conservative_regridder(self, input_grid, extra_coords, target_grid):
    input_coords = cx.coords.compose(*extra_coords, input_grid)
    inputs = cx.field(np.ones(input_coords.shape), input_coords)
    regridder = interpolators.ConservativeRegridder(target_grid)
    outputs = regridder(inputs)
    actual_out_coords = cx.get_coordinate(outputs)
    expected_out_coords = cx.coords.compose(*extra_coords, target_grid)
    self.assertEqual(actual_out_coords, expected_out_coords)

  @parameterized.named_parameters(
      dict(
          testcase_name='TL63_to_EL120',
          input_grid=coordinates.LonLatGrid.TL63(),
          extra_coords=[cx.SizedAxis('level', 3)],
          target_grid=coordinates.LonLatGrid.EL120(),
      ),
  )
  def test_bilinear_regridder(self, input_grid, extra_coords, target_grid):
    input_coords = cx.coords.compose(*extra_coords, input_grid)
    inputs = cx.field(np.ones(input_coords.shape), input_coords)
    regridder = interpolators.BilinearRegridder(target_grid)
    outputs = regridder(inputs)
    expected_out_coords = cx.coords.compose(*extra_coords, target_grid)
    expected = cx.field(np.ones(expected_out_coords.shape), expected_out_coords)
    cx.testing.assert_fields_allclose(outputs, expected, atol=1e-6)

  def test_bilinear_regridder_periodic_longitude(self):
    source_grid = coordinates.LonLatGrid.TL63()
    target_grid = coordinates.LonLatGrid.EL120()
    s_lon = np.asarray(source_grid.fields['longitude'].data)
    t_lon = np.asarray(target_grid.fields['longitude'].data)
    # Create a field that is 0 everywhere except at 0 deg where it is 1.0.
    # Target lon 358.5 deg lies between 357.1875 deg and 360.0 (0.0) deg.
    field_data = np.zeros(source_grid.shape, dtype=np.float32)
    field_data[0, :] = 1.0  # lon = 0.0 deg
    inputs = {'f': cx.field(field_data, source_grid)}
    regridder = interpolators.BilinearRegridder(target_grid)
    outputs = regridder(inputs)['f']

    # At target lon = 358.5 deg (last index -1 of 240x121), periodic bilinear
    # interpolation between 357.1875 (val=0) and 360.0 (val=1) gives:
    # (358.5 - 357.1875) / (360.0 - 357.1875) = 1.3125 / 2.8125 = 7 / 15.
    expected_val = (t_lon[-1] - s_lon[-1]) / (360.0 - s_lon[-1])
    lat_coord = target_grid.fields['latitude'].coordinate
    expected_last_lon = cx.field(
        np.full(lat_coord.shape, expected_val), lat_coord
    )
    cx.testing.assert_fields_allclose(
        outputs.isel(longitude=-1), expected_last_lon, atol=1e-6
    )

  def test_bilinear_regridder_skipna(self):
    source_grid = coordinates.LonLatGrid.TL63()
    target_grid = coordinates.LonLatGrid.T21()
    field_data = np.ones(source_grid.shape, dtype=np.float32)
    field_data[0, 0] = np.nan
    inputs = cx.field(field_data, source_grid)

    out_strict = interpolators.BilinearRegridder(target_grid, skipna=False)(
        inputs
    )
    self.assertTrue(np.isnan(out_strict.data).any())

    out_skipna = interpolators.BilinearRegridder(target_grid, skipna=True)(
        inputs
    )
    expected = cx.field(np.ones(target_grid.shape), target_grid)
    cx.testing.assert_fields_allclose(out_skipna, expected, atol=1e-6)

  def test_bilinear_regridder_exact(self):
    # Target nodes coincide with source nodes, hence values are selected.
    source_grid = coordinates.LonLatGrid(longitude_nodes=4, latitude_nodes=3)
    target_grid = coordinates.LonLatGrid(longitude_nodes=2, latitude_nodes=3)
    data = np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    regridder = interpolators.BilinearRegridder(target_grid)
    outputs = regridder(cx.field(data.astype(np.float32), source_grid))
    expected = cx.field(np.array([[0, 1, 2], [7, 8, 9]]), target_grid)
    cx.testing.assert_fields_allclose(outputs, expected, atol=1e-5)

  def test_bilinear_regridder_reproduces_linear_latitude_profile(self):
    # Linear interpolation is exact for fields that are linear in latitude, with
    # values clamped poleward of the outermost source latitudes.
    source_grid = coordinates.LonLatGrid.TL63()
    target_grid = coordinates.LonLatGrid.EL120()
    source_lat = source_grid.fields['latitude'].data
    target_lat = target_grid.fields['latitude'].data
    inputs = cx.field(
        np.broadcast_to(source_lat, source_grid.shape), source_grid
    )
    outputs = interpolators.BilinearRegridder(target_grid)(inputs)
    expected_lat = np.clip(target_lat, source_lat.min(), source_lat.max())
    expected = cx.field(
        np.broadcast_to(expected_lat, target_grid.shape), target_grid
    )
    cx.testing.assert_fields_allclose(outputs, expected, atol=1e-4)

  @parameterized.named_parameters(
      dict(
          testcase_name='gaussian_to_equiangular_with_poles',
          source_grid=coordinates.LonLatGrid.TL63(),
          target_grid=coordinates.LonLatGrid.EL120(),
      ),
      dict(
          testcase_name='equiangular_with_poles_to_gaussian',
          source_grid=coordinates.LonLatGrid.EL120(),
          target_grid=coordinates.LonLatGrid.T21(),
      ),
      dict(
          testcase_name='shifted_longitudes',
          source_grid=coordinates.LonLatGrid.T21(),
          target_grid=coordinates.LonLatGrid(
              longitude_nodes=96, latitude_nodes=48, longitude_offset=0.3
          ),
      ),
  )
  def test_bilinear_regridder_gather_matches_matmul(
      self, source_grid, target_grid
  ):
    level = cx.SizedAxis('level', 2)
    data = np.random.RandomState(0).randn(2, *source_grid.shape)
    data[0, :5, :5] = np.nan
    inputs = cx.field(data, level, source_grid)
    for skipna in (False, True):
      with self.subTest(f'{skipna=}'):
        matmul = interpolators.BilinearRegridder(
            target_grid, skipna=skipna, method='matmul'
        )
        gather = interpolators.BilinearRegridder(
            target_grid, skipna=skipna, method='gather'
        )
        cx.testing.assert_fields_allclose(
            gather(inputs), matmul(inputs), atol=1e-5
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='cpu',
          platform='cpu',
          expected_op='stablehlo.gather',
          unexpected_op='stablehlo.dot_general',
      ),
      dict(
          testcase_name='tpu',
          platform='tpu',
          expected_op='stablehlo.dot_general',
          unexpected_op='stablehlo.gather',
      ),
  )
  def test_bilinear_regridder_auto_method_depends_on_platform(
      self, platform, expected_op, unexpected_op
  ):
    source_grid = coordinates.LonLatGrid.T21()
    target_grid = coordinates.LonLatGrid.TL31()
    inputs = cx.field(np.ones(source_grid.shape, np.float32), source_grid)
    regridder = interpolators.BilinearRegridder(target_grid)
    traced = jax.jit(regridder).trace(inputs)
    hlo = traced.lower(lowering_platforms=(platform,)).as_text()
    self.assertIn(expected_op, hlo)
    self.assertNotIn(unexpected_op, hlo)

  def test_bilinear_regridder_rejects_unknown_method(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported'):
      interpolators.BilinearRegridder(
          coordinates.LonLatGrid.T21(), method='nearest'
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='bilinear',
          regridder_cls=interpolators.BilinearRegridder,
          skipna=False,
      ),
      dict(
          testcase_name='bilinear_skipna',
          regridder_cls=interpolators.BilinearRegridder,
          skipna=True,
      ),
      dict(
          testcase_name='conservative',
          regridder_cls=interpolators.ConservativeRegridder,
          skipna=False,
      ),
      dict(
          testcase_name='conservative_skipna',
          regridder_cls=interpolators.ConservativeRegridder,
          skipna=True,
      ),
  )
  def test_regridding_nans(self, regridder_cls, skipna):
    source_grid = coordinates.LonLatGrid.TL127()
    target_grid = coordinates.LonLatGrid.TL63()
    lon = np.deg2rad(source_grid.fields['longitude'].data)[:, np.newaxis]
    lat = np.deg2rad(source_grid.fields['latitude'].data)[np.newaxis, :]
    in_valid = lat**2 + (lon - np.pi) ** 2 < (np.pi / 2) ** 2
    inputs = cx.field(np.where(in_valid, 1.0, np.nan), source_grid)
    outputs = regridder_cls(target_grid, skipna=skipna)(inputs).data
    out_valid = ~np.isnan(outputs)
    # Fractions of valid values are similar but not identical.
    np.testing.assert_allclose(out_valid.mean(), in_valid.mean(), atol=0.1)
    np.testing.assert_allclose(outputs[out_valid], 1.0, rtol=1e-5)

  def test_conservative_regridder_nested_grids_nans(self):
    # Cell edges of nested grids coincide, for which float32 round-off results
    # in tiny weights of adjacent source cells that must not propagate NaNs.
    ratio = 3
    source_grid = coordinates.LonLatGrid(360, 181, 'equiangular_with_poles')
    target_grid = coordinates.LonLatGrid(120, 61, 'equiangular_with_poles')
    # NaNs on a checkerboard of target cells, each covering source cells
    # (ratio * i - 1, ratio * i, ratio * i + 1) along both dimensions.
    target_i, target_j = np.indices(target_grid.shape)
    target_is_nan = (target_i + target_j) % 2 == 0
    source_i, source_j = np.indices(source_grid.shape)
    source_to_target_i = np.round(source_i / ratio).astype(int)
    source_to_target_i %= target_grid.shape[0]
    source_to_target_j = np.round(source_j / ratio).astype(int)
    source_is_nan = target_is_nan[source_to_target_i, source_to_target_j]
    inputs = cx.field(np.where(source_is_nan, np.nan, 1.0), source_grid)
    outputs = interpolators.ConservativeRegridder(target_grid)(inputs).data
    np.testing.assert_array_equal(np.isnan(outputs), target_is_nan)
    np.testing.assert_allclose(outputs[~target_is_nan], 1.0, rtol=1e-5)

  def test_lon_lat_regridder_rejects_padded_grids(self):
    unpadded = coordinates.LonLatGrid.T21()
    padded = coordinates.LonLatGrid(
        longitude_nodes=64, latitude_nodes=32, lon_lat_padding=(4, 4)
    )
    inputs = cx.field(np.ones(unpadded.shape), unpadded)
    with self.assertRaisesRegex(ValueError, 'unpadded'):
      interpolators.BilinearRegridder(padded)(inputs)

  def test_lon_lat_regridder_shares_weights_across_fields(self):
    source_grid = coordinates.LonLatGrid.TL63()
    target_grid = coordinates.LonLatGrid.EL120()
    regridder = interpolators.BilinearRegridder(target_grid)
    field = cx.field(np.ones(source_grid.shape), source_grid)
    consts = jax.make_jaxpr(regridder)({'a': field, 'b': field}).consts
    lon_weights_shape = (target_grid.shape[0], source_grid.shape[0])
    lon_weights = [c for c in consts if np.shape(c) == lon_weights_shape]
    self.assertLen(lon_weights, 1)

  def test_bilinear_regridder_gather_shares_neighbors_across_fields(self):
    source_grid = coordinates.LonLatGrid.TL63()
    target_grid = coordinates.LonLatGrid.EL120()
    regridder = interpolators.BilinearRegridder(target_grid, method='gather')
    field = cx.field(np.ones(source_grid.shape), source_grid)
    consts = jax.make_jaxpr(regridder)({'a': field, 'b': field}).consts
    lon_neighbors_shape = (target_grid.shape[0], 2)
    lon_neighbors = [c for c in consts if np.shape(c) == lon_neighbors_shape]
    self.assertLen(lon_neighbors, 2)  # indices and weights.


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
