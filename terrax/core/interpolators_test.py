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
from terrax.core import patch_geometry


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

  @parameterized.named_parameters(
      dict(testcase_name='bilinear', method='bilinear'),
      dict(testcase_name='nearest', method='nearest'),
  )
  def test_lon_lat_to_swath_interpolator(self, method: str):
    grid = coordinates.LonLatGrid.TL63()
    lon_1d = np.deg2rad(grid.fields['longitude'].data)[:, None]
    lat_1d = np.deg2rad(grid.fields['latitude'].data)[None, :]
    smooth_fn = lambda lon_r, lat_r: np.sin(lon_r) * np.cos(lat_r)
    f_grid = cx.field(smooth_fn(lon_1d, lat_1d).astype(np.float32), grid)

    patch = coordinates.SwathPatch(
        scanlines=8,
        fov=6,
        max_scan_angle=52.725,
        satellite_altitude_km=833.0,
    )
    patch_axis = cx.LabeledAxis('patch_index', np.arange(2))
    coord = cx.coords.compose(patch_axis, patch)

    # Patch 0 crosses 360->0 boundary; Patch 1 has trailing NaN scanlines.
    lons_p0 = np.linspace(355.0, 365.0, 6, dtype=np.float32) % 360.0
    lats_p0 = np.linspace(-20.0, 20.0, 8, dtype=np.float32)
    lon_p0, lat_p0 = np.meshgrid(lons_p0, lats_p0)

    lon_p1 = lon_p0.copy()
    lat_p1 = lat_p0.copy()
    lon_p1[5:, :] = np.nan
    lat_p1[5:, :] = np.nan

    lon_query = cx.field(np.stack([lon_p0, lon_p1], axis=0), coord)
    lat_query = cx.field(np.stack([lat_p0, lat_p1], axis=0), coord)

    interp = interpolators.LonLatToSwathInterpolator(method=method)
    out = interp(f_grid, lon_query, lat_query)
    self.assertEqual(out.coordinate, coord)

    expected_p0 = smooth_fn(np.deg2rad(lon_p0), np.deg2rad(lat_p0))
    atol = 0.02 if method == 'bilinear' else 0.1
    np.testing.assert_allclose(out.data[0], expected_p0, atol=atol)
    # Trailing NaN scanlines in Patch 1 must remain NaN (even for nearest).
    self.assertTrue(np.all(np.isnan(out.data[1, 5:, :])))
    self.assertTrue(np.all(np.isfinite(out.data[1, :5, :])))

    with self.subTest('skip_missing_and_include_query_fields'):
      scalar_f = cx.field(np.float32(3.0))
      with self.assertRaises(ValueError):
        interp({'a': f_grid, 's': scalar_f}, lon_query, lat_query)
      interp_skip = interpolators.LonLatToSwathInterpolator(
          method=method,
          skip_missing=True,
          include_query_fields_in_output=True,
      )
      out_dict = interp_skip({
          'a': f_grid,
          's': scalar_f,
          'longitude': lon_query,
          'latitude': lat_query,
      })
      cx.testing.assert_fields_equal(out_dict['s'], scalar_f)
      cx.testing.assert_fields_equal(out_dict['longitude'], lon_query)

    with self.subTest('shares_neighbors_across_dict_keys_and_levels'):
      lev = cx.SizedAxis('level', 4)
      f_3d = cx.field(
          np.ones((4,) + grid.shape, dtype=np.float32), lev, grid
      )
      jaxpr = jax.make_jaxpr(interp)(
          {'a': f_3d, 'b': f_3d, 'longitude': lon_query, 'latitude': lat_query}
      )
      # Latitude searchsorted constant table (shape == (n_lat,)) must be bound
      # only once across both 3D fields and all 4 vertical levels, and uniform
      # longitude needs no (n_lon,) searchsorted table at all.
      lat_tables = [
          c for c in jaxpr.consts if np.shape(c) == (grid.shape[1],)
      ]
      lon_tables = [
          c for c in jaxpr.consts if np.shape(c) == (grid.shape[0],)
      ]
      self.assertLen(lat_tables, 1)
      self.assertEmpty(lon_tables)

  def test_swath_to_lon_lat_interpolator_and_gradients(self):
    target_grid = coordinates.LonLatGrid.T21()
    patch = coordinates.SwathPatch(
        scanlines=20,
        fov=16,
        max_scan_angle=52.725,
        satellite_altitude_km=833.0,
    )
    sat_axis = cx.LabeledAxis('satellite', np.array([224, 225]))
    patch_axis = cx.LabeledAxis('patch_index', np.arange(2))
    coord = cx.coords.compose(sat_axis, patch_axis, patch)

    # Construct swath patches including footprints at the northernmost corner
    # grid cells (lon_idx in {0, n_lon - 1}, lat_idx == n_lat - 1) alongside an
    # all-NaN padded patch (satellite=1, patch_index=1) to verify NaN queries
    # never corrupt corner cells via 0.0 * NaN.
    lons = np.linspace(330.0, 390.0, 16, dtype=np.float32) % 360.0
    lats = np.linspace(50.0, 89.0, 20, dtype=np.float32)
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    lon_data = np.broadcast_to(lon_2d, coord.shape).copy()
    lat_data = np.broadcast_to(lat_2d, coord.shape).copy()
    lon_data[1, 1, :, :] = np.nan
    lat_data[1, 1, :, :] = np.nan

    lon_q = cx.field(lon_data, coord)
    lat_q = cx.field(lat_data, coord)
    val_data = np.where(np.isfinite(lon_data), 250.0, np.nan).astype(np.float32)
    f_swath = cx.field(val_data, coord)

    splat = interpolators.SwathToLonLatInterpolator(
        target_grid=target_grid,
        method='bilinear',
        reduction_dims=('satellite', 'patch_index'),
    )
    gridded = splat(
        {'tb': f_swath, 'longitude': lon_q, 'latitude': lat_q}
    )['tb']
    self.assertEqual(gridded.coordinate, target_grid)
    # Corner cells (0, -1) and (-1, -1) covered by valid footprints must remain
    # finite (250.0), uncorrupted by the NaN-padded patch.
    np.testing.assert_allclose(gridded.data[0, -1], 250.0, rtol=1e-5)
    np.testing.assert_allclose(gridded.data[-1, -1], 250.0, rtol=1e-5)
    valid_cells = np.isfinite(gridded.data)
    self.assertGreater(np.sum(valid_cells), 20)
    np.testing.assert_allclose(gridded.data[valid_cells], 250.0, rtol=1e-5)

    # Verify gradients w.r.t. swath values are finite everywhere (no 0*NaN or
    # 0/0 corruption from unobserved grid cells or NaN footprints).
    def _loss(raw_swath_vals):
      f = cx.field(raw_swath_vals, coord)
      g = splat.interpolate_field(f, lon_q, lat_q)
      return jax.numpy.nansum(g.data**2)

    grads = jax.grad(_loss)(jax.numpy.ones(coord.shape, dtype=np.float32))
    self.assertTrue(np.all(np.isfinite(np.asarray(grads))))

  def test_weighted_patch_stitching(self):
    target_grid = coordinates.LonLatGrid.T21()
    patch = coordinates.LocalGridPatch(
        x_nodes=16, y_nodes=16, dx_deg=2.0, dy_deg=2.0
    )
    patch_axis = cx.LabeledAxis('patch_index', np.arange(2))
    c_lon = cx.field(np.array([100.0, 115.0], dtype=np.float32), patch_axis)
    c_lat = cx.field(np.array([0.0, 0.0], dtype=np.float32), patch_axis)
    lon_q, lat_q = patch_geometry.local_patch_lon_lat(patch, c_lon, c_lat)

    # Patch 0 has value 10.0, Patch 1 has value 20.0; with raised-cosine
    # blending weights, the boundary of Patch 1 approaches weight 0.
    vals = np.ones((2, 16, 16), dtype=np.float32)
    vals[0] = 10.0
    vals[1] = 20.0
    f_patches = cx.field(vals, cx.coords.compose(patch_axis, patch))
    weights = patch_geometry.patch_blending_weights(patch, taper_fraction=0.35)

    stitcher = interpolators.PatchToLonLatInterpolator(
        target_grid=target_grid,
        method='bilinear',
        reduction_dims=('patch_index',),
    )
    out = stitcher(
        {'val': f_patches, 'longitude': lon_q, 'latitude': lat_q},
        weights=weights,
    )['val']
    self.assertEqual(out.coordinate, target_grid)
    valid = out.data[np.isfinite(out.data)]
    self.assertGreater(valid.size, 10)
    self.assertTrue(np.all((valid >= 10.0 - 1e-4) & (valid <= 20.0 + 1e-4)))

  def test_patch_to_local_patch_interpolator(self):
    swath = coordinates.SwathPatch(
        scanlines=24, fov=20, max_scan_angle=45.0, satellite_altitude_km=833.0
    )
    local_patch = coordinates.LocalGridPatch(
        x_nodes=12, y_nodes=12, dx_deg=1.0, dy_deg=1.0
    )
    # Create a swath covering roughly lon in [95, 110], lat in [-8, 8], which
    # partially overlaps a LocalGridPatch centered at (105, 0) spanning
    # roughly [99.5, 110.5] x [-5.5, 5.5].
    lons = np.linspace(95.0, 105.0, 20, dtype=np.float32)
    lats = np.linspace(-8.0, 8.0, 24, dtype=np.float32)
    lon_2d, lat_2d = np.meshgrid(lons, lats)
    lon_s = cx.field(lon_2d, swath)
    lat_s = cx.field(lat_2d, swath)
    tb = cx.field(np.full(swath.shape, 275.0, dtype=np.float32), swath)

    co_proj = interpolators.SwathToLocalPatchInterpolator(
        target_patch=local_patch,
        method='bilinear',
        return_coverage_mask=True,
    )
    res = co_proj(
        {'tb': tb, 'longitude': lon_s, 'latitude': lat_s},
        center_lon=cx.field(np.float32(105.0)),
        center_lat=cx.field(np.float32(0.0)),
    )
    self.assertEqual(res['tb'].coordinate, local_patch)
    self.assertIn('coverage_mask', res)
    mask = res['coverage_mask'].data
    # Left half of local_patch (lon <= 105) is covered; right edge is uncovered.
    self.assertGreater(np.sum(mask > 0.5), 20)
    self.assertGreater(np.sum(mask == 0.0), 10)
    covered_vals = res['tb'].data[mask > 0.5]
    np.testing.assert_allclose(covered_vals, 275.0, rtol=1e-5)
    self.assertTrue(np.all(np.isnan(res['tb'].data[mask == 0.0])))

    # Verify NaN-padded scanlines do not corrupt corner cells of local_patch,
    # and coverage_mask stays strictly 2D on local_patch even with skip_missing
    # scalar keys first and 3D level fields.
    lon_full, lat_full = patch_geometry.local_patch_lon_lat(
        local_patch, cx.field(np.float32(105.0)), cx.field(np.float32(0.0))
    )
    swath_small = coordinates.SwathPatch(
        scanlines=14, fov=12, max_scan_angle=45.0, satellite_altitude_km=833.0
    )
    lon_pad = np.full(swath_small.shape, np.nan, dtype=np.float32)
    lat_pad = np.full(swath_small.shape, np.nan, dtype=np.float32)
    lon_pad[:12, :] = lon_full.data
    lat_pad[:12, :] = lat_full.data
    lev = cx.SizedAxis('level', 3)
    tb_3d = cx.field(
        np.full((3,) + swath_small.shape, 300.0, dtype=np.float32),
        lev,
        swath_small,
    )
    co_proj_skip = interpolators.PatchToLocalPatchInterpolator(
        target_patch=local_patch,
        method='bilinear',
        skip_missing=True,
        return_coverage_mask=True,
    )
    res_3d = co_proj_skip(
        {
            'scalar_meta': cx.field(np.float32(1.0)),
            'tb_3d': tb_3d,
            'longitude': cx.field(lon_pad, swath_small),
            'latitude': cx.field(lat_pad, swath_small),
        },
        center_lon=cx.field(np.float32(105.0)),
        center_lat=cx.field(np.float32(0.0)),
    )
    self.assertEqual(res_3d['coverage_mask'].coordinate, local_patch)
    self.assertEqual(
        res_3d['tb_3d'].coordinate, cx.coords.compose(lev, local_patch)
    )
    np.testing.assert_allclose(
        res_3d['tb_3d'].data[:, -1, -1], 300.0, rtol=1e-5
    )
    np.testing.assert_allclose(res_3d['coverage_mask'].data[-1, -1], 1.0)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
