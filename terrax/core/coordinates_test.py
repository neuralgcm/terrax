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
import collections
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from coordax import testing as coordax_testing
from dinosaur import hybrid_coordinates
from dinosaur import sigma_coordinates
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates
from terrax.core import parallelism
from terrax.core import spherical_harmonics
from terrax.core import units
from terrax.core import xarray_utils


class CoordinatesTest(parameterized.TestCase):
  """Tests that coordinate have expected shapes and dims."""

  @parameterized.named_parameters(
      dict(
          testcase_name='spherical_harmonic',
          coords=coordinates.SphericalHarmonicGrid.TL31(),
          expected_dims=('longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(64, 33),
      ),
      dict(
          testcase_name='lon_lat',
          coords=coordinates.LonLatGrid.T21(),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 32),
      ),
      dict(
          testcase_name='product_of_levels',
          coords=cx.coords.compose(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.PressureLevels([50, 100, 200, 800, 1000]),
              coordinates.HybridLevels.with_n_levels(7),
              coordinates.LayerLevels(3),
          ),
          expected_dims=('sigma', 'pressure', 'hybrid', 'layer_index'),
          expected_shape=(4, 5, 7, 3),
      ),
      dict(
          testcase_name='sigma_and_sigma_boundaries',
          coords=cx.coords.compose(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.SigmaBoundaries.equidistant(4),
          ),
          expected_dims=('sigma', 'sigma_boundaries'),
          expected_shape=(4, 5),
      ),
      dict(
          testcase_name='sigma_spherical_harmonic_product',
          coords=cx.coords.compose(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.SphericalHarmonicGrid.T21(),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='batched_trajectory',
          coords=cx.coords.compose(
              cx.SizedAxis('batch', 7),
              coordinates.TimeDelta(np.arange(5) * np.timedelta64(1, 'h')),
              coordinates.PressureLevels([50, 200, 800, 1000]),
              coordinates.LonLatGrid.T21(),
          ),
          expected_dims=(
              'batch',
              'timedelta',
              'pressure',
              'longitude',
              'latitude',
          ),
          expected_shape=(7, 5, 4, 64, 32),
          expected_field_transform=lambda f: f.untag('batch').tag('batch'),
      ),
      dict(
          testcase_name='coordinate_shard_none',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=1, z=2),
              dimension_partitions={'longitude': None, 'latitude': None},
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(128, 64),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
      dict(
          testcase_name='coordinate_shard_longitude',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=1, z=2),
              dimension_partitions={'longitude': ('x', 'z'), 'latitude': None},
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(32, 64),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
      dict(
          testcase_name='coordinate_shard_longitude_and_latitude',
          coords=parallelism.CoordinateShard(
              coordinate=coordinates.LonLatGrid.T42(),
              spmd_mesh_shape=collections.OrderedDict(x=2, y=4, z=2),
              dimension_partitions={'longitude': 'x', 'latitude': ('y', 'z')},
          ),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 8),  # unchanged.
          supports_xarray_roundtrip=False,
      ),
  )
  def test_coordinates(
      self,
      coords: cx.Coordinate,
      expected_dims: tuple[str, ...],
      expected_shape: tuple[int, ...],
      expected_field_transform: Callable[[cx.Field], cx.Field] = lambda x: x,
      supports_xarray_roundtrip: bool = True,
  ):
    """Tests that coordinates are pytrees and have expected shape and dims."""
    with self.subTest('pytree_roundtrip'):
      leaves, tree_def = jax.tree.flatten(coords)
      reconstructed = jax.tree.unflatten(tree_def, leaves)
      self.assertEqual(reconstructed, coords)

    with self.subTest('dims'):
      self.assertEqual(coords.dims, expected_dims)

    with self.subTest('shape'):
      self.assertEqual(coords.shape, expected_shape)

    if supports_xarray_roundtrip:
      with self.subTest('xarray_roundtrip'):
        field = cx.field(np.zeros(coords.shape), coords)
        data_array = field.to_xarray()
        reconstructed = xarray_utils.from_xarray(data_array)
        expected = expected_field_transform(field)
        coordax_testing.assert_fields_equal(reconstructed, expected)


class CoordinatesMethodsTest(parameterized.TestCase):
  """Tests methods of coordinate objects."""

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma_axis_minus_3',
          shape=(4, 5, 3),
          sigma_axis=-3,
      ),
      dict(
          testcase_name='sigma_axis_1',
          shape=(5, 4, 3),
          sigma_axis=1,
      ),
  )
  def test_sigma_level_integrate(self, shape, sigma_axis):
    sigma_coord = coordinates.SigmaLevels.equidistant(shape[sigma_axis])
    pos_sigma_axis = sigma_axis if sigma_axis >= 0 else sigma_axis + len(shape)
    coords = cx.coords.compose(*[
        sigma_coord if i == pos_sigma_axis else cx.SizedAxis(f'ax{i}', shape[i])
        for i in range(len(shape))
    ])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    field = cx.field(data, coords)
    integrated_field = sigma_coord.integrate(field)
    expected_data = sigma_coordinates.sigma_integral(
        data,
        sigma_coord.sigma_levels,
        axis=sigma_axis,
        keepdims=False,
    )
    np.testing.assert_allclose(integrated_field.data, expected_data, atol=1e-6)
    expected_dims = tuple(d for d in coords.dims if d not in sigma_coord.dims)
    self.assertEqual(integrated_field.dims, expected_dims)

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma_axis_minus_3',
          shape=(4, 5, 3),
          sigma_axis=-3,
      ),
      dict(
          testcase_name='sigma_axis_1',
          shape=(5, 4, 3),
          sigma_axis=1,
      ),
  )
  def test_sigma_level_integrate_over_pressure(self, shape, sigma_axis):
    sigma_coord = coordinates.SigmaLevels.equidistant(shape[sigma_axis])
    pos_sigma_axis = sigma_axis if sigma_axis >= 0 else sigma_axis + len(shape)
    coords = cx.coords.compose(*[
        sigma_coord if i == pos_sigma_axis else cx.SizedAxis(f'ax{i}', shape[i])
        for i in range(len(shape))
    ])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rng = np.random.RandomState(0)
    sp_data = rng.uniform(
        size=[s for i, s in enumerate(shape) if i != pos_sigma_axis]
    )
    sp_coords = cx.coords.compose(
        *[c for c in coords.axes if not isinstance(c, coordinates.SigmaLevels)]
    )
    field = cx.field(data, coords)
    sp_field = cx.field(sp_data, sp_coords)
    integrated_field = sigma_coord.integrate_over_pressure(field, sp_field)
    expected_data = sp_data * sigma_coordinates.sigma_integral(
        data,
        sigma_coord.sigma_levels,
        axis=sigma_axis,
        keepdims=False,
    )
    np.testing.assert_allclose(integrated_field.data, expected_data, atol=1e-6)
    expected_dims = tuple(d for d in coords.dims if d not in sigma_coord.dims)
    self.assertEqual(integrated_field.dims, expected_dims)

  @parameterized.named_parameters(
      dict(
          testcase_name='hybrid_axis_minus_3',
          shape=(4, 5, 3),
          hybrid_axis=-3,
      ),
      dict(
          testcase_name='hybrid_axis_1',
          shape=(5, 4, 3),
          hybrid_axis=1,
      ),
  )
  def test_hybrid_level_integrate_over_pressure(self, shape, hybrid_axis):
    sim_units = units.SI_UNITS
    hybrid_coord = coordinates.HybridLevels.with_n_levels(shape[hybrid_axis])
    pos_hybrid_axis = (
        hybrid_axis if hybrid_axis >= 0 else hybrid_axis + len(shape)
    )
    coords = cx.coords.compose(*[
        hybrid_coord
        if i == pos_hybrid_axis
        else cx.SizedAxis(f'ax{i}', shape[i])
        for i in range(len(shape))
    ])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    sp_shape = list(shape)
    sp_shape.pop(pos_hybrid_axis)
    sp_data = np.ones(sp_shape, dtype=np.float32)
    sp_coords = cx.coords.compose(*[
        c
        for c in coords.coordinates
        if not isinstance(c, coordinates.HybridLevels)
    ])
    field = cx.field(data, coords)
    sp_field = cx.field(sp_data, sp_coords)
    integrated_field = hybrid_coord.integrate_over_pressure(
        field, sp_field, sim_units=sim_units
    )
    a_nondim = sim_units.nondimensionalize(
        hybrid_coord.hybrid_levels.a_boundaries * units.parse_units('hPa')
    )
    nondim_levels = hybrid_coordinates.HybridCoordinates(
        a_nondim, hybrid_coord.hybrid_levels.b_boundaries
    )
    expected_data = hybrid_coordinates.integral_over_pressure(
        data,
        sp_data,
        nondim_levels,
        axis=hybrid_axis,
        keepdims=False,
    )
    np.testing.assert_allclose(integrated_field.data, expected_data, atol=1e-6)
    expected_dims = tuple(d for d in coords.dims if d not in hybrid_coord.dims)
    self.assertEqual(integrated_field.dims, expected_dims)

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma_axis_minus_3',
          shape=(4, 5, 3),
          sigma_axis=-3,
      ),
      dict(
          testcase_name='sigma_axis_1',
          shape=(5, 4, 3),
          sigma_axis=1,
      ),
  )
  def test_sigma_level_integrate_cumulative(self, shape, sigma_axis):
    sigma_coord = coordinates.SigmaLevels.equidistant(shape[sigma_axis])
    pos_sigma_axis = sigma_axis if sigma_axis >= 0 else sigma_axis + len(shape)
    coords = cx.coords.compose(*[
        sigma_coord if i == pos_sigma_axis else cx.SizedAxis(f'ax{i}', shape[i])
        for i in range(len(shape))
    ])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    field = cx.field(data, coords)
    integrated_field = sigma_coord.integrate_cumulative(field)
    expected_data = sigma_coordinates.cumulative_sigma_integral(
        data,
        sigma_coord.sigma_levels,
        axis=sigma_axis,
    )
    np.testing.assert_allclose(integrated_field.data, expected_data, atol=1e-6)
    self.assertEqual(integrated_field.dims, coords.dims)

  def test_lon_lat_grid_integrate(self):
    grid = coordinates.LonLatGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    radius = 123.4
    integral = grid.integrate(field, radius=radius)
    np.testing.assert_allclose(integral.data, 4 * np.pi * radius**2, rtol=1e-5)

  @parameterized.parameters(2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
  def test_lon_lat_grid_gaussian_grids(self, n):
    grid = getattr(coordinates.LonLatGrid, f'F{n}')()
    self.assertEqual(grid.shape, (4 * n, 2 * n))
    self.assertEqual(grid.latitude_spacing, 'gauss')
    field = cx.field(np.ones(grid.shape), grid)
    np.testing.assert_allclose(grid.mean(field).data, 1.0, rtol=1e-5)

  def test_lon_lat_grid_gaussian_grid_aliases(self):
    grid_cls = coordinates.LonLatGrid
    self.assertEqual(grid_cls.F16(), grid_cls.T21())
    self.assertEqual(grid_cls.F24(), grid_cls.T31())
    self.assertEqual(grid_cls.F32(), grid_cls.TL63())
    self.assertEqual(grid_cls.F48(), grid_cls.TL95())
    self.assertEqual(grid_cls.F64(), grid_cls.TL127())

  @parameterized.parameters(
      (120, (240, 121)),
      (180, (360, 181)),
      (720, (1440, 721)),
      (1800, (3600, 1801)),
  )
  def test_equiangular_linear_grids(self, max_wavenumber, expected_shape):
    lon_lat = getattr(coordinates.LonLatGrid, f'EL{max_wavenumber}')()
    ylm = getattr(coordinates.SphericalHarmonicGrid, f'EL{max_wavenumber}')()
    self.assertEqual(lon_lat.shape, expected_shape)
    self.assertEqual(lon_lat.latitude_spacing, 'equiangular_with_poles')
    self.assertEqual(
        ylm,
        coordinates.SphericalHarmonicGrid.construct(
            max_wavenumber=max_wavenumber
        ),
    )

  def test_lon_lat_grid_construct_explicit_nodes(self):
    grid = coordinates.LonLatGrid.construct(
        longitude_nodes=382, latitude_nodes=191
    )
    self.assertEqual(grid, coordinates.LonLatGrid(382, 191))
    with self.assertRaises(ValueError):
      coordinates.LonLatGrid.construct(
          gaussian_nodes=64, longitude_nodes=382, latitude_nodes=191
      )
    with self.assertRaises(ValueError):
      coordinates.LonLatGrid.construct(longitude_nodes=382)

  def test_lon_lat_grid_partial_integrate(self):
    n_lon, n_lat = 64, 32
    grid = coordinates.LonLatGrid(
        longitude_nodes=n_lon,
        latitude_nodes=n_lat,
        lon_lat_padding=(8, 4),
    )
    # data that is 1 everywhere except 2 on first half of longitudes
    data = np.ones(grid.shape)
    data[: (n_lon // 2), :] = 2
    field = cx.field(data, grid)

    field_lat = grid.integrate(field, dims='longitude')
    self.assertEqual(field_lat.coordinate, cx.SelectedAxis(grid, axis=1))

    field_lon = grid.integrate(field, dims='latitude')
    self.assertEqual(field_lon.coordinate, cx.SelectedAxis(grid, axis=0))

    scalar = grid.integrate(field)
    self.assertEqual(scalar.coordinate, cx.Scalar())

    sclar_from_lon = grid.integrate(field_lon, dims='longitude')
    sclar_from_lat = grid.integrate(field_lat, dims='latitude')
    cx.testing.assert_fields_allclose(sclar_from_lon, scalar)
    cx.testing.assert_fields_allclose(sclar_from_lat, scalar)

  def test_lon_lat_grid_mean(self):
    grid = coordinates.LonLatGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    mean = grid.mean(field)
    np.testing.assert_allclose(mean.data, 1.0, rtol=1e-5)

    n_lon, n_lat = 64, 32
    grid = coordinates.LonLatGrid(
        longitude_nodes=n_lon,
        latitude_nodes=n_lat,
        lon_lat_padding=(8, 4),
    )
    # data that is 1 everywhere except 2 on first half of longitudes
    data = np.ones(grid.shape)
    data[: (n_lon // 2), :] = 2
    field = cx.field(data, grid)

    field_lat = grid.mean(field, dims='longitude')
    self.assertEqual(field_lat.coordinate, cx.SelectedAxis(grid, axis=1))
    np.testing.assert_allclose(field_lat.data, 1.5, rtol=1e-5)

    field_lon = grid.mean(field, dims='latitude')
    self.assertEqual(field_lon.coordinate, cx.SelectedAxis(grid, axis=0))
    np.testing.assert_allclose(field_lon.data, data[:, 0], rtol=1e-5)

    scalar = grid.mean(field)
    self.assertEqual(scalar.coordinate, cx.Scalar())
    np.testing.assert_allclose(scalar.data, 1.5, rtol=1e-5)

    sclar_from_lon = grid.mean(field_lon, dims='longitude')
    sclar_from_lat = grid.mean(field_lat, dims='latitude')
    cx.testing.assert_fields_allclose(sclar_from_lon, scalar)
    cx.testing.assert_fields_allclose(sclar_from_lat, scalar)

  @parameterized.named_parameters(
      dict(testcase_name='float', c=2.5),
      dict(testcase_name='array', c=np.array(2.5)),
      dict(
          testcase_name='field_with_named_axes',
          c=cx.field(np.eye(3), cx.SizedAxis('a', 3), cx.SizedAxis('b', 3)),
      ),
  )
  def test_spherical_harmonic_grid_add_constant(self, c):
    """Tests that `add_constant` is consistent with nodal addition."""
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    a, b = cx.SizedAxis('a', 3), cx.SizedAxis('b', 3)
    levels = coordinates.SigmaLevels.equidistant(4)
    coords = cx.coords.compose(a, levels, b, ylm_grid)
    x_data = np.zeros(coords.shape)
    rng = np.random.RandomState(4)
    x_data[:, :, :, 0, 0] = 0.13
    x_data[:, :, :, 2:4, 2:6] = rng.uniform(size=(2, 4))
    x = cx.field(x_data, coords)
    grid = coordinates.LonLatGrid.T21()
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=grid,
        ylm_grid=ylm_grid,
    )
    expected = ylm_map.to_modal(ylm_map.to_nodal(x) + c)
    actual = ylm_grid.add_constant(x, c)
    coordax_testing.assert_fields_allclose(actual, expected, atol=1e-5)

  def test_spherical_harmonic_grid_add_constant_raises_with_positional_axes(
      self,
  ):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    x = cx.field(np.zeros(ylm_grid.shape), ylm_grid)
    c = cx.field(np.arange(3.0))
    with self.assertRaisesRegex(
        ValueError, 'Adding non-scalar constants without axes is not supported'
    ):
      ylm_grid.add_constant(x, c)

  def test_spherical_harmonic_grid_add_constant_raises_with_conflicting_axes(
      self,
  ):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    x = cx.field(np.zeros(ylm_grid.shape), ylm_grid)
    conflicting_axis = cx.SizedAxis('total_wavenumber', ylm_grid.shape[-1])
    c = cx.field(np.zeros(ylm_grid.shape[-1]), conflicting_axis)
    with self.assertRaisesRegex(
        ValueError, 'cannot have any of the dimensions'
    ):
      ylm_grid.add_constant(x, c)

  def test_spherical_harmonic_grid_add_constant_raises_with_new_axes(self):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    levels = coordinates.SigmaLevels.equidistant(4)
    x = cx.field(np.zeros(levels.shape + ylm_grid.shape), levels, ylm_grid)
    c = cx.field(np.arange(5), cx.SizedAxis('new_dim', 5))
    with self.assertRaisesRegex(
        ValueError, 'Introduction of new axes via add_constant is not supported'
    ):
      ylm_grid.add_constant(x, c)

  def test_sigma_levels_pressure_centers(self):
    levels = coordinates.SigmaLevels.equidistant(4)
    grid = coordinates.LonLatGrid.T21()
    ps = np.random.RandomState(0).uniform(9e4, 1e5, size=grid.shape)
    actual = levels.pressure_centers(cx.field(ps, grid))
    expected = cx.field(
        levels.centers[:, np.newaxis, np.newaxis] * ps, levels, grid
    )
    coordax_testing.assert_fields_allclose(actual, expected)

  def test_pressure_levels_pressure_centers(self):
    levels = coordinates.PressureLevels([10, 500, 1000])
    with self.subTest('pascal'):
      actual = levels.pressure_centers()
      expected = cx.field(np.array([1e3, 5e4, 1e5]), levels)
      coordax_testing.assert_fields_allclose(actual, expected)
    with self.subTest('nondimensional'):
      sim_units = units.SI_UNITS
      actual = levels.pressure_centers(sim_units=sim_units)
      expected = cx.field(
          sim_units.nondimensionalize(
              np.array([1e3, 5e4, 1e5]) * units.parse_units('Pa')
          ),
          levels,
      )
      coordax_testing.assert_fields_allclose(actual, expected)

  def test_hybrid_levels_ecmwf137_interpolated_p_top(self):
    p_top = 1.0  # hPa.
    levels = coordinates.HybridLevels.ecmwf137_interpolated(32, p_top=p_top)
    self.assertEqual(levels.shape, (32,))
    # Top interface is the first ECMWF interface at or below `p_top`.
    top_pressure = levels.a_boundaries[0] + levels.b_boundaries[0] * 1013.25
    self.assertGreaterEqual(top_pressure, p_top)
    full = coordinates.HybridLevels.ecmwf137_interpolated(32)
    self.assertEqual(full.a_boundaries[0], 0.0)


class CoordinatesSelectionTest(parameterized.TestCase):
  """Tests selection on coordinate objects."""

  def test_timedelta_selection(self):
    deltas = np.array([1, 2, 3], dtype='timedelta64[s]')
    time_coord = coordinates.TimeDelta(deltas)
    with self.subTest('select_by_value'):
      sel = time_coord.sel(timedelta=np.timedelta64(2, 's'))
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('select_by_slice'):
      sel = time_coord.sel(
          {time_coord: slice(np.timedelta64(1, 's'), np.timedelta64(2, 's'))}
      )
      expected = coordinates.TimeDelta(np.array([1, 2], dtype='timedelta64[s]'))
      self.assertEqual(sel, expected)
    with self.subTest('select_by_axis'):
      axis = coordinates.TimeDelta(np.array([1, 2], dtype='timedelta64[s]'))
      sel = time_coord.sel({time_coord: axis})
      self.assertEqual(sel, axis)

  def test_sigma_levels_selection(self):
    boundaries = [0.0, 0.5, 1.0]
    sigma_coord = coordinates.SigmaLevels(boundaries)
    with self.subTest('select_by_index'):
      sel = sigma_coord.isel(sigma=0)
      self.assertIsInstance(sel, cx.Scalar)
    with self.subTest('select_by_slice'):
      sliced = sigma_coord.isel(sigma=slice(0, 1))
      self.assertIsInstance(sliced, cx.LabeledAxis)
      self.assertEqual(sliced.shape, (1,))
      self.assertEqual(sliced.ticks[0], 0.25)
    sigma = coordinates.SigmaLevels.equidistant(8)
    with self.subTest('select_by_value_slice'):
      sel = sigma.sel({sigma: slice(0.2, 0.5)})
      sigma_values = sigma.fields['sigma'].data
      sigma_values = sigma_values[(0.2 <= sigma_values) & (sigma_values <= 0.5)]
      expected = cx.LabeledAxis('sigma', sigma_values)
      self.assertEqual(sel, expected)
    with self.subTest('select_by_labled_axis'):
      sigma_values = sigma.fields['sigma'].data
      axis = cx.LabeledAxis('sigma', sigma_values[::2])
      sel = sigma.sel({sigma: axis})
      self.assertEqual(sel, axis)

  def test_pressure_levels_selection(self):
    centers = [100.0, 200.0, 300.0]
    p_coord = coordinates.PressureLevels(centers)
    with self.subTest('exact_match'):
      sel = p_coord.sel(pressure=200.0)
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('nearest_neighbor'):
      sel = p_coord.sel(pressure=205.0, method='nearest')
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('by_labeled_axis'):
      request_pressure = coordinates.PressureLevels([100.0, 300.0])
      sel = p_coord.sel(pressure=request_pressure)
      self.assertEqual(sel, request_pressure)
    with self.subTest('select_by_value_slice'):
      sel = p_coord.sel({p_coord: slice(200, 800)})
      pressure_values = p_coord.fields['pressure'].data
      pressure_values = pressure_values[
          (200 <= pressure_values) & (pressure_values <= 800)
      ]
      expected = coordinates.PressureLevels(pressure_values)
      self.assertEqual(sel, expected)
    with self.subTest('select_by_labled_axis'):
      p_start = coordinates.PressureLevels.with_era5_levels()
      p_target = coordinates.PressureLevels.with_13_era5_levels()
      sel = p_start.sel({p_start: p_target})
      self.assertEqual(sel, p_target)

  def test_lon_lat_grid_selection(self):
    grid = coordinates.LonLatGrid.T21()  # 64x32
    with self.subTest('slice_longitude'):
      sliced = grid.isel(longitude=slice(0, 10))
      self.assertIsInstance(sliced, cx.CartesianProduct)
      self.assertEqual(sliced.shape, (10, 32))
    with self.subTest('index_latitude'):
      indexed = grid.isel(latitude=5)  # longitude preserved, remains original.
      self.assertEqual(indexed, grid.axes[0])
    with self.subTest('select_point'):
      lons, lats = grid.fields['longitude'].data, grid.fields['latitude'].data
      point = grid.sel(longitude=lons[5], latitude=lats[10])
      self.assertEqual(point, cx.Scalar())
    with self.subTest('select_range'):
      region = grid.sel(longitude=slice(0, 20))
      self.assertEqual(region.shape, (4, 32))
      np.testing.assert_array_less(region.fields['longitude'].data, 20)
    with self.subTest('using_grid_axis_as_key'):
      axis = cx.SelectedAxis(grid, axis=0)
      region = grid.isel({axis: slice(0, 20)})
      self.assertEqual(region.shape, (20, 32))
    with self.subTest('using_labled_axis_as_value'):
      axis = cx.LabeledAxis('longitude', grid.fields['longitude'].data[::2])
      region = grid.sel({grid.axes[0]: axis})
      self.assertEqual(region.shape, axis.shape + grid.shape[1:])

  def test_spherical_harmonic_grid_selection(self):
    grid = coordinates.SphericalHarmonicGrid.T21()
    with self.subTest('slice_wavenumbers'):
      sliced = grid.isel(longitude_wavenumber=slice(0, 5))
      self.assertIsInstance(sliced, cx.CartesianProduct)
      self.assertEqual(sliced.shape, (5, grid.shape[1]))
    with self.subTest('index_total_wavenumber'):
      sliced = grid.isel(total_wavenumber=10)
      self.assertEqual(sliced, grid.axes[0])
    with self.subTest('using_grid_axis_as_key'):
      axis = cx.SelectedAxis(grid, axis=1)
      region = grid.isel({axis: slice(0, 5)})
      self.assertEqual(region.shape, (grid.shape[0], 5))
    with self.subTest('using_labled_axis_as_value'):
      axis = cx.LabeledAxis(
          'total_wavenumber', grid.fields['total_wavenumber'].data[:5]
      )
      region = grid.sel({grid.axes[1]: axis})
      self.assertEqual(region.shape, grid.shape[0:1] + axis.shape)

  def test_sigma_boundaries_selection(self):
    boundaries = np.array([0.0, 0.5, 1.0])
    sigma_b = coordinates.SigmaBoundaries(boundaries)
    with self.subTest('select_by_index'):
      sel = sigma_b.isel(sigma_boundaries=0)
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('select_by_value'):
      sel = sigma_b.sel(sigma_boundaries=0.5)
      self.assertEqual(sel, cx.Scalar())

  def test_hybrid_levels_selection(self):
    a = [0.0, 0.1, 0.2]
    b = [1.0, 0.9, 0.8]
    hybrid = coordinates.HybridLevels(a, b)
    with self.subTest('select_by_index'):
      sel = hybrid.isel(hybrid=0)
      self.assertEqual(sel, cx.Scalar())

  def test_layer_levels_selection(self):
    layers = coordinates.LayerLevels(n_layers=5)
    with self.subTest('select_by_index'):
      sel = layers.isel(layer_index=2)
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('select_by_value'):
      sel = layers.sel(layer_index=3)
      self.assertEqual(sel, cx.Scalar())

  def test_soil_levels_selection(self):
    centers = [0.1, 0.5, 1.5]
    soil = coordinates.SoilLevels(centers)
    with self.subTest('select_by_index'):
      sel = soil.isel(soil_levels=1)
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('select_by_value'):
      sel = soil.sel(soil_levels=1.5)
      self.assertEqual(sel, cx.Scalar())
    with self.subTest('select_by_axis'):
      axis = coordinates.SoilLevels([0.1, 1.5])
      sel = soil.sel(soil_levels=axis)
      self.assertEqual(sel, axis)

  def test_swath_patch_viewing_geometry(self):
    patch = coordinates.SwathPatch.atms(scanlines=10)
    self.assertEqual(patch.dims, ('scanline', 'fov'))
    self.assertEqual(patch.shape, (10, 96))
    _, fov_axis = patch.axes

    with self.subTest('scan_angles_symmetry'):
      expected_angles = cx.field(
          np.linspace(-52.725, 52.725, 96, dtype=np.float32), fov_axis
      )
      cx.testing.assert_fields_allclose(
          patch.scan_angles, expected_angles, rtol=1e-5
      )
      flipped = cx.field(-patch.scan_angles.data[::-1], fov_axis)
      cx.testing.assert_fields_allclose(
          patch.scan_angles, flipped, atol=1e-5
      )

    with self.subTest('footprint_weights_and_spherical_quadrature'):
      cx.testing.assert_fields_allclose(
          cx.cmap(jnp.mean)(patch.footprint_weights.untag(fov_axis)),
          cx.field(np.float32(1.0)),
          rtol=1e-5,
      )
      # Limb cell Jacobian (~4.27x nadir on ATMS) exceeds nadir without squaring
      # slant_range (~8.28x).
      ratio = float(
          patch.footprint_weights.data[0] / patch.footprint_weights.data[47]
      )
      self.assertGreater(ratio, 4.0)
      self.assertLess(ratio, 4.6)

      # Quadrature check: spherical band mean of f(psi) = cos(4 * psi) across
      # the ATMS swath (with spherical surface element dA ~ cos(psi) dphi dpsi)
      # matches exact \int cos(4*psi)*cos(psi) dpsi / \int cos(psi) dpsi.
      r_ratio = 1.0 + patch.satellite_altitude_km / 6371.0
      scan_rad = np.deg2rad(np.abs(patch.scan_angles.data))
      psi = np.arcsin(r_ratio * np.sin(scan_rad)) - scan_rad
      half_step_rad = np.deg2rad(patch.max_scan_angle / (patch.fov - 1))
      beta_edge = np.deg2rad(patch.max_scan_angle) + half_step_rad
      psi_edge = np.arcsin(r_ratio * np.sin(beta_edge)) - beta_edge
      exact_band_mean = (
          0.1 * np.sin(5.0 * psi_edge) + (1.0 / 6.0) * np.sin(3.0 * psi_edge)
      ) / np.sin(psi_edge)
      f_psi = cx.field(
          np.broadcast_to(np.cos(4.0 * psi).astype(np.float32), patch.shape),
          patch,
      )
      cx.testing.assert_fields_allclose(
          patch.mean(f_psi, area_weighted=True),
          cx.field(np.float32(exact_band_mean)),
          atol=5e-4,
      )

    with self.subTest('horizon_and_altitude_validation'):
      with self.assertRaisesRegex(ValueError, 'exceeds geometric horizon'):
        coordinates.SwathPatch(
            scanlines=10,
            fov=96,
            max_scan_angle=75.0,
            satellite_altitude_km=833.0,
        )
      with self.assertRaisesRegex(
          ValueError, 'Underground satellites are not supported yet'
      ):
        coordinates.SwathPatch(
            scanlines=10,
            fov=96,
            max_scan_angle=45.0,
            satellite_altitude_km=-10.0,
        )

  def test_patch_coordinate_abc_and_reductions(self):
    with self.subTest('incomplete_subclass_raises_type_error'):

      class _IncompletePatch(coordinates.PatchCoordinate):
        pass

      with self.assertRaises(TypeError):
        _IncompletePatch()  # pyrefly: ignore[bad-instantiation]

    patch = coordinates.SwathPatch(
        scanlines=6,
        fov=4,
        max_scan_angle=52.725,
        satellite_altitude_km=833.0,
    )
    with self.subTest('mean_and_area_weighted_mean'):
      ones = cx.field(np.ones(patch.shape, dtype=np.float32), patch)
      cx.testing.assert_fields_allclose(
          patch.mean(ones, area_weighted=True),
          cx.field(np.float32(1.0)),
          rtol=1e-5,
      )
      with_nans = np.ones(patch.shape, dtype=np.float32)
      with_nans[3:, :] = np.nan
      f_nans = cx.field(with_nans, patch)
      cx.testing.assert_fields_allclose(
          patch.mean(f_nans, area_weighted=True, skipna=True),
          cx.field(np.float32(1.0)),
          rtol=1e-5,
      )
      mean_no_skip = patch.mean(f_nans, area_weighted=True, skipna=False)
      self.assertTrue(np.isnan(float(mean_no_skip.data)))
      # Unnormalized weights must be normalized by sum(w) even for skipna=False.
      unnorm_w = cx.field(np.full(patch.shape, 5.0, dtype=np.float32), patch)
      cx.testing.assert_fields_allclose(
          patch._reduce_mean(ones * 3.0, weights=unnorm_w, skipna=False),
          cx.field(np.float32(3.0)),
          rtol=1e-5,
      )

  def test_swath_patch_selection_and_xarray_roundtrip(self):
    patch = coordinates.SwathPatch.atms(scanlines=20)
    with self.subTest('isel_full_and_slice'):
      self.assertEqual(patch.isel(scanline=slice(None)), patch)
      sub = patch.isel(scanline=slice(0, 5), fov=0)
      self.assertEqual(sub.dims, ('scanline',))
      self.assertEqual(sub.shape, (5,))

    with self.subTest('xarray_roundtrip'):
      f = cx.field(np.zeros(patch.shape, dtype=np.float32), patch)
      da = f.to_xarray()
      restored = cx.Field.from_xarray(da, (coordinates.SwathPatch,))
      cx.testing.assert_fields_equal(restored, f)
      self.assertEqual(restored.coordinate, patch)

      # Missing attributes should not match SwathPatch.
      da_no_attrs = da.copy()
      da_no_attrs.coords['scanline'].attrs = {}
      da_no_attrs.coords['fov'].attrs = {}
      with self.assertRaises(ValueError):
        cx.Field.from_xarray(da_no_attrs, (coordinates.SwathPatch,))

  def test_lon_lat_patch_and_local_grid_patch(self):
    with self.subTest('lon_lat_patch'):
      ll_patch = coordinates.LonLatPatch(longitude_nodes=12, latitude_nodes=8)
      self.assertIsInstance(ll_patch, coordinates.PatchCoordinate)
      self.assertEqual(ll_patch.dims, ('patch_lon', 'patch_lat'))
      self.assertEqual(ll_patch.shape, (12, 8))
      ones = cx.field(np.ones(ll_patch.shape, dtype=np.float32), ll_patch)
      lats = cx.field(
          np.broadcast_to(
              np.linspace(-30.0, 30.0, 8, dtype=np.float32), ll_patch.shape
          ),
          ll_patch,
      )
      cx.testing.assert_fields_allclose(
          ll_patch.mean(ones, latitude=lats),
          cx.field(np.float32(1.0)),
          rtol=1e-5,
      )
      batch_ax = cx.SizedAxis('patch_index', 2)
      batched_coord = cx.coords.compose(batch_ax, ll_patch)
      batched_lats = cx.field(
          np.stack([lats.data, lats.data + 15.0], axis=0), batched_coord
      )
      batched_vals = cx.field(
          np.full(batched_coord.shape, 2.5, dtype=np.float32), batched_coord
      )
      cx.testing.assert_fields_allclose(
          ll_patch.mean(batched_vals, latitude=batched_lats, skipna=False),
          cx.field(np.full((2,), 2.5, dtype=np.float32), batch_ax),
          rtol=1e-5,
      )
      da = ones.to_xarray()
      restored = cx.Field.from_xarray(da, (coordinates.LonLatPatch,))
      cx.testing.assert_fields_equal(restored, ones)

    with self.subTest('local_grid_patch'):
      local_patch = coordinates.LocalGridPatch(
          x_nodes=9, y_nodes=7, dx_deg=0.5, dy_deg=0.5
      )
      self.assertIsInstance(local_patch, coordinates.PatchCoordinate)
      self.assertEqual(local_patch.dims, ('patch_x', 'patch_y'))
      self.assertEqual(local_patch.shape, (9, 7))
      x_axis, y_axis = local_patch.axes
      cx.testing.assert_fields_allclose(
          local_patch.fields['patch_x'],
          cx.field(np.linspace(-2.0, 2.0, 9, dtype=np.float32), x_axis),
      )
      cx.testing.assert_fields_allclose(
          local_patch.fields['patch_y'],
          cx.field(np.linspace(-1.5, 1.5, 7, dtype=np.float32), y_axis),
      )
      ones = cx.field(np.ones(local_patch.shape, dtype=np.float32), local_patch)
      cx.testing.assert_fields_allclose(
          local_patch.mean(ones, area_weighted=True),
          cx.field(np.float32(1.0)),
          rtol=1e-5,
      )
      da = ones.to_xarray()
      restored = cx.Field.from_xarray(da, (coordinates.LocalGridPatch,))
      cx.testing.assert_fields_equal(restored, ones)


if __name__ == '__main__':
  absltest.main()
