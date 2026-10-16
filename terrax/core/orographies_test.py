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
"""Tests for modules that hold orographic data."""

from absl.testing import absltest
import coordax as cx
from flax import nnx
import jax
import numpy as np
from terrax.core import coordinates
from terrax.core import orographies
from terrax.core import scales
from terrax.core import spherical_harmonics
from terrax.core import units
import xarray


def _dataset_on_grid(
    name: str,
    values: np.ndarray,
    grid: coordinates.LonLatGrid,
    **attrs: str,
) -> xarray.Dataset:
  data_array = cx.field(values, grid).to_xarray().assign_attrs(attrs)
  return xarray.Dataset({name: data_array})


class OrographyTest(absltest.TestCase):

  def test_update_from_xarray_uses_units_attribute(self):
    grid = coordinates.LonLatGrid.T21()
    sim_units = units.DEFAULT_UNITS
    dataset = _dataset_on_grid(
        'orography', np.full(grid.shape, 1.5), grid, units='km'
    )
    orography = orographies.Orography(grid=grid, rngs=nnx.Rngs(0))

    orography.update_from_xarray(dataset, sim_units=sim_units)

    expected = sim_units.nondimensionalize(1500 * scales.units.meter)
    np.testing.assert_allclose(
        orography.nodal_orography.data, expected, rtol=1e-6
    )

  def test_update_from_xarray_derives_orography_from_geopotential(self):
    grid = coordinates.LonLatGrid.T21()
    sim_units = units.DEFAULT_UNITS
    geopotential = np.full(grid.shape, 1000.0)
    geopotential *= scales.GRAVITY_ACCELERATION.magnitude
    dataset = _dataset_on_grid('geopotential_at_surface', geopotential, grid)
    orography = orographies.Orography(grid=grid, rngs=nnx.Rngs(0))

    orography.update_from_xarray(dataset, sim_units=sim_units)

    expected = sim_units.nondimensionalize(1000 * scales.units.meter)
    np.testing.assert_allclose(
        orography.nodal_orography.data, expected, rtol=1e-6
    )
    self.assertNotIn('orography', dataset)  # input dataset is not modified.

  def test_update_from_xarray_raises_on_grid_mismatch(self):
    data_grid = coordinates.LonLatGrid.T42()
    dataset = _dataset_on_grid(
        'orography', np.zeros(data_grid.shape), data_grid
    )
    grid = coordinates.LonLatGrid.T21()
    orography = orographies.Orography(grid=grid, rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'does not match'):
      orography.update_from_xarray(dataset, sim_units=units.DEFAULT_UNITS)


class ModalOrographyTest(absltest.TestCase):

  def test_update_from_xarray(self):
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    grid = ylm_map.nodal_grid
    sim_units = units.DEFAULT_UNITS
    dataset = _dataset_on_grid(
        'orography', np.full(grid.shape, 1000.0), grid, units='m'
    )
    # Orography is transposed to the order of `grid` dimensions.
    dataset = dataset.transpose('latitude', 'longitude')
    orography = orographies.ModalOrography(ylm_map=ylm_map, rngs=nnx.Rngs(0))

    orography.update_from_xarray(
        dataset, data_ylm_map=ylm_map, sim_units=sim_units
    )

    expected = sim_units.nondimensionalize(1000 * scales.units.meter)
    np.testing.assert_allclose(
        orography.nodal_orography.data, expected, rtol=1e-5
    )

  def test_update_from_xarray_raises_on_grid_mismatch(self):
    ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    data_grid = coordinates.LonLatGrid.T42()
    dataset = _dataset_on_grid(
        'orography', np.zeros(data_grid.shape), data_grid
    )
    orography = orographies.ModalOrography(ylm_map=ylm_map, rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'does not match'):
      orography.update_from_xarray(
          dataset, data_ylm_map=ylm_map, sim_units=units.DEFAULT_UNITS
      )


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
