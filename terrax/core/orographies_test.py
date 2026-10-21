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
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates
from terrax.core import orographies
from terrax.core import parallelism
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


def _make_ylm_map(padded: bool = False) -> spherical_harmonics.FixedYlmMapping:
  if padded:
    spmd_mesh = jax.sharding.Mesh(
        devices=np.array(jax.devices()).reshape((2, 2, 2)),
        axis_names=['z', 'x', 'y'],
    )
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={
            'spatial': {
                'longitude': 'x',
                'latitude': 'y',
                'longitude_wavenumber': 'x',
                'total_wavenumber': 'y',
                'level': 'z',
            }
        },
    )
    schema_key = 'spatial'
  else:
    mesh = parallelism.default_mesh()
    schema_key = None
  return spherical_harmonics.FixedYlmMapping(
      lon_lat_grid=coordinates.LonLatGrid.T21(),
      ylm_grid=coordinates.SphericalHarmonicGrid.T21(
          mesh=mesh, partition_schema_key=schema_key
      ),
      mesh=mesh,
      partition_schema_key=schema_key,
  )


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


class ModalOrographyTest(parameterized.TestCase):

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

  def test_mem_layout_shapes_and_padding_invariance(self):
    unpadded_map = _make_ylm_map(padded=False)
    padded_map = _make_ylm_map(padded=True)
    unpadded_grid = unpadded_map.ylm_grid
    padded_grid = padded_map.ylm_grid
    self.assertNotEqual(padded_grid.wavenumber_padding, (0, 0))
    num_modes = int(unpadded_grid.fields['mask'].data.sum())

    orography_modes = orographies.ModalOrographyWithCorrection(
        ylm_map=padded_map,
        correction_scale=0.1,
        mem_layout='modes',
        rngs=nnx.Rngs(0),
    )
    orography_2d = orographies.ModalOrographyWithCorrection(
        ylm_map=padded_map,
        correction_scale=0.1,
        mem_layout='2d',
        rngs=nnx.Rngs(0),
    )

    self.assertEqual(orography_modes.orography.shape, (num_modes,))
    self.assertEqual(orography_modes.correction.shape, (num_modes,))
    self.assertEqual(orography_2d.orography.shape, unpadded_grid.shape)
    self.assertEqual(orography_2d.correction.shape, unpadded_grid.shape)
    cx.testing.assert_field_properties(
        orography_2d.modal_orography,
        dims=padded_grid.dims,
        shape=padded_grid.shape,
    )
    self.assertEqual(orography_2d.modal_orography.coordinate, padded_grid)

  @parameterized.parameters((False,), (True,))
  def test_modes_and_2d_equivalence_and_gradients(self, padded):
    ylm_map = _make_ylm_map(padded)
    orography_modes = orographies.ModalOrographyWithCorrection(
        ylm_map=ylm_map,
        correction_scale=0.25,
        mem_layout='modes',
        rngs=nnx.Rngs(0),
    )
    orography_2d = orographies.ModalOrographyWithCorrection(
        ylm_map=ylm_map,
        correction_scale=0.25,
        mem_layout='2d',
        rngs=nnx.Rngs(0),
    )

    unpadded_grid = _make_ylm_map(padded=False).ylm_grid
    unpadded_mask = unpadded_grid.fields['mask'].data
    rng = np.random.default_rng(42)
    base_2d = rng.normal(size=unpadded_grid.shape).astype(np.float32)
    corr_2d = rng.normal(size=unpadded_grid.shape).astype(np.float32)

    orography_2d.orography.set_value(jnp.asarray(base_2d))
    orography_2d.correction.set_value(jnp.asarray(corr_2d))
    orography_modes.orography.set_value(jnp.asarray(base_2d[unpadded_mask]))
    orography_modes.correction.set_value(jnp.asarray(corr_2d[unpadded_mask]))

    cx.testing.assert_fields_allclose(
        orography_2d.modal_orography,
        orography_modes.modal_orography,
        rtol=1e-6,
    )
    cx.testing.assert_fields_allclose(
        orography_2d.nodal_orography,
        orography_modes.nodal_orography,
        rtol=1e-6,
    )

    def loss_fn(module):
      modal = cx.cmap(jnp.sin)(module.modal_orography)
      return cx.cmap(jnp.sum)(modal.untag(ylm_map.ylm_grid)).data

    loss_modes, grads_modes = nnx.jit(nnx.value_and_grad(loss_fn))(
        orography_modes
    )
    loss_2d, grads_2d = nnx.jit(nnx.value_and_grad(loss_fn))(orography_2d)

    np.testing.assert_allclose(loss_2d, loss_modes, rtol=1e-6)
    np.testing.assert_allclose(
        grads_2d.correction.get_value()[unpadded_mask],
        grads_modes.correction.get_value(),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        grads_2d.correction.get_value()[~unpadded_mask], 0.0
    )

  @parameterized.parameters((False,), (True,))
  def test_update_from_xarray_matches_across_layouts(self, padded):
    ylm_map = _make_ylm_map(padded)
    data_ylm_map = _make_ylm_map(padded=False)
    grid = data_ylm_map.nodal_grid
    rng = np.random.default_rng(7)
    geopotential = rng.uniform(0.0, 1000.0, size=grid.shape).astype(np.float32)
    ds = _dataset_on_grid('geopotential_at_surface', geopotential, grid)

    orography_modes = orographies.ModalOrography(
        ylm_map=ylm_map, mem_layout='modes', rngs=nnx.Rngs(0)
    )
    orography_2d = orographies.ModalOrography(
        ylm_map=ylm_map, mem_layout='2d', rngs=nnx.Rngs(0)
    )
    kwargs = {
        'data_ylm_map': data_ylm_map,
        'sim_units': units.DEFAULT_UNITS,
    }
    orography_modes.update_from_xarray(ds, **kwargs)
    orography_2d.update_from_xarray(ds, **kwargs)

    cx.testing.assert_fields_allclose(
        orography_2d.modal_orography,
        orography_modes.modal_orography,
        rtol=1e-6,
    )
    cx.testing.assert_fields_allclose(
        orography_2d.nodal_orography,
        orography_modes.nodal_orography,
        rtol=1e-6,
    )


if __name__ == '__main__':
  chex.set_n_cpu_devices(8)
  jax.config.parse_flags_with_absl()
  absltest.main()
