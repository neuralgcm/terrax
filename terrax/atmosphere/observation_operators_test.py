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

"""Tests for atmosphere-specific observation operators."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
import jax_datetime as jdt
import numpy as np
from terrax.atmosphere import observation_operators
from terrax.core import coordinates
from terrax.core import orographies
from terrax.core import spherical_harmonics
from terrax.core import units


class ObservationOperatorsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    n_sigma = 12
    self.ylm_map = spherical_harmonics.FixedYlmMapping(
        lon_lat_grid=coordinates.LonLatGrid.T21(),
        ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
    )
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    self.grid = coordinates.LonLatGrid.T21()
    self.in_sigma = coordinates.SigmaLevels.equidistant(n_sigma)
    self.source_coords = cx.coords.compose(self.in_sigma, self.ylm_grid)
    self.sim_units = units.DEFAULT_UNITS
    self.orography_module = orographies.ModalOrography(
        ylm_map=self.ylm_map,
        rngs=nnx.Rngs(0),
    )
    self.ref_temperatures = np.linspace(220, 250, num=n_sigma)
    zero_like = lambda c: cx.field(np.zeros(c.shape), c)
    self.prognostic_fields = {
        'divergence': zero_like(self.source_coords),
        'vorticity': zero_like(self.source_coords),
        'specific_humidity': zero_like(self.source_coords),
        'temperature': zero_like(self.source_coords),
        'log_surface_pressure': zero_like(self.ylm_grid),
        'time': cx.field(jdt.to_datetime('2001-01-01')),
    }

  def test_returns_pressure_level_outputs(self):
    pressure_levels = coordinates.PressureLevels.with_13_era5_levels()
    target_coords = cx.coords.compose(pressure_levels, self.grid)
    operator = observation_operators.StandardVariablesObservationOperator(
        ylm_map=self.ylm_map,
        orography=self.orography_module,
        levels=pressure_levels,
        sim_units=self.sim_units,
        observation_correction=None,
    )
    query = {
        'temperature': target_coords,
        'u_component_of_wind': target_coords,
        'specific_humidity': target_coords,
    }
    actual = operator.observe(inputs=self.prognostic_fields, query=query)
    for key in query:
      self.assertEqual(cx.get_coordinate(actual[key]), query[key])

  def test_returns_sigma_level_outputs(self):
    target_sigma_levels = coordinates.SigmaLevels.equidistant(10)
    target_coords = cx.coords.compose(target_sigma_levels, self.grid)
    operator = observation_operators.StandardVariablesObservationOperator(
        ylm_map=self.ylm_map,
        orography=self.orography_module,
        levels=target_sigma_levels,
        sim_units=self.sim_units,
        observation_correction=None,
    )
    query = {
        'temperature': target_coords,
        'u_component_of_wind': target_coords,
        'specific_humidity': target_coords,
    }
    actual = operator.observe(inputs=self.prognostic_fields, query=query)
    for key in query:
      self.assertEqual(cx.get_coordinate(actual[key]), query[key])

  def test_lapse_rate_below_ground_extrapolation_on_hybrid_levels(self):
    hybrid_levels = coordinates.HybridLevels.ecmwf137_interpolated(32)
    coords = cx.coords.compose(hybrid_levels, self.ylm_grid)
    sim_units = units.SI_UNITS
    t_ref, ps_ref = 250.0, 60000.0  # high terrain, surface at 600 hPa.
    nodal_t = np.full(hybrid_levels.shape + self.grid.shape, t_ref)
    nodal_t[-1] += 5.0  # strong gradient between the two lowest levels.
    nodal_t = cx.field(nodal_t, hybrid_levels, self.grid)
    nodal_lsp = cx.field(np.full(self.grid.shape, np.log(ps_ref)), self.grid)
    zeros = cx.field(np.zeros(coords.shape), coords)
    inputs = {
        'divergence': zeros,
        'vorticity': zeros,
        'specific_humidity': zeros,
        'temperature': self.ylm_map.to_modal(nodal_t),
        'log_surface_pressure': self.ylm_map.to_modal(nodal_lsp),
        'time': cx.field(jdt.to_datetime('2001-01-01')),
    }
    pressure_levels = coordinates.PressureLevels([300, 500, 850, 1000])
    target_coords = cx.coords.compose(pressure_levels, self.grid)
    query = {'temperature': target_coords, 'geopotential': target_coords}
    outputs = {}
    for mode in ['linear', 'lapse_rate']:
      operator = observation_operators.StandardVariablesObservationOperator(
          ylm_map=self.ylm_map,
          orography=self.orography_module,
          levels=pressure_levels,
          sim_units=sim_units,
          observation_correction=None,
          below_ground_extrapolation=mode,
      )
      outputs[mode] = operator.observe(inputs=inputs, query=query)
    for key in query:
      self.assertEqual(
          outputs['lapse_rate'][key].dims, outputs['linear'][key].dims
      )
    t_lin = outputs['linear']['temperature'].data
    t_lapse = outputs['lapse_rate']['temperature'].data
    # Above the lowest model level both methods agree.
    np.testing.assert_allclose(t_lin[:2], t_lapse[:2], rtol=1e-5)
    # Linear extrapolation is dominated by the lowest-levels gradient.
    self.assertGreater(np.abs(t_lin[-1] - t_ref).max(), 100.0)
    # Lapse-rate extrapolation follows the standard atmosphere profile.
    p_low = hybrid_levels.a_boundaries[-2:].mean() * 100 + (
        hybrid_levels.b_boundaries[-2:].mean() * ps_ref
    )
    exponent = 287.0 * 0.0065 / 9.80616
    expected = (t_ref + 5.0) * (100000.0 / p_low) ** exponent
    np.testing.assert_allclose(t_lapse[-1], expected, rtol=5e-3)
    self.assertTrue(np.all(np.isfinite(
        outputs['lapse_rate']['geopotential'].data)))
    # Geopotential decreases below ground (higher pressure -> lower height).
    z_lapse = outputs['lapse_rate']['geopotential'].data
    self.assertTrue(np.all(z_lapse[-1] < z_lapse[-2]))


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
