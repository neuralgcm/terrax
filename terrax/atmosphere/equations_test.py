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

"""Tests for atmosphere-specific equations and helpers."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from coordax import testing as cx_testing
from dinosaur import primitive_equations
from dinosaur import time_integration
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.atmosphere import equations
from terrax.atmosphere import idealized_states
from terrax.core import coordinates
from terrax.core import equations as core_equations
from terrax.core import orographies
from terrax.core import spherical_harmonics
from terrax.core import time_integrators
from terrax.core import units


class AtmosphereEquationsAndHelpersTests(parameterized.TestCase):

  def test_temperature_linearization_roundtrip_with_grid(self):
    """Tests linearization and delinearization with LonLatGrid."""
    levels = coordinates.SigmaLevels.equidistant(5)
    ref_temp_field = cx.field(280.0 + np.arange(levels.shape[0]), levels)
    grid = coordinates.LonLatGrid.T21()
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    rng = np.random.RandomState(42)

    temperature = cx.field(rng.randn(*levels.shape, *grid.shape), levels, grid)
    inputs = {
        'temperature': temperature,
        'time': cx.field(jdt.to_datetime('2025-01-01')),
        'sh': cx.field(rng.randn(*ylm_grid.shape), ylm_grid),
    }

    linearize = equations.get_temperature_linearization_transform(
        ref_temperatures=ref_temp_field
    )
    actual_linearized = linearize(inputs)

    with self.subTest('direct'):
      self.assertNotIn('temperature', actual_linearized)
      self.assertIn('temperature_variation', actual_linearized)
      # check that other fields are untouched.
      expected_others = {k: v for k, v in inputs.items() if k != 'temperature'}
      actual_others = {
          k: v
          for k, v in actual_linearized.items()
          if k != 'temperature_variation'
      }
      chex.assert_trees_all_equal(actual_others, expected_others)
      expected_variation = temperature - ref_temp_field
      cx_testing.assert_fields_allclose(
          actual_linearized['temperature_variation'], expected_variation
      )

    with self.subTest('roundtrip'):
      delinearize = equations.get_temperature_delinearization_transform(
          ref_temperatures=ref_temp_field
      )
      actual_roundtrip = delinearize(actual_linearized)
      expected = jax.tree.map(jnp.asarray, inputs)
      chex.assert_trees_all_close(actual_roundtrip, expected, atol=5e-5)

  def test_temperature_linearization_roundtrip_with_ylm_grid(self):
    """Tests linearization and delinearization with SphericalHarmonicGrid."""
    levels = coordinates.SigmaLevels.equidistant(5)
    ref_temp_field = cx.field(280.0 + np.arange(levels.shape[0]), levels)
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    grid = coordinates.LonLatGrid.T21()
    rng = np.random.RandomState(42)

    temperature = cx.field(
        rng.randn(*levels.shape, *ylm_grid.shape), levels, ylm_grid
    )
    inputs = {
        'temperature': temperature,
        'time': cx.field(jdt.to_datetime('2025-01-01')),
        'lonlat': cx.field(rng.randn(*grid.shape), grid),
    }

    linearize = equations.get_temperature_linearization_transform(
        ref_temperatures=ref_temp_field
    )
    actual_linearized = linearize(inputs)

    with self.subTest('direct'):
      self.assertNotIn('temperature', actual_linearized)
      self.assertIn('temperature_variation', actual_linearized)
      # check that other fields are untouched.
      expected_others = {k: v for k, v in inputs.items() if k != 'temperature'}
      actual_others = {
          k: v
          for k, v in actual_linearized.items()
          if k != 'temperature_variation'
      }
      chex.assert_trees_all_equal(actual_others, expected_others)

    with self.subTest('roundtrip'):
      delinearize = equations.get_temperature_delinearization_transform(
          ref_temperatures=ref_temp_field
      )
      actual_roundtrip = delinearize(actual_linearized)
      expected = jax.tree.map(jnp.asarray, inputs)
      chex.assert_trees_all_close(actual_roundtrip, expected, atol=5e-5)


def _relative_l2(x: cx.Field, y: cx.Field) -> float:
  x, y = x.data, y.data
  return float(np.sqrt(np.square(x - y).sum() / np.square(y).sum()))


class PrimitiveEquationsTest(parameterized.TestCase):
  """Tests for PrimitiveEquations and SemiLagrangianPrimitiveEquations."""

  def setUp(self):
    super().setUp()
    self.grid = coordinates.LonLatGrid.T21()
    self.ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    self.ylm_map = spherical_harmonics.FixedYlmMapping(self.grid, self.ylm_grid)
    self.sim_units = units.DEFAULT_UNITS

  def _setup(
      self,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      dycore_cls: type[
          equations.PrimitiveEquations
          | equations.SemiLagrangianPrimitiveEquations
      ] = equations.PrimitiveEquations,
      **dycore_kwargs,
  ):
    """Returns JW initial state with a tracer and primitive equations."""
    state = idealized_states.perturbed_jw(
        self.ylm_map, levels, jax.random.key(0), self.sim_units
    )
    ref_temperatures = state.pop('ref_temperatures')
    nodal_orography = state.pop('orography')
    del state['geopotential']
    nodal_tracers = dycore_kwargs.get('nodal_tracers', ())
    tracer = 1e-3 * self.ylm_map.to_nodal(state['temperature'])
    if 'tracer' not in nodal_tracers:
      tracer = self.ylm_map.to_modal(tracer)
    state['tracer'] = tracer
    mask = self.ylm_grid.fields['mask'].data
    modal_orography = self.ylm_map.to_modal(nodal_orography).data[mask]
    orography = orographies.ModalOrography(
        ylm_map=self.ylm_map,
        initializer=lambda *args, **kwargs: modal_orography,
        rngs=nnx.Rngs(0),
    )
    equation = dycore_cls(
        ylm_map=self.ylm_map,
        levels=levels,
        sim_units=self.sim_units,
        reference_temperatures=ref_temperatures,
        tracer_names=('tracer',),
        orography_module=orography,
        **dycore_kwargs,
    )
    return state, equation

  def _nondim_minutes(self, minutes: float) -> float:
    return self.sim_units.nondimensionalize_timedelta64(
        np.timedelta64(minutes, 'm')
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma',
          levels=coordinates.SigmaLevels.equidistant(8),
          expected_cls=primitive_equations.PrimitiveEquationsSigma,
      ),
      dict(
          testcase_name='hybrid',
          levels=coordinates.HybridLevels.with_n_levels(8),
          expected_cls=primitive_equations.PrimitiveEquationsHybrid,
      ),
  )
  def test_eulerian_primitive_equations(self, levels, expected_cls):
    state, equation = self._setup(levels)
    self.assertIsInstance(equation, time_integrators.ImplicitExplicitODE)
    self.assertNotIsInstance(
        equation, time_integrators.SemiLagrangianImplicitExplicitODE
    )
    self.assertIs(type(equation.primitive_equation), expected_cls)
    tendencies = equation.explicit_terms(state)
    self.assertEqual(set(tendencies.keys()), set(state.keys()))
    self.assertFalse(hasattr(equation, 'nonadvective_terms'))

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma',
          levels=coordinates.SigmaLevels.equidistant(8),
          expected_cls=primitive_equations.SemiLagrangianPrimitiveEquations,
      ),
      dict(
          testcase_name='hybrid',
          levels=coordinates.HybridLevels.with_n_levels(8),
          expected_cls=(
              primitive_equations.SemiLagrangianPrimitiveEquationsHybrid
          ),
      ),
  )
  def test_semi_lagrangian_primitive_equations(self, levels, expected_cls):
    state, equation = self._setup(
        levels,
        dycore_cls=equations.SemiLagrangianPrimitiveEquations,
        vertical_interpolation_order='cubic',
        monotone_tracers=True,
    )
    self.assertIsInstance(
        equation, time_integrators.SemiLagrangianImplicitExplicitODE
    )
    self.assertNotIsInstance(equation, time_integrators.ImplicitExplicitODE)
    dinosaur_equation = equation.primitive_equation
    self.assertIs(type(dinosaur_equation), expected_cls)
    self.assertEqual(dinosaur_equation.vertical_interpolation_order, 'cubic')
    self.assertTrue(dinosaur_equation.monotone_tracers)
    with self.assertRaisesRegex(TypeError, 'semi-Lagrangian equation'):
      equation.explicit_terms(state)
    with self.assertRaisesRegex(ValueError, 'ImplicitExplicitODE'):
      core_equations.ComposedODE([equation])  # pyrefly: ignore[bad-argument-type]

  def test_unknown_nodal_tracers_raises(self):
    with self.assertRaisesRegex(ValueError, 'nodal_tracers'):
      self._setup(
          coordinates.SigmaLevels.equidistant(8),
          dycore_cls=equations.SemiLagrangianPrimitiveEquations,
          nodal_tracers=['unknown'],
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma',
          levels=coordinates.SigmaLevels.equidistant(8),
          nodal_tracers=(),
      ),
      dict(
          testcase_name='hybrid',
          levels=coordinates.HybridLevels.with_n_levels(8),
          nodal_tracers=(),
      ),
      dict(
          testcase_name='sigma_nodal_tracer',
          levels=coordinates.SigmaLevels.equidistant(8),
          nodal_tracers=('tracer',),
      ),
      dict(
          testcase_name='hybrid_nodal_tracer',
          levels=coordinates.HybridLevels.with_n_levels(8),
          nodal_tracers=('tracer',),
      ),
  )
  def test_semi_lagrangian_step_matches_dinosaur(self, levels, nodal_tracers):
    state, equation = self._setup(
        levels,
        dycore_cls=equations.SemiLagrangianPrimitiveEquations,
        nodal_tracers=nodal_tracers,
    )
    dt = self._nondim_minutes(30)
    integrator = time_integrators.SemiLagrangianCrankNicolsonRK2(
        equation, dt, off_centering=0.1
    )
    actual = nnx.jit(lambda integrator, x: integrator(x))(integrator, state)

    dinosaur_step = time_integration.semi_lagrangian_crank_nicolson_rk2(
        equation.primitive_equation, dt, off_centering=0.1
    )
    dinosaur_state = equation._to_primitive_equations_state(state)
    expected = equation._from_primitive_equations_state(
        jax.jit(dinosaur_step)(dinosaur_state), is_tendency=False
    )
    with self.subTest('coordinates'):
      actual_coords = {k: v.coordinate for k, v in actual.items()}
      expected_coords = {k: v.coordinate for k, v in state.items()}
      self.assertEqual(actual_coords, expected_coords)
    with self.subTest('values'):
      # terrax carries absolute temperature (O(1e3) l=0 coefficient) between
      # stages, so float32 roundoff differs slightly from the dinosaur path
      # that carries temperature variations.
      chex.assert_trees_all_close(actual, expected, rtol=1e-5, atol=1e-5)
    with self.subTest('state_evolved'):
      self.assertGreater(
          _relative_l2(actual['vorticity'], state['vorticity']), 1e-6
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma',
          levels=coordinates.SigmaLevels.equidistant(8),
      ),
      dict(
          testcase_name='hybrid',
          levels=coordinates.HybridLevels.with_n_levels(8),
      ),
  )
  def test_composed_with_held_suarez(self, levels):
    state, sl_pe = self._setup(
        levels, dycore_cls=equations.SemiLagrangianPrimitiveEquations
    )
    hs = equations.HeldSuarezForcing(
        ylm_map=self.ylm_map,
        levels=levels,
        sim_units=self.sim_units,
        reference_temperatures=sl_pe.t_ref_tuple,
    )
    composed = core_equations.compose_equations([sl_pe, hs])
    self.assertIsInstance(composed, core_equations.ComposedSemiLagrangianODE)

    with self.subTest('compose_equations_validation'):
      _, eulerian_pe = self._setup(levels)
      self.assertIsInstance(
          core_equations.compose_equations([eulerian_pe, hs]),
          core_equations.ComposedODE,
      )
      self.assertIsInstance(
          core_equations.compose_equations([hs]),
          core_equations.ComposedExplicitODE,
      )
      with self.assertRaisesRegex(ValueError, 'mixing'):
        core_equations.compose_equations([sl_pe, eulerian_pe])

    with self.subTest('nonadvective_terms'):
      actual = composed.nonadvective_terms(state)
      pe_terms = sl_pe.nonadvective_terms(state)
      hs_terms = hs.explicit_terms(state)
      expected = {k: v + hs_terms.get(k, 0.0) for k, v in pe_terms.items()}
      chex.assert_trees_all_close(actual, expected, rtol=1e-6, atol=1e-6)

    with self.subTest('step'):
      dt = self._nondim_minutes(30)
      integrator = time_integrators.SemiLagrangianCrankNicolsonRK2(composed, dt)
      step_fn = nnx.jit(lambda integrator, x: integrator(x))
      next_state = step_fn(integrator, state)
      self.assertEqual(set(next_state.keys()), set(state.keys()))
      for k, v in next_state.items():
        self.assertTrue(np.all(np.isfinite(v.data)), msg=f'{k} not finite')

  @parameterized.named_parameters(
      dict(
          testcase_name='sigma',
          levels=coordinates.SigmaLevels.equidistant(8),
      ),
      dict(
          testcase_name='hybrid',
          levels=coordinates.HybridLevels.with_n_levels(8),
      ),
  )
  def test_semi_lagrangian_consistent_with_eulerian(self, levels):
    """SL at dt=30min tracks the Eulerian solver at dt=10min over 6 hours."""
    state, sl_equation = self._setup(
        levels, dycore_cls=equations.SemiLagrangianPrimitiveEquations
    )
    _, eulerian_equation = self._setup(levels)

    @nnx.jit(static_argnums=2)
    def run(integrator, x, steps):
      return jax.lax.fori_loop(0, steps, lambda _, x: integrator(x), x)

    sl_integrator = time_integrators.SemiLagrangianCrankNicolsonRK2(
        sl_equation, self._nondim_minutes(30)
    )
    eulerian_integrator = time_integrators.ImexRk3Sil(
        eulerian_equation, self._nondim_minutes(10)
    )
    sl_final = sl_equation.linearize_transform(run(sl_integrator, state, 12))
    eulerian_final = eulerian_equation.linearize_transform(
        run(eulerian_integrator, state, 36)
    )
    to_nodal = self.ylm_map.to_nodal
    # Divergence is excluded: it is dominated by transient gravity waves from
    # the initial perturbation, which the two schemes treat differently.
    # Measured (sigma, hybrid): T' 4.0e-4, 4.6e-4; ln(ps) 3.1e-5, 3.0e-5;
    # vorticity 4.2e-3, 4.9e-3; tracer 1.9e-4, 2.5e-4.
    tolerances = {
        'temperature_variation': 1.5e-3,
        'log_surface_pressure': 1e-4,
        'vorticity': 1.5e-2,
        'tracer': 1e-3,
    }
    for k, tol in tolerances.items():
      with self.subTest(k):
        l2 = _relative_l2(to_nodal(sl_final[k]), to_nodal(eulerian_final[k]))
        self.assertLess(l2, tol)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
