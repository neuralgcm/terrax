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

"""Tests for statistic rescaling schemes in terrax.metrics.scaling."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.core import coordinates
from terrax.metrics import scaling


class CoordinateMaskScalerTest(parameterized.TestCase):

  def test_coordinate_mask_scaler(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_scaler = scaling.CoordinateMaskScaler(
        mask_coord=mask_coord, masked_value=0.0, unmasked_value=5.0
    )
    scales = mask_scaler.scales(field)
    expected_scales = np.array([5.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(scales.data, expected_scales)

  def test_coordinate_mask_scaler_with_context(self):
    x = cx.SizedAxis('x', 4)
    field = cx.field(np.ones(x.shape), x)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_scaler = scaling.CoordinateMaskScaler(mask_coord=mask_coord)

    context_time_match = {'timedelta': cx.field(np.timedelta64(6, 'h'))}
    scales_match = mask_scaler.scales(field, context=context_time_match)
    np.testing.assert_allclose(scales_match.data, 0.0)

    context_time_no_match = {'timedelta': cx.field(np.timedelta64(3, 'h'))}
    scales_no_match = mask_scaler.scales(field, context=context_time_no_match)
    np.testing.assert_allclose(scales_no_match.data, 1.0)

  def test_missing_dimension_raises_error(self):
    """Tests that a missing dimension raises a ValueError."""
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    mask_coord = cx.SizedAxis('nondim', 2)
    with self.assertRaisesRegex(
        ValueError, "Coordinate for 'nondim' not found"
    ):
      scaler = scaling.CoordinateMaskScaler(
          mask_coord=mask_coord, skip_missing=False
      )
      scaler.scales(field)


class GeneralizedLeadTimeScalerTest(parameterized.TestCase):

  def test_normalization(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 3, 8, 15]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.GeneralizedLeadTimeScaler(base_squared_error_in_hours=1.0)
    scales = scaler.scales(field)
    np.testing.assert_allclose(scales.data.mean(), 1.0, atol=1e-6)

  def test_normalization_with_weights_power(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 24]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=4.0, weights_power=2.0
    )
    scales = scaler.scales(field)
    np.testing.assert_allclose(scales.data.mean(), 1.0, atol=1e-6)

  def test_normalization_in_context(self):
    x = cx.SizedAxis('x', 1)
    field = cx.field(np.ones(x.shape), x)
    scaler = scaling.GeneralizedLeadTimeScaler(base_squared_error_in_hours=7.0)
    step_delta = jdt.Timedelta.from_timedelta64(np.timedelta64(6, 'h'))
    n_steps = 4
    scaled_weights = []
    for i in range(n_steps):
      ctx = {
          'timedelta': cx.field(step_delta * i),
          'times': cx.field(step_delta * jnp.arange(n_steps)),
      }
      scaled_weights.append(scaler.scales(field, context=ctx).data)
    scaled_weights = np.array(scaled_weights)
    np.testing.assert_allclose(scaled_weights.mean(), 1.0, atol=1e-6)

  def test_asymptotic_normalization(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)

    # Case 1: max_t = 18. T_asymp = 18. Ratio = 1.
    # asymptotic_norm = 0.5.
    # expected_scale = (1 + 0.5 * 1) / (1 + 1) = 0.75
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=1.0,
        asymptotic_squared_error_in_hours=240.0,
        asymptotic_norm=0.5,
        norm_transition_timescale_in_hours=18.0,
    )
    scales = scaler.scales(field)
    np.testing.assert_allclose(scales.data.mean(), 0.75, atol=1e-6)

    # Case 2: max_t >> T_asymp
    # max_t = 1800, T_asymp = 18. Ratio = 100.
    # asymptotic_norm = 0.5.
    # expected ~ (1 + 50) / 101 ~ 0.505
    time_coord_long = coordinates.TimeDelta(
        np.linspace(0, 1800, 100) * np.timedelta64(1, 'h')
    )
    field_long = cx.field(np.ones(time_coord_long.shape), time_coord_long)
    scales_long = scaler.scales(field_long)
    expected_long = (1 + 0.5 * (1800 / 18)) / (1 + 1800 / 18)
    np.testing.assert_allclose(
        scales_long.data.mean(), expected_long, atol=1e-6
    )

    # Case 3: max_t << T_asymp
    # max_t = 0.18. Ratio = 0.01.
    # expected ~ (1 + 0.005) / 1.01 ~ 0.995
    time_coord_short = coordinates.TimeDelta(
        np.array([0, 6]) * np.timedelta64(1, 'm')
    )
    field_short = cx.field(np.ones(time_coord_short.shape), time_coord_short)
    scales_short = scaler.scales(field_short)
    expected_short = (1 + 0.5 * (0.1 / 18)) / (1 + 0.1 / 18)
    np.testing.assert_allclose(
        scales_short.data.mean(), expected_short, atol=1e-6
    )

  def test_asymptotic_normalization_with_power(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=1.0,
        asymptotic_norm=0.5,
        norm_transition_power=2.0,
        norm_transition_timescale_in_hours=18.0,
    )
    scales = scaler.scales(field)
    # Ratio = (18/18)**2 = 1.
    # expected = (1 + 0.5 * 1) / (1 + 1) = 0.75
    np.testing.assert_allclose(scales.data.mean(), 0.75, atol=1e-6)

    # Check with max_t = 36. Ratio = (36/18)**2 = 4.
    # expected = (1 + 0.5 * 4) / (1 + 4) = 3 / 5 = 0.6
    time_coord_long = coordinates.TimeDelta(
        np.array([0, 36]) * np.timedelta64(1, 'h')
    )
    field_long = cx.field(np.ones(time_coord_long.shape), time_coord_long)
    scales_long = scaler.scales(field_long)
    np.testing.assert_allclose(scales_long.data.mean(), 0.6, atol=1e-6)

  def test_asymptotic_normalization_raises_error_when_missing_timescale(self):
    with self.assertRaisesRegex(
        ValueError, 'norm_transition_timescale_in_hours'
    ):
      f = cx.field(np.zeros(1), coordinates.TimeDelta([np.timedelta64(1, 'h')]))
      scaling.GeneralizedLeadTimeScaler(
          base_squared_error_in_hours=1.0, asymptotic_norm=0.5
      ).scales(f)


class SigmoidWavenumberScalerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='number',
          ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
          cutoff_wavenumber=18,
          cutoff_fraction=None,
          expected_cutoff=18,
      ),
      dict(
          testcase_name='dict',
          ylm_grid=coordinates.SphericalHarmonicGrid.TL63(),
          cutoff_wavenumber={coordinates.SphericalHarmonicGrid.TL63(): 50},
          cutoff_fraction=None,
          expected_cutoff=50,
      ),
      dict(
          testcase_name='dict_matches_padded_grid',
          ylm_grid=coordinates.SphericalHarmonicGrid(
              longitude_wavenumbers=22,
              total_wavenumbers=23,
              wavenumber_padding=(2, 1),
          ),
          cutoff_wavenumber={coordinates.SphericalHarmonicGrid.T21(): 18},
          cutoff_fraction=None,
          expected_cutoff=18,
      ),
      dict(
          testcase_name='fraction',
          ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
          cutoff_wavenumber=None,
          cutoff_fraction=0.75,
          expected_cutoff=0.75 * 21,
      ),
      dict(
          testcase_name='dict_fallback_to_fraction',
          ylm_grid=coordinates.SphericalHarmonicGrid.T21(),
          cutoff_wavenumber={coordinates.SphericalHarmonicGrid.TL63(): 50},
          cutoff_fraction=0.75,
          expected_cutoff=0.75 * 21,
      ),
  )
  def test_sigmoid_profile(
      self, ylm_grid, cutoff_wavenumber, cutoff_fraction, expected_cutoff
  ):
    field = cx.field(np.ones(ylm_grid.shape), ylm_grid)
    scaler = scaling.SigmoidWavenumberScaler(
        cutoff_wavenumber=cutoff_wavenumber, cutoff_fraction=cutoff_fraction
    )
    scales = scaler.scales(field)
    ls = ylm_grid.fields['total_wavenumber']
    mask = ylm_grid.fields['mask']
    zeros = cx.field(np.zeros(ylm_grid.shape), ylm_grid)
    # Coordinates match `ylm_grid` and scales vanish for l >= l_cutoff.
    cx.testing.assert_fields_allclose(scales * (ls >= expected_cutoff), zeros)
    # Scales vanish on padded modes.
    cx.testing.assert_fields_allclose(scales * ~mask, zeros)
    # Scale at l = 0 is 1.0 and decreases monotonically with l.
    m0_scales = scales.isel(longitude_wavenumber=0)
    cx.testing.assert_fields_allclose(
        m0_scales.isel(total_wavenumber=0), cx.field(1.0)
    )
    self.assertTrue(np.all(np.diff(m0_scales.data) <= 1e-6))

  def test_missing_from_dict_without_fraction_raises(self):
    ylm_grid = coordinates.SphericalHarmonicGrid.T21()
    field = cx.field(np.ones(ylm_grid.shape), ylm_grid)
    scaler = scaling.SigmoidWavenumberScaler(
        cutoff_wavenumber={coordinates.SphericalHarmonicGrid.TL63(): 50}
    )
    with self.assertRaisesRegex(ValueError, 'not found in'):
      scaler.scales(field)

  @parameterized.named_parameters(
      dict(
          testcase_name='nothing_set',
          cutoff_wavenumber=None,
          cutoff_fraction=None,
          regex='At least one of',
      ),
      dict(
          testcase_name='number_and_fraction',
          cutoff_wavenumber=18,
          cutoff_fraction=0.75,
          regex='only used as a fallback',
      ),
  )
  def test_invalid_cutoff_raises(
      self, cutoff_wavenumber, cutoff_fraction, regex
  ):
    with self.assertRaisesRegex(ValueError, regex):
      scaling.SigmoidWavenumberScaler(
          cutoff_wavenumber=cutoff_wavenumber, cutoff_fraction=cutoff_fraction
      )

  def test_skip_missing_and_error(self):
    grid = coordinates.LonLatGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    scaler_skip = scaling.SigmoidWavenumberScaler(
        cutoff_wavenumber=18, skip_missing=True
    )
    cx.testing.assert_fields_allclose(scaler_skip.scales(field), cx.field(1.0))

    scaler_no_skip = scaling.SigmoidWavenumberScaler(
        cutoff_wavenumber=18, skip_missing=False
    )
    with self.assertRaisesRegex(ValueError, 'No SphericalHarmonicGrid'):
      scaler_no_skip.scales(field)


class LeadTimeScalerTest(parameterized.TestCase):

  def test_lead_time_scaler(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    scaler = scaling.LeadTimeScaler(
        base_squared_error_in_hours=6.0, normalize_weights=False
    )
    scales = scaler.scales(field)
    expected = cx.field(
        1.0 / np.sqrt(np.array([1.0, 2.0, 3.0, 4.0])), time_coord
    )
    cx.testing.assert_fields_allclose(scales, expected, atol=1e-6)

  def test_lead_time_scaler_with_context(self):
    x = cx.SizedAxis('x', 1)
    field = cx.field(np.ones(x.shape), x)
    scaler = scaling.LeadTimeScaler(
        base_squared_error_in_hours=6.0, normalize_weights=True
    )
    step_delta = jdt.Timedelta.from_timedelta64(np.timedelta64(6, 'h'))
    n_steps = 4
    times = cx.field(step_delta * jnp.arange(n_steps))
    scaled_weights = []
    for i in range(n_steps):
      ctx = {
          'timedelta': cx.field(step_delta * i),
          'times': times,
      }
      scaled_weights.append(scaler.scales(field, context=ctx).data)
    scaled_weights = np.array(scaled_weights)
    np.testing.assert_allclose(np.sum(scaled_weights**2), 1.0, atol=1e-6)


class CompileTimeEvalTest(parameterized.TestCase):

  def test_grid_area_scaler_lowering_has_no_trig_ops(self):
    grid = coordinates.LonLatGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    scaler = scaling.GridAreaScaler()

    def fn(f):
      return scaler.scales(f).data

    lowered_text = jax.jit(fn).lower(field).as_text()
    self.assertNotIn('sine', lowered_text.lower())
    self.assertNotIn('cosine', lowered_text.lower())

  def test_sigmoid_wavenumber_scaler_lowering_has_no_exp_ops(self):
    grid = coordinates.SphericalHarmonicGrid.T21()
    field = cx.field(np.ones(grid.shape), grid)
    scaler = scaling.SigmoidWavenumberScaler(cutoff_wavenumber=18)

    def fn(f):
      return scaler.scales(f).data

    lowered_text = jax.jit(fn).lower(field).as_text()
    self.assertNotIn('exponential', lowered_text.lower())

  def test_lead_time_scalers_static_coord_lowering_has_no_sqrt_ops(self):
    time_coord = coordinates.TimeDelta(
        np.array([0, 6, 12, 18]) * np.timedelta64(1, 'h')
    )
    field = cx.field(np.ones(time_coord.shape), time_coord)
    lt_scaler = scaling.LeadTimeScaler(base_squared_error_in_hours=6.0)
    glt_scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=6.0,
        asymptotic_norm=0.5,
        norm_transition_timescale_in_hours=12.0,
    )

    for scaler in (lt_scaler, glt_scaler):
      with self.subTest(scaler=type(scaler).__name__):
        lowered_text = (
            jax.jit(lambda f, s=scaler: s.scales(f).data)
            .lower(field)
            .as_text()
        )
        self.assertNotIn('sqrt', lowered_text.lower())

  def test_coordinate_mask_scaler_jit_with_dynamic_context(self):
    x = cx.SizedAxis('x', 4)
    field = cx.field(np.ones(x.shape), x)
    mask_deltas = np.array([6, 18]) * np.timedelta64(1, 'h')
    mask_coord = coordinates.TimeDelta(mask_deltas)
    mask_scaler = scaling.CoordinateMaskScaler(mask_coord=mask_coord)

    step_delta = jdt.Timedelta.from_timedelta64(np.timedelta64(6, 'h'))

    @jax.jit
    def jitted_fn(f, step):
      ctx = {'timedelta': cx.field(step_delta * step)}
      return mask_scaler.scales(f, context=ctx)

    for step_val in [1, 2]:
      with self.subTest(step=step_val):
        eager_ctx = {'timedelta': cx.field(step_delta * step_val)}
        eager_res = mask_scaler.scales(field, context=eager_ctx)
        jitted_res = jitted_fn(field, jnp.int32(step_val))
        cx.testing.assert_fields_allclose(jitted_res, eager_res)

  def test_lead_time_scaler_jit_with_dynamic_context(self):
    x = cx.SizedAxis('x', 1)
    field = cx.field(np.ones(x.shape), x)
    scaler = scaling.LeadTimeScaler(
        base_squared_error_in_hours=7.0, normalize_weights=True
    )
    step_delta = jdt.Timedelta.from_timedelta64(np.timedelta64(6, 'h'))
    n_steps = 4
    times = cx.field(step_delta * jnp.arange(n_steps))

    @jax.jit
    def jitted_fn(f, step):
      ctx = {
          'timedelta': cx.field(step_delta * step),
          'times': times,
      }
      return scaler.scales(f, context=ctx)

    for step_val in range(n_steps):
      with self.subTest(step=step_val):
        eager_ctx = {
            'timedelta': cx.field(step_delta * step_val),
            'times': times,
        }
        eager_res = scaler.scales(field, context=eager_ctx)
        jitted_res = jitted_fn(field, jnp.int32(step_val))
        cx.testing.assert_fields_allclose(jitted_res, eager_res, atol=1e-6)

  def test_generalized_lead_time_scaler_jit_with_dynamic_context(self):
    x = cx.SizedAxis('x', 1)
    field = cx.field(np.ones(x.shape), x)
    scaler = scaling.GeneralizedLeadTimeScaler(
        base_squared_error_in_hours=7.0,
        asymptotic_norm=0.5,
        norm_transition_timescale_in_hours=18.0,
    )
    step_delta = jdt.Timedelta.from_timedelta64(np.timedelta64(6, 'h'))
    n_steps = 4
    times = cx.field(step_delta * jnp.arange(n_steps))

    @jax.jit
    def jitted_fn(f, step):
      ctx = {
          'timedelta': cx.field(step_delta * step),
          'times': times,
      }
      return scaler.scales(f, context=ctx)

    for step_val in range(n_steps):
      with self.subTest(step=step_val):
        eager_ctx = {
            'timedelta': cx.field(step_delta * step_val),
            'times': times,
        }
        eager_res = scaler.scales(field, context=eager_ctx)
        jitted_res = jitted_fn(field, jnp.int32(step_val))
        cx.testing.assert_fields_allclose(jitted_res, eager_res, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
