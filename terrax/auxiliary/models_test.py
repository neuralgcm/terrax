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
"""Tests for auxiliary models."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np
from terrax.auxiliary import models
from terrax.core import api
from terrax.core import diagnostics as diagnostics_lib
from terrax.core import transforms
from terrax.core import typing

config.parse_flags_with_absl()


class LabeledStateModelTest(parameterized.TestCase):

  def test_basic_stage_and_advance(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    x = cx.field(np.array([2.5, 4.0]), s_axis)
    model.assimilate({'state': {'total': x}})
    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [2.5, 4.0])

    update = cx.field(np.array([0.5, -1.0]), s_axis)
    model.stage_updates({'total': update})
    model.advance()

    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [3.0, 3.0])

  def test_update_every(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(3, 'h'),
    )
    x = cx.field(np.array([1.0, 2.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    update = cx.field(np.array([10.0, 10.0]), s_axis)
    model.stage_updates({'total': update})

    for _ in range(2):
      model.advance()
      state = model.prognostics.get_value()
      np.testing.assert_allclose(state['total'].data, [1.0, 2.0])

    model.advance()
    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [11.0, 12.0])

  def test_accumulate_updates(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        update_every=np.timedelta64(3, 'h'),
        collect_update_method='accumulate',
    )
    x = cx.field(np.array([0.0, 0.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    update = cx.field(np.array([1.0, 2.0]), s_axis)
    for _ in range(3):
      model.stage_updates({'total': update})
      model.advance()

    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['total'].data, [3.0, 6.0])
    staged = model._staged_updates.get_value()
    np.testing.assert_allclose(staged['total'].data, [0.0, 0.0])

  def test_observe(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    x = cx.field(np.array([3.0, 5.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    with self.subTest('full_state'):
      query = {'state': {'total': s_axis}}
      obs = model.observe(query)
      np.testing.assert_allclose(obs['state']['total'].data, [3.0, 5.0])

    with self.subTest('partial_state'):
      a_s_axis = cx.LabeledAxis('station', ['A'])
      query = {'state': {'total': a_s_axis}}
      obs = model.observe(query)
      np.testing.assert_allclose(obs['state']['total'].data, [3.0])

  def test_missing_assimilation_input_raises(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    with self.assertRaisesRegex(
        ValueError, 'missing required prognostic variable'
    ):
      model.assimilate({'state': {}})

  def test_invalid_update_every_raises(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    with self.assertRaisesRegex(ValueError, 'positive integer multiple'):
      models.LabeledStateModel(
          prognostic_coords={'total': s_axis},
          data_key='state',
          model_timestep=np.timedelta64(1, 'h'),
          update_every=np.timedelta64(90, 'm'),
      )

  def test_apply_update_method_set(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        apply_update_method='set',
    )
    x = cx.field(np.array([10.0, 20.0]), s_axis)
    model.assimilate({'state': {'total': x}})

    replacement = cx.field(np.array([99.0, 88.0]), s_axis)
    model.stage_updates({'total': replacement})
    model.advance()
    state = model.prognostics.get_value()
    cx.testing.assert_fields_allclose(state['total'], replacement)

  def test_update_mapping(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])
    model = models.LabeledStateModel(
        prognostic_coords={'state_var': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        collect_update_method='accumulate',
        update_mapping={'update_var': 'state_var'},
    )
    x = cx.field(np.array([1.0, 2.0]), s_axis)
    model.assimilate({'state': {'state_var': x}})

    update = cx.field(np.array([3.0, 4.0]), s_axis)
    model.stage_updates({'update_var': update})
    model.advance()

    state = model.prognostics.get_value()
    np.testing.assert_allclose(state['state_var'].data, [4.0, 6.0])

  def test_diagnostics_merged(self):
    s_axis = cx.LabeledAxis('station', ['A', 'B'])

    extractor = diagnostics_lib.ExtractTransformedOutputs(
        transform=transforms.SelectKeys('state_var')
    )
    diagnostic = diagnostics_lib.TimeOffsetDiagnostic(
        extract=extractor,
        extract_coords={'state_var': s_axis},
        offset={'minus_24h': np.timedelta64(24, 'h')},
        resolution=np.timedelta64(24, 'h'),
        default_timedelta=np.timedelta64(24, 'h'),
        output_mapping={'state_var_minus_24h': 'condition_var'},
    )

    model = models.LabeledStateModel(
        prognostic_coords={'state_var': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(24, 'h'),
        diagnostics={'t_minus_24h': diagnostic},
    )
    x = cx.field(np.array([1.0, 2.0]), s_axis)
    model.assimilate({'state': {'state_var': x}})

    # Advance two steps to fully propagate time t state into the lag buffer
    model.advance()
    model.advance()

    obs = model.observe(
        {'state': {'state_var': s_axis, 'condition_var': s_axis}}
    )
    np.testing.assert_allclose(obs['state']['condition_var'].data, [1.0, 2.0])


def _get_station_rates(coord: cx.LabeledAxis) -> jnp.ndarray:
  """Computes decay rates dynamically based on station label indices."""
  rates = []
  for tick in coord.ticks:
    station_id = int(tick[1:])
    rates.append(-0.1 * ((1 / 5) ** (station_id - 1)))
  return jnp.array(rates)


class _DecreaseToZeroModel(api.Model):
  """Base model where delta observation requires total state as a Field."""

  def assimilate(self, inputs):
    pass

  def advance(self):
    pass

  def observe(self, queries):
    results = {}
    for op_key, q in queries.items():
      if op_key == 'state':
        obs = {}
        delta_q_coord, delta_is_aux = typing.unwrap_auxiliary(q.get('delta'))
        total, total_is_aux = typing.unwrap_auxiliary(q.get('total'))

        if delta_q_coord is not None:
          if not cx.is_field(total):
            raise ValueError('Model requires total to be a Field in query.')
          rates = cx.field(_get_station_rates(delta_q_coord), delta_q_coord)
          delta = cx.cmap(jnp.where)(total > 0, rates, 0.0)

          if not delta_is_aux:
            obs['delta'] = delta
          if not total_is_aux:
            obs['total'] = total
        results[op_key] = obs
    return results

  @property
  def timestep(self):
    return np.timedelta64(1, 'h')

  @property
  def inputs_spec(self):
    return {}


class WithObservedStateTest(parameterized.TestCase):

  def _make_wrapped_model(self):
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])
    base_model = _DecreaseToZeroModel()
    obs_model = models.LabeledStateModel(
        prognostic_coords={'total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
    )
    wrapper = models.WithObservedState(
        base_model=base_model,
        observation_models={'state': obs_model},
        coupling_query={},
        observation_query_forwarding={'state': ('total',)},
    )
    return wrapper, s_axis

  def test_base_model_requires_field_in_query(self):
    base_model = _DecreaseToZeroModel()
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])

    query_invalid = {'state': {'delta': s_axis, 'total': s_axis}}
    with self.assertRaisesRegex(ValueError, 'Field in query'):
      base_model.observe(query_invalid)

    total_field = cx.field(np.array([1.0, 0.0]), s_axis)
    query_valid = {'state': {'delta': s_axis, 'total': total_field}}
    obs = base_model.observe(query_valid)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, 0.0])
    np.testing.assert_allclose(obs['state']['total'].data, [1.0, 0.0])

  def test_wrapper_forwards_query_and_enables_coordinate_query(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})

    query = {'state': {'delta': s_axis, 'total': s_axis}}
    obs = wrapper.observe(query)
    self.assertIn('state', obs)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, 0.0])
    np.testing.assert_allclose(obs['state']['total'].data, [1.0, 0.0])

  def test_wrapper_observes_subset_of_stations(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.5]), s_axis)
    wrapper.assimilate({'state': {'total': x}})

    # Query with only station S2.
    s_subset = cx.LabeledAxis('station', ['S2'])
    query = {'state': {'delta': s_subset, 'total': s_subset}}
    obs = wrapper.observe(query)
    self.assertIn('state', obs)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.02])
    np.testing.assert_allclose(obs['state']['total'].data, [0.5])

  def test_observe_preserves_auxiliary_tags(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([1.0, 0.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})
    query = {'state': {'delta': s_axis, 'total': typing.Auxiliary(s_axis)}}
    obs = wrapper.observe(query)
    self.assertIn('state', obs)
    self.assertIn('delta', obs['state'])
    self.assertNotIn('total', obs['state'])
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, 0.0])

  def test_observe_missing_forwarded_key_raises(self):
    wrapper, s_axis = self._make_wrapped_model()
    query_missing = {'state': {'delta': s_axis}}
    with self.assertRaisesRegex(ValueError, 'requires key'):
      wrapper.observe(query_missing)

  def test_dual_coupling_and_observation_query_forwarding(self):
    s_axis = cx.LabeledAxis('station', ['S1', 'S2'])
    base_model = _DecreaseToZeroModel()
    obs_model = models.LabeledStateModel(
        prognostic_coords={'prog_total': s_axis, 'lag_total': s_axis},
        data_key='state',
        model_timestep=np.timedelta64(1, 'h'),
        update_mapping={'delta': 'prog_total'},
    )
    wrapper = models.WithObservedState(
        base_model=base_model,
        observation_models={'state': obs_model},
        coupling_query={
            'state': {'delta': s_axis, 'total': typing.Auxiliary(s_axis)}
        },
        observation_query_forwarding={'state': {'total': 'lag_total'}},
        coupling_query_forwarding={'state': {'total': 'prog_total'}},
    )
    x_prog = cx.field(np.array([10.0, 5.0]), s_axis)
    x_lag = cx.field(np.array([100.0, 50.0]), s_axis)
    wrapper.assimilate({'state': {'prog_total': x_prog, 'lag_total': x_lag}})

    query = {'state': {'delta': s_axis, 'total': s_axis}}
    obs = wrapper.observe(query)
    np.testing.assert_allclose(obs['state']['delta'].data, [-0.1, -0.02])
    np.testing.assert_allclose(obs['state']['total'].data, [100.0, 50.0])

    wrapper.advance()
    state = wrapper.observation_models['state'].prognostics.get_value()
    np.testing.assert_allclose(state['prog_total'].data, [9.9, 4.98])

  def test_direct_state_observation_shortcut(self):
    wrapper, s_axis = self._make_wrapped_model()
    x = cx.field(np.array([10.0, 20.0]), s_axis)
    wrapper.assimilate({'state': {'total': x}})
    query = {'state_state': {'total': s_axis}}
    obs = wrapper.observe(query)
    self.assertIn('state_state', obs)
    np.testing.assert_allclose(obs['state_state']['total'].data, [10.0, 20.0])

  def test_timestep_delegated_to_base(self):
    wrapper, _ = self._make_wrapped_model()
    self.assertEqual(wrapper.timestep, np.timedelta64(1, 'h'))

  def test_inputs_spec_merges_both_models(self):
    wrapper, _ = self._make_wrapped_model()
    spec = wrapper.inputs_spec
    self.assertIn('state', spec)
    self.assertIn('total', spec['state'])


if __name__ == '__main__':
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
