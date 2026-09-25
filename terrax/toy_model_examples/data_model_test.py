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
"""Tests that DataModel outputs expected values.."""

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.core import coordinates
from terrax.core import diagnostics
from terrax.core import observation_operators
from terrax.core import transforms
from terrax.toy_model_examples import data_model


class DataModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.k = cx.LabeledAxis('k', np.arange(10))
    self.j = cx.LabeledAxis('j', np.arange(5))
    self.t0 = jdt.Datetime.from_isoformat('2000-01-01T00:00:00')

  def test_data_model(self):
    kj = cx.coords.compose(self.k, self.j)
    keys_to_coords = {'x': self.k, 'y': kj}
    observation_key = 'data'
    ones_like = lambda *c: cx.field(jnp.ones(cx.coords.compose(*c).shape), *c)

    model = data_model.DataModel(
        keys_to_coords=keys_to_coords,
        observation_key=observation_key,
        model_timestep=np.timedelta64(1, 'h'),
    )

    dt = coordinates.TimeDelta(np.arange(3) * np.timedelta64(1, 'h'))
    times = cx.field(self.t0[None] + dt.deltas, dt)
    time_arange = cx.field(np.arange(dt.shape[0]), dt)
    x_data = time_arange * ones_like(self.k)
    y_data = (time_arange**2) * ones_like(kj) * 2

    dynamic_inputs = {
        observation_key: {'time': times, 'x': x_data, 'y': y_data}
    }

    model.update_dynamic_inputs(dynamic_inputs)
    dt0 = dt.isel(timedelta=slice(0, 1))
    model_inputs = {
        observation_key: {
            'time': cx.field(self.t0[None], dt0),
            'x': ones_like(dt0, self.k),
            'y': ones_like(dt0, kj),
        }
    }

    model.assimilate(model_inputs)

    with self.subTest('assimilate_updates_prognostics'):
      prognostics = model.prognostics
      self.assertEqual(prognostics['time'].data, self.t0)
      cx.testing.assert_fields_equal(prognostics['x'], ones_like(self.k))
      cx.testing.assert_fields_equal(prognostics['y'], ones_like(kj))

    with self.subTest('advance_updates_prognostics'):
      model.advance()
      prognostics = model.prognostics
      cx.testing.assert_fields_equal(prognostics['x'], ones_like(self.k))
      cx.testing.assert_fields_equal(prognostics['y'], 2 * ones_like(kj))
      model.advance()
      prognostics = model.prognostics
      cx.testing.assert_fields_equal(prognostics['x'], 2 * ones_like(self.k))
      cx.testing.assert_fields_equal(prognostics['y'], 8 * ones_like(kj))
      self.assertEqual(  # called advance() twice --> +2h.
          prognostics['time'].data, self.t0 + np.timedelta64(2, 'h')
      )

    with self.subTest('observe_outputs'):
      queries = {observation_key: {'x': self.k}}
      observed = model.observe(queries)
      cx.testing.assert_fields_equal(
          observed[observation_key]['x'], 2 * ones_like(self.k)
      )

  def test_aggregate_diagnostic_uses_right_closed_window(self):
    """Aggregates over `n` steps must cover times `t + dt, ..., t + n * dt`."""
    observation_key = 'data'
    window_steps = 3
    total_steps = 6
    timestep = np.timedelta64(1, 'h')
    ones_k = cx.field(jnp.ones(self.k.shape), self.k)

    identity_operator = observation_operators.TransformObservationOperator(
        transforms.Identity()
    )
    diagnostic = diagnostics.IntervalDiagnostic(
        extract=diagnostics.ExtractFixedQueryObservations(
            observation_operator=identity_operator, query={'x': self.k}
        ),
        extract_coords={'x': self.k},
        interval=window_steps * timestep,
        resolution=window_steps * timestep,
        default_timedelta=timestep,
    )
    model = data_model.DataModel(
        keys_to_coords={'x': self.k},
        observation_key=observation_key,
        model_timestep=timestep,
        aggregate_state_diagnostics={'x_sum': diagnostic},
    )

    # `x` at the i-th time slice is uniformly equal to `i`.
    dt = coordinates.TimeDelta(np.arange(total_steps + 1) * timestep)
    times = cx.field(self.t0[None] + dt.deltas, dt)
    x_data = cx.field(np.arange(dt.shape[0]), dt) * ones_k
    model.update_dynamic_inputs(
        {observation_key: {'time': times, 'x': x_data}}
    )
    dt0 = dt.isel(timedelta=slice(0, 1))
    model.assimilate({
        observation_key: {
            'time': cx.field(self.t0[None], dt0),
            'x': cx.field(jnp.zeros(dt0.shape + self.k.shape), dt0, self.k),
        }
    })

    for _ in range(total_steps):
      model.advance()

    # The last completed window covers steps 4, 5 and 6, i.e. the aggregate is
    # right-closed: it excludes the state at the start of the window.
    observed = model.observe({'diagnostics': {'x': self.k}})
    expected = sum(range(total_steps - window_steps + 1, total_steps + 1))
    cx.testing.assert_fields_allclose(
        observed['diagnostics']['x'], expected * ones_k
    )


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
