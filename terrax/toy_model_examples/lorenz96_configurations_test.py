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

import os

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from fiddle.experimental import serialization
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.core import api
from terrax.core import checkpointing
from terrax.core import coordinates
from terrax.toy_model_examples import lorenz96
from terrax.toy_model_examples import lorenz96_configurations as configs


def _slow_inputs(k: cx.Coordinate) -> dict[str, dict[str, cx.Field]]:
  """Returns initial `slow` inputs with a dummy timedelta axis."""
  timedelta = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  x = np.sin(np.linspace(0, 4 * np.pi, k.shape[0]))
  return {
      'slow': {
          'x': cx.field(x[None, :], timedelta, k),
          'time': cx.field(jdt.to_datetime('2000-01-01')[None], timedelta),
      }
  }


class Lorenz96ConfigurationsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='truth', builder=configs.two_scale_truth, kwargs={}),
      dict(
          testcase_name='one_level', builder=configs.one_level_model, kwargs={}
      ),
      dict(
          testcase_name='parameterized',
          builder=configs.parameterized_model,
          kwargs=dict(noise='temporal'),
      ),
      dict(
          testcase_name='parameterized_deterministic_with_y',
          builder=configs.parameterized_model,
          kwargs=dict(noise=None, y_operator=True, y_noise='spatiotemporal'),
      ),
      dict(
          testcase_name='learned',
          builder=configs.learned_model,
          kwargs=dict(k_size=12, hidden_size=8, y_operator=True),
      ),
  )
  def test_builders_match_configs_and_serialize(self, builder, kwargs):
    direct = builder(**kwargs)
    config = builder.as_buildable(**kwargs)
    json_str = serialization.dump_json(config)
    model = api.Model.from_fiddle_config(serialization.load_json(json_str))
    self.assertIsInstance(model, type(direct))
    direct_params = nnx.to_pure_dict(nnx.state(direct, nnx.Param))
    model_params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    self.assertEqual(
        jax.tree.structure(direct_params), jax.tree.structure(model_params)
    )
    self.assertIsNotNone(model.fiddle_config)

  @parameterized.named_parameters(
      dict(testcase_name='parameterized', builder=configs.parameterized_model),
      dict(testcase_name='learned', builder=configs.learned_model),
  )
  def test_models_run_and_observe_y(self, builder):
    model = builder(k_size=12, y_operator=True)
    k = model.k_axis
    model.initialize_random_processes(cx.field(jax.random.key(0)))
    model.assimilate(_slow_inputs(k))
    x0 = model.x.get_value().data
    model.advance()
    self.assertFalse(np.allclose(model.x.get_value().data, x0))
    j = lorenz96.ring_axis(configs.J_SIZE, 'j')
    observed = model.observe({'fast': {'y': cx.coords.compose(k, j)}})
    y = observed['fast']['y']
    self.assertEqual(y.coordinate, cx.coords.compose(k, j))
    self.assertTrue(np.all(np.isfinite(y.data)))
    # Noise inputs are advanced with the model, so repeated observations differ.
    model.advance()
    y_next = model.observe({'fast': {'y': cx.coords.compose(k, j)}})
    self.assertFalse(np.allclose(y.data, y_next['fast']['y'].data))

  def test_with_y_operator_keeps_core_model(self):
    config = configs.parameterized_model.as_buildable(k_size=12)
    extended = configs.with_y_operator(config, j_size=5, noise='temporal')
    self.assertNotIn('fast', config.operators)
    self.assertIn('fast', extended.operators)
    core = api.Model.from_fiddle_config(config)
    model = api.Model.from_fiddle_config(extended)
    core_params = nnx.to_pure_dict(nnx.state(core.parameterizations, nnx.Param))
    params = nnx.to_pure_dict(nnx.state(model.parameterizations, nnx.Param))
    jax.tree.map(np.testing.assert_array_equal, core_params, params)
    k, j = model.k_axis, lorenz96.ring_axis(5, 'j')
    model.initialize_random_processes(cx.field(jax.random.key(0)))
    model.assimilate(_slow_inputs(k))
    observed = model.observe({'fast': {'y': cx.coords.compose(k, j)}})
    self.assertEqual(observed['fast']['y'].coordinate, cx.coords.compose(k, j))
    # Round trip through JSON works for the extended config.
    serialization.load_json(serialization.dump_json(extended))

  def test_with_y_operator_on_config_without_operators(self):
    config = configs.one_level_model.as_buildable(k_size=12)
    self.assertNotIn('operators', config.__arguments__)
    model = api.Model.from_fiddle_config(
        configs.with_y_operator(config, j_size=3, hidden_size=4)
    )
    self.assertEqual(set(model.operators), {'slow', 'fast'})

  def test_pointwise_y_operator(self):
    """Tests the `y` operator with noise fields that are correlated along j."""
    core_config = configs.parameterized_model.as_buildable(k_size=12)
    make_config = lambda **kwargs: configs.with_y_operator(
        core_config,
        operator_fn=configs.pointwise_y_observation_operator,
        j_size=5,
        hidden_size=8,
        noise_fields=('smooth', 'long_waves'),
        **kwargs,
    )
    config = make_config()
    json_str = serialization.dump_json(config)
    model = api.Model.from_fiddle_config(serialization.load_json(json_str))
    k, j = model.k_axis, lorenz96.ring_axis(5, 'j')
    kj = cx.coords.compose(k, j)

    with self.subTest('observes_y'):
      model.initialize_random_processes(cx.field(jax.random.key(0)))
      model.assimilate(_slow_inputs(k))
      y = model.observe({'fast': {'y': kj}})['fast']['y']
      self.assertEqual(y.coordinate, kj)
      self.assertTrue(np.all(np.isfinite(y.data)))
      model.advance()
      y_next = model.observe({'fast': {'y': kj}})['fast']['y']
      self.assertFalse(np.allclose(y.data, y_next.data))

    def count_params(model):
      params = nnx.state(model.operators['fast'], nnx.Param)
      return len(jax.tree.leaves(nnx.to_pure_dict(params)))

    with self.subTest('learnable_noise'):
      # One correlation length per noise field and a wavelength of `long_waves`.
      frozen = api.Model.from_fiddle_config(make_config(learnable_noise=False))
      self.assertEqual(count_params(model), count_params(frozen) + 3)

    with self.subTest('gradients_reach_noise_parameters'):
      operator = model.operators['fast']
      prognostics = model.prognostics

      def neighbor_product(operator):
        y = operator.observe(prognostics, {'y': kj})['y']
        return (y.data * jnp.roll(y.data, 1, axis=-1)).mean()

      grads = nnx.grad(neighbor_product, argnums=nnx.DiffState(0, nnx.Param))(
          operator
      )
      raw_grads = [
          float(grad)
          for path, grad in nnx.to_flat_state(grads)
          if 'raw_parameters' in path
      ]
      self.assertLen(raw_grads, 3)
      self.assertTrue(all(g != 0.0 and np.isfinite(g) for g in raw_grads))

  def test_pointwise_y_operator_defaults(self):
    """Tests the operator with all default noise fields on the default ring."""
    config = configs.with_y_operator(
        configs.parameterized_model.as_buildable(k_size=6, hidden_size=8),
        operator_fn=configs.pointwise_y_observation_operator,
        hidden_size=8,
    )
    model = api.Model.from_fiddle_config(config)
    noises = [
        module
        for _, module in nnx.iter_modules(model.operators['fast'])
        if isinstance(module, lorenz96.CorrelatedNoise)
    ]
    self.assertLen(noises, len(configs.NOISE_FIELD_PRESETS))
    k, j = model.k_axis, lorenz96.ring_axis(configs.J_SIZE, 'j')
    kj = cx.coords.compose(k, j)
    model.initialize_random_processes(cx.field(jax.random.key(0)))
    model.assimilate(_slow_inputs(k))
    y = model.observe({'fast': {'y': kj}})['fast']['y']
    self.assertEqual(y.coordinate, kj)
    self.assertTrue(np.all(np.isfinite(y.data)))

  def test_pointwise_y_operator_validates_noise_fields(self):
    k = lorenz96.ring_axis(6)
    build = configs.pointwise_y_observation_operator
    with self.subTest('string'):
      with self.assertRaisesRegex(TypeError, 'must be a sequence'):
        build(k, noise_fields='smooth')
    with self.subTest('empty'):
      with self.assertRaisesRegex(ValueError, 'at least one noise field'):
        build(k, noise_fields=())

  def test_checkpoint_registry_round_trip(self):
    path = os.path.join(self.create_tempdir().full_path, 'ckpt')
    model = api.Model.from_fiddle_config(
        configs.parameterized_model.as_buildable(k_size=12, hidden_size=8)
    )
    checkpointing.save_checkpoint(model, path)
    with absltest.mock.patch.dict(configs.LOCAL_CHECKPOINTS, {'test': path}):
      self.assertIn('test', configs.checkpoint_names())
      loaded = configs.load_checkpoint('test')
    expected = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    actual = nnx.to_pure_dict(nnx.state(loaded, nnx.Param))
    jax.tree.map(np.testing.assert_array_equal, expected, actual)
    with self.assertRaisesRegex(KeyError, 'Unknown checkpoint'):
      configs.checkpoint_path('missing')


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
