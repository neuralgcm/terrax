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

"""Tests that transforms produce outputs with expected structure."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.core import coordinates
from terrax.core import random_processes
from terrax.core import transforms
from terrax.toy_model_examples import lorenz96


def _add_dummy_timedelta(
    inputs: dict[str, dict[str, cx.Field]],
) -> dict[str, dict[str, cx.Field]]:
  """Adds a dummy timedelta dimension to all fields in a pytree."""
  t_del = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  add_td = lambda f: f.broadcast_like(cx.coords.compose(t_del, f.coordinate))
  return jax.tree.map(add_td, inputs, is_leaf=cx.is_field)


def _make_x_l96_inputs(
    k: cx.Coordinate, t0: jdt.Datetime
) -> dict[str, dict[str, cx.Field]]:
  """Helper to generate 'slow' inputs for Lorenz96 models."""
  dt = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  x_init = abs(10 * np.sin(np.linspace(0, 13 * 2 * np.pi, k.sizes['k'])))
  inputs = {
      'slow': {
          'x': cx.field(x_init[None, :], dt, k),
          'time': cx.field(t0[None], dt),
      }
  }
  return inputs


def _make_y_l96_inputs(
    k: cx.Coordinate, j: cx.Coordinate, t0: jdt.Datetime
) -> dict[str, dict[str, cx.Field]]:
  """Helper to generate 'fast' inputs for Lorenz96 models."""
  dt = coordinates.TimeDelta(np.zeros(1) * np.timedelta64(1, 'h'))
  kj = cx.coords.compose(k, j)
  rng = np.random.RandomState(0)
  inputs = {
      'fast': {
          'y': cx.field(rng.uniform(-0.5, 0.5, dt.shape + kj.shape), dt, kj),
          'time': cx.field(t0[None], dt),
      }
  }
  return inputs


def _get_coordinates(
    observed: dict[str, dict[str, cx.Field]],
) -> dict[str, dict[str, cx.Coordinate]]:
  """Helper to get coordinates of observed fields."""
  return jax.tree.map(cx.get_coordinate, observed, is_leaf=cx.is_field)


class Lorenz96Test(parameterized.TestCase):
  """Tests that lorenz96 models methods produce expected outputs."""

  def setUp(self):
    super().setUp()
    self.k = cx.LabeledAxis('k', np.arange(36))
    self.j = cx.LabeledAxis('j', np.arange(8))
    self.t0 = jdt.Datetime.from_isoformat('2000-01-01')

  def test_lorenz96_with_two_scales(self):
    """Tests Lorenz96WithTwoScales model methods produce expected outputs."""
    k, j, t0 = self.k, self.j, self.t0
    kj = cx.coords.compose(k, j)
    l96_model = lorenz96.Lorenz96WithTwoScales(k_axis=k, j_axis=j)
    self.assertEqual(l96_model.x.coordinate, k)
    self.assertEqual(l96_model.y.coordinate, kj)

    inputs = {**_make_x_l96_inputs(k, t0), **_make_y_l96_inputs(k, j, t0)}
    l96_model.assimilate(inputs)
    self.assertEqual(l96_model.x.coordinate, k)
    self.assertEqual(l96_model.y.coordinate, kj)

    l96_model.advance()
    self.assertEqual(l96_model.x.coordinate, k)
    self.assertEqual(l96_model.y.coordinate, kj)

    queries = {'slow': {'x': k}, 'fast': {'y': kj}}
    observed = l96_model.observe(queries)
    chex.assert_trees_all_equal(_get_coordinates(observed), queries)
    chex.assert_tree_all_finite(observed)

  def test_lorenz96_with_param(self):
    """Tests Lorenz96 with parameterization produce expected outputs."""
    k, t0 = self.k, self.t0
    random_process = random_processes.NormalUncorrelated.construct(
        coord=k, mean=0.0, std=1.0, rngs=nnx.Rngs(0)
    )
    param = lorenz96.StochasticLinearParameterization(
        random_process=random_process
    )
    l96_model = lorenz96.Lorenz96(k_axis=k, parameterizations=[param])
    self.assertEqual(l96_model.x.coordinate, k)

    inputs = _make_x_l96_inputs(k, t0)
    key = cx.field(jax.random.key(0))
    l96_model.initialize_random_processes(key)
    l96_model.assimilate(inputs)
    self.assertEqual(l96_model.x.coordinate, k)
    l96_model.advance()
    self.assertEqual(l96_model.x.coordinate, k)

    queries = {'slow': {'x': k}}
    observed = l96_model.observe(queries)
    chex.assert_trees_all_equal(_get_coordinates(observed), queries)

  def test_lorenz96_fast_mode(self):
    """Tests Lorenz96FastMode model methods produce expected outputs."""
    k, j, t0 = self.k, self.j, self.t0
    kj = cx.coords.compose(k, j)
    l96_model = lorenz96.Lorenz96FastMode(k_axis=k, j_axis=j)
    self.assertEqual(l96_model.y.coordinate, kj)

    inputs = _make_y_l96_inputs(k, j, t0)
    l96_model.assimilate(inputs)
    self.assertEqual(l96_model.y.coordinate, kj)
    l96_model.advance()
    self.assertEqual(l96_model.y.coordinate, kj)

    queries = {'fast': {'y': kj}}
    observed = l96_model.observe(queries)
    chex.assert_trees_all_equal(_get_coordinates(observed), queries)

  def test_stochastic_parameterization_noise_advances_with_model(self):
    """Tests that a model step advances noise of parameterizations once."""
    k, t0 = self.k, self.t0
    random_process = lorenz96.CorrelatedNoise(
        coord=k, correlation_steps=2.0, rngs=nnx.Rngs(0)
    )
    param = lorenz96.StochasticLinearParameterization(
        random_process=random_process
    )
    l96_model = lorenz96.Lorenz96(k_axis=k, parameterizations=[param])
    l96_model.initialize_random_processes(cx.field(jax.random.key(0)))
    l96_model.assimilate(_make_x_l96_inputs(k, t0))
    noise_before = random_process.state_values(k).data
    reference_process = nnx.clone(random_process)
    l96_model.advance()
    noise_after = random_process.state_values(k).data
    self.assertFalse(np.allclose(noise_before, noise_after))
    # The noise is advanced by the model only, and only once per step.
    reference_process.advance()
    np.testing.assert_allclose(
        noise_after, reference_process.state_values(k).data, rtol=1e-6
    )

  def test_stencil_features(self):
    """Tests that stencil features hold values at shifted grid points."""
    k = self.k
    x = cx.field(np.arange(k.shape[0], dtype=np.float32), k)
    features = lorenz96.StencilFeatures(coord=k, shifts=(-1, 0, 2))(
        {'x': x, 'time': cx.field(self.t0)}
    )
    self.assertEqual(set(features), {'x-1', 'x', 'x+2'})
    np.testing.assert_array_equal(features['x'].data, x.data)
    np.testing.assert_array_equal(features['x-1'].data, np.roll(x.data, 1))
    np.testing.assert_array_equal(features['x+2'].data, np.roll(x.data, -2))
    for f in features.values():
      self.assertEqual(f.coordinate, k)

  @parameterized.named_parameters(
      dict(testcase_name='white', correlation_steps=0.0, correlation_length=0),
      dict(
          testcase_name='temporal', correlation_steps=4.0, correlation_length=0
      ),
      dict(testcase_name='both', correlation_steps=4.0, correlation_length=2.0),
  )
  def test_correlated_noise_statistics(
      self, correlation_steps, correlation_length
  ):
    """Tests unit variance and one-step autocorrelation of CorrelatedNoise."""
    k = lorenz96.ring_axis(64)
    noise = lorenz96.CorrelatedNoise(
        coord=k,
        correlation_steps=correlation_steps,
        correlation_length=correlation_length,
        rngs=nnx.Rngs(0),
    )
    samples = self._sample_noise(noise, steps=20)  # (step, ensemble, k)
    self.assertEqual(samples.shape, (21, 512, 64))
    np.testing.assert_allclose(samples.mean(), 0.0, atol=0.03)
    np.testing.assert_allclose(samples.var(), 1.0, atol=0.05)
    lag_1 = np.mean(samples[1:] * samples[:-1])
    np.testing.assert_allclose(lag_1, noise.phi, atol=0.03)
    neighbor = np.mean(samples * np.roll(samples, 1, axis=-1))
    expected_neighbor = (
        np.exp(-0.25 / correlation_length**2) if correlation_length else 0.0
    )
    np.testing.assert_allclose(neighbor, expected_neighbor, atol=0.03)

  def _sample_noise(self, noise, steps, ensemble_size=512):
    """Returns samples of `noise` with shape (steps + 1, ensemble, ...)."""
    ensemble = cx.SizedAxis('ensemble', ensemble_size)
    keys = jax.random.split(jax.random.key(1), ensemble.size)
    noise.unconditional_sample(cx.field(keys, ensemble))
    samples = [noise.state_values().data]
    for _ in range(steps):
      noise.advance()
      samples.append(noise.state_values().data)
    return np.stack(samples)

  def test_correlated_noise_with_wavelength(self):
    """Tests that covariances of the noise are set by its wave-like kernel."""
    channel, j = cx.SizedAxis('channel', 2), lorenz96.ring_axis(10, 'j')
    noise = lorenz96.CorrelatedNoise(
        coord=cx.coords.compose(channel, j),
        correlation_length=2.0,
        wavelength=5.0,
        dim='j',
        rngs=nnx.Rngs(0),
    )
    samples = self._sample_noise(noise, steps=4)
    self.assertEqual(samples.shape, (5, 512, 2, 10))
    kernel = np.asarray(noise.kernel())
    distance = np.minimum(np.arange(10), 10 - np.arange(10))
    envelope = np.exp(-0.5 * (distance / 2.0) ** 2)
    expected_kernel = envelope * np.cos(2 * np.pi * distance / 5.0)
    expected_kernel /= np.sqrt(np.sum(expected_kernel**2))
    np.testing.assert_allclose(kernel, expected_kernel, rtol=1e-5, atol=1e-6)
    for lag in range(6):
      with self.subTest(lag=lag):
        covariance = np.mean(samples * np.roll(samples, lag, axis=-1))
        expected = np.sum(kernel * np.roll(kernel, lag))
        np.testing.assert_allclose(covariance, expected, atol=0.03)
    with self.subTest('spectral_peak'):
      power = np.abs(np.fft.rfft(samples, axis=-1)) ** 2
      self.assertEqual(np.argmax(power.mean(axis=(0, 1, 2))), 2)  # 10 / 5.

  @parameterized.named_parameters(
      dict(testcase_name='integer_drift', drift=3.0),
      dict(testcase_name='fractional_drift', drift=-1.25),
  )
  def test_correlated_noise_with_drift(self, drift):
    """Tests that noise patterns travel by `drift` grid points per step."""
    k = lorenz96.ring_axis(16)
    noise = lorenz96.CorrelatedNoise(
        coord=k,
        correlation_steps=4.0,
        correlation_length=1.5,
        drift=drift,
        rngs=nnx.Rngs(0),
    )
    samples = self._sample_noise(noise, steps=20)  # (step, ensemble, k)
    np.testing.assert_allclose(samples.var(), 1.0, atol=0.05)
    np.testing.assert_allclose(samples.var(axis=(0, 1)), 1.0, atol=0.1)
    # Cross-covariance between consecutive steps as a function of the shift
    # equals `phi` times the autocovariance of the kernel shifted by `drift`.
    kernel_hat = np.fft.rfft(np.asarray(noise.kernel()))
    wavenumbers = np.arange(kernel_hat.size)
    shift = np.exp(-2j * np.pi * wavenumbers * drift / 16)
    expected = noise.phi * np.fft.irfft(np.abs(kernel_hat) ** 2 * shift, n=16)
    actual = [
        np.mean(samples[1:] * np.roll(samples[:-1], s, axis=-1))
        for s in range(16)
    ]
    np.testing.assert_allclose(actual, expected, atol=0.04)
    self.assertEqual(np.argmax(actual), np.argmax(expected))

  def test_correlated_noise_with_dim_in_the_middle(self):
    """Tests that correlations follow `dim` wherever it is in the layout."""
    j, channel = lorenz96.ring_axis(10, 'j'), cx.SizedAxis('channel', 3)
    noise = lorenz96.CorrelatedNoise(
        coord=cx.coords.compose(j, channel),
        correlation_steps=2.0,
        correlation_length=1.5,
        dim='j',
        drift=2.0,
        rngs=nnx.Rngs(0),
    )
    samples = self._sample_noise(noise, steps=8)
    self.assertEqual(noise.state_values().dims, ('ensemble', 'j', 'channel'))
    self.assertEqual(samples.shape, (9, 512, 10, 3))
    kernel = np.asarray(noise.kernel())
    along_j = np.mean(samples * np.roll(samples, 1, axis=2))
    expected_along_j = np.sum(kernel * np.roll(kernel, 1))
    np.testing.assert_allclose(along_j, expected_along_j, atol=0.03)
    across_channels = np.mean(samples * np.roll(samples, 1, axis=3))
    np.testing.assert_allclose(across_channels, 0.0, atol=0.03)
    # Patterns travel 2 points along `j` in one step.
    travelled = np.mean(samples[1:] * np.roll(samples[:-1], 2, axis=2))
    np.testing.assert_allclose(travelled, noise.phi, atol=0.03)

  def test_drifting_ar1_step(self):
    """Tests the shift and the treatment of the Nyquist mode of a ring."""
    phi, drift = 0.6, 0.3
    step = functools.partial(lorenz96._drifting_ar1_step, phi=phi, drift=drift)
    nyquist = (-1.0) ** np.arange(8)
    zeros = np.zeros(8)
    z = np.random.RandomState(0).normal(size=8)
    with self.subTest('nyquist_mode_is_damped'):
      gain = phi * np.cos(np.pi * drift)
      np.testing.assert_allclose(
          step(nyquist, zeros), gain * nyquist, atol=1e-5
      )
    with self.subTest('nyquist_innovation_keeps_unit_variance'):
      scale = np.sqrt(1 - (phi * np.cos(np.pi * drift)) ** 2)
      np.testing.assert_allclose(
          step(zeros, nyquist), scale * nyquist, atol=1e-5
      )
    with self.subTest('integer_drift_is_a_roll'):
      shifted = lorenz96._drifting_ar1_step(z, zeros, phi=1.0, drift=3.0)
      np.testing.assert_allclose(shifted, np.roll(z, 3), atol=1e-5)
    with self.subTest('gradients_near_vanishing_nyquist_gain'):
      # cos(pi * drift) vanishes for drift == 0.5 (up to round-off).
      total = lambda drift: lorenz96._drifting_ar1_step(z, z, 0.6, drift).sum()
      self.assertTrue(np.isfinite(jax.grad(total)(0.5)))

  def test_correlated_noise_learnable_parameters(self):
    """Tests that learnable parameters are `nnx.Param`s with gradients."""
    j = lorenz96.ring_axis(10, 'j')
    make_noise = lambda learnable: lorenz96.CorrelatedNoise(
        coord=j,
        correlation_steps=2.0,
        correlation_length=1.5,
        dim='j',
        wavelength=4.0,
        drift=1.0,
        learnable=learnable,
        rngs=nnx.Rngs(0),
    )
    with self.subTest('static_by_default'):
      self.assertEmpty(nnx.state(make_noise(()), nnx.Param))

    learnable = (
        'correlation_length',
        'wavelength',
        'correlation_steps',
        'drift',
    )
    noise = make_noise(learnable)
    with self.subTest('initial_values'):
      self.assertEqual(set(noise.raw_parameters), set(learnable))
      np.testing.assert_allclose(noise.correlation_length, 1.5, rtol=1e-6)
      np.testing.assert_allclose(noise.wavelength, 4.0, rtol=1e-6)
      np.testing.assert_allclose(noise.correlation_steps, 2.0, rtol=1e-6)
      np.testing.assert_allclose(noise.drift, 1.0, rtol=1e-6)

    with self.subTest('gradients'):

      def neighbor_product(noise):
        noise.advance()
        values = noise.state_values().data
        return (values * jnp.roll(values, 1)).mean()

      grads = nnx.grad(neighbor_product, argnums=nnx.DiffState(0, nnx.Param))(
          noise
      )
      grads = nnx.to_pure_dict(grads)['raw_parameters']
      self.assertEqual(set(grads), set(learnable))
      for name, grad in grads.items():
        self.assertTrue(np.isfinite(grad), msg=name)
        self.assertNotEqual(float(grad), 0.0, msg=name)

    with self.subTest('positive_parameters_stay_positive'):
      noise.raw_parameters['correlation_length'].set_value(jnp.asarray(-50.0))
      self.assertGreater(float(noise.correlation_length), 0.0)

  def test_correlated_noise_validates_arguments(self):
    make_noise = functools.partial(
        lorenz96.CorrelatedNoise, coord=self.k, rngs=nnx.Rngs(0)
    )
    with self.subTest('wavelength_without_correlation_length'):
      with self.assertRaisesRegex(ValueError, 'positive correlation_length'):
        make_noise(wavelength=5.0)
    with self.subTest('non_positive_wavelength'):
      with self.assertRaisesRegex(ValueError, 'must be positive or None'):
        make_noise(correlation_length=1.0, wavelength=0.0)
    with self.subTest('negative_scales'):
      with self.assertRaisesRegex(ValueError, 'must be non-negative'):
        make_noise(correlation_length=-1.0)
    with self.subTest('unknown_dim'):
      with self.assertRaisesRegex(ValueError, 'is not a dimension'):
        make_noise(dim='j')
    with self.subTest('drift_without_correlation_steps'):
      with self.assertRaisesRegex(ValueError, 'positive correlation_steps'):
        make_noise(drift=1.0)
    with self.subTest('unknown_learnable'):
      with self.assertRaisesRegex(ValueError, 'Unknown parameter'):
        make_noise(correlation_length=1.0, learnable=('length',))
    with self.subTest('switched_off_learnable'):
      with self.assertRaisesRegex(ValueError, 'switched off'):
        make_noise(correlation_length=1.0, learnable=('wavelength',))

  def test_correlated_noise_rejects_other_coordinates(self):
    noise = lorenz96.CorrelatedNoise(coord=self.k, rngs=nnx.Rngs(0))
    with self.assertRaisesRegex(ValueError, 'defined on'):
      noise.state_values(self.j)

  def test_transform_parameterization_and_learned_model(self):
    """Tests TransformParameterization and LearnedLorenz96 advance the state."""
    k, t0 = self.k, self.t0
    damping = transforms.ApplyFnToKeys(lambda f: -0.1 * f, ['x'])
    renamed = transforms.Sequential(
        [damping, transforms.RenameKeys({'x': 'x_delta'})]
    )
    param = lorenz96.TransformParameterization(renamed, key='x_delta')
    l96_model = lorenz96.Lorenz96(k_axis=k, parameterizations=[param])
    learned = lorenz96.LearnedLorenz96(k_axis=k, transform=renamed)
    inputs = _make_x_l96_inputs(k, t0)
    x0 = inputs['slow']['x'].isel(timedelta=-1).data
    for model in (l96_model, learned):
      model.assimilate(inputs)
      model.advance()
      observed = model.observe({'slow': {'x': k, 'time': cx.Scalar()}})
      self.assertEqual(observed['slow']['x'].coordinate, k)
      self.assertFalse(np.allclose(observed['slow']['x'].data, x0))
    np.testing.assert_allclose(learned.x.get_value().data, 0.9 * x0, rtol=1e-6)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
