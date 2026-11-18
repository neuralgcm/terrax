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

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import numpy as np
from terrax.metrics import base
from terrax.metrics import probabilistic_losses
from terrax.metrics import probabilistic_metrics


class ProbabilisticMetricsTest(parameterized.TestCase):

  def test_crps_metric(self):
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 4)
    rng = np.random.default_rng(seed=0)
    predictions = {'x': cx.field(rng.random(e.shape + d.shape), e, d)}
    targets = {'x': cx.field(rng.random(d.shape), d)}
    crps_metric = probabilistic_metrics.CRPS()
    crps_statistics = base.compute_unique_statistics_for_all_metrics(
        {'crps': crps_metric}, predictions, targets
    )
    crps_values = crps_metric.values_from_mean_statistics(crps_statistics)
    skill = crps_statistics['EnergySkill_ensemble_beta_1.0']['x']
    spread = crps_statistics['EnergySpread_ensemble_beta_1.0']['x']
    expected_crps = skill - 0.5 * spread
    cx.testing.assert_fields_allclose(crps_values['x'], expected_crps)
    loss = probabilistic_losses.CRPS()
    cx.testing.assert_fields_allclose(loss.total(crps_values), expected_crps)

    with self.subTest('debug_terms'):
      debug_terms = loss.debug_terms(crps_statistics, crps_values)
      cx.testing.assert_fields_allclose(
          debug_terms['spread_skill_ratio_x'],
          spread / skill,
      )
      cx.testing.assert_fields_allclose(
          debug_terms['relative_crps_x'],
          cx.field(np.full(d.shape, 1.0), d),
      )

  def test_crps_pseudo_huber_kernel(self):
    """CRPS with `delta > 0` uses the pseudo-Huber kernel in both terms."""
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 5)
    rng = np.random.default_rng(seed=0)
    pred_data = rng.normal(size=e.shape + d.shape)
    target_data = rng.normal(size=d.shape)
    predictions = {'x': cx.field(pred_data, e, d)}
    targets = {'x': cx.field(target_data, d)}
    delta = 0.7
    rho = lambda r: np.sqrt(r**2 + delta**2) - delta

    with self.subTest('delta_zero_matches_default'):
      default_crps = probabilistic_metrics.CRPS()
      zero_delta_crps = probabilistic_metrics.CRPS(delta=0.0)
      self.assertEqual(
          {k: v.unique_name for k, v in default_crps.statistics.items()},
          {k: v.unique_name for k, v in zero_delta_crps.statistics.items()},
      )

    crps_metric = probabilistic_metrics.CRPS(delta=delta)
    stats = base.compute_unique_statistics_for_all_metrics(
        {'crps': crps_metric}, predictions, targets
    )
    skill = stats['EnergySkill_ensemble_beta_1.0_delta_0.7']['x']
    spread = stats['EnergySpread_ensemble_beta_1.0_delta_0.7']['x']
    expected_skill = np.mean(rho(pred_data - target_data), axis=0)
    expected_spread = rho(pred_data[0] - pred_data[1])
    np.testing.assert_allclose(skill.data, expected_skill, rtol=1e-5)
    np.testing.assert_allclose(spread.data, expected_spread, rtol=1e-5)
    values = crps_metric.values_from_mean_statistics(stats)
    np.testing.assert_allclose(
        values['x'].data,
        expected_skill - 0.5 * expected_spread,
        rtol=1e-5,
        atol=1e-6,
    )

    with self.subTest('large_delta_approaches_fair_squared_error'):
      # For |errors| << delta, the score tends to the fair squared error of
      # the ensemble mean, which for 2 members is (X - Y)(X' - Y) / (2 delta).
      large_delta = 1e3
      large_delta_crps = probabilistic_metrics.CRPS(delta=large_delta)
      stats = base.compute_unique_statistics_for_all_metrics(
          {'crps': large_delta_crps}, predictions, targets
      )
      values = large_delta_crps.values_from_mean_statistics(stats)
      errors = pred_data - target_data
      expected = errors[0] * errors[1] / (2 * large_delta)
      np.testing.assert_allclose(values['x'].data, expected, rtol=1e-3)

    with self.subTest('continuous_gradient_at_zero_error'):
      skill_stat = probabilistic_metrics.EnergySkill(delta=delta)
      zero_error_preds = np.stack([target_data, target_data])

      def total_skill(pred):
        return skill_stat.compute({'x': cx.field(pred, e, d)}, targets)['x']

      grad = jax.grad(lambda p: total_skill(p).data.sum())(zero_error_preds)
      np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    # TODO(dkochkov): Consider including tests covering: unbiasedness of the
    # 2-member estimator against Monte-Carlo estimates of E[ρ(X - Y)] and
    # E[ρ(X - X')], and properness on a Gaussian toy problem (expected score
    # over a grid of means and spreads minimized at the truth for several
    # delta).

  @parameterized.parameters(
      probabilistic_metrics.EnergySkill,
      probabilistic_metrics.EnergySpread,
      probabilistic_metrics.CRPS,
      probabilistic_metrics.EnergyScore,
  )
  def test_negative_delta_raises_at_construction(self, cls):
    with self.assertRaisesRegex(ValueError, 'delta must be non-negative'):
      cls(delta=-0.1)

  def test_energy_spread_broadcast_target_nans_behavior(self):
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 4)
    predictions = {'x': cx.field(np.ones(e.shape + d.shape), e, d)}
    targets = {'x': cx.field(np.array([np.nan, 1.0, 2.0, 3.0]), d)}

    with self.subTest('no_broadcast_target'):
      spread = probabilistic_metrics.EnergySpread(broadcast_target_nans=False)
      spread_results = spread.compute(predictions, targets)
      self.assertEqual(spread_results['x'].data[0], 0.0)  # no nans.

    with self.subTest('broadcast_target'):
      spread = probabilistic_metrics.EnergySpread(broadcast_target_nans=True)
      spread_results = spread.compute(predictions, targets)
      self.assertTrue(np.isnan(spread_results['x'].data[0]))  # broadcasted.
      self.assertEqual(spread_results['x'].data[1], 0.0)

  def test_crps_large_ensemble(self):
    e, d = cx.SizedAxis('ensemble', 4), cx.SizedAxis('d', 2)
    predictions = {
        'x': cx.field(np.arange(1, 9).reshape((4, 2)).astype(float), e, d)
    }
    targets = {'x': cx.field(np.array([4.0, 5.0]), d)}

    metric = probabilistic_metrics.CRPS()
    stats = base.compute_unique_statistics_for_all_metrics(
        {'crps': metric}, predictions, targets
    )
    values = metric.values_from_mean_statistics(stats)

    expected_spread = cx.field(np.full(d.shape, 20.0 / 6.0), d)
    cx.testing.assert_fields_allclose(
        stats['EnergySpread_ensemble_beta_1.0']['x'], expected_spread
    )

    expected_skill = cx.field(np.full(d.shape, 2.0), d)
    cx.testing.assert_fields_allclose(
        stats['EnergySkill_ensemble_beta_1.0']['x'], expected_skill
    )

    expected_crps = expected_skill - 0.5 * expected_spread
    cx.testing.assert_fields_allclose(values['x'], expected_crps, atol=1e-6)

  def test_sum_loss_of_crps_with_different_kernels(self):
    """Each term of a `SumLoss` is computed from its own skill and spread."""
    e, d = cx.SizedAxis('ensemble', 4), cx.SizedAxis('d', 3)
    rng = np.random.default_rng(seed=0)
    predictions = {'x': cx.field(rng.normal(size=e.shape + d.shape), e, d)}
    targets = {'x': cx.field(rng.normal(size=d.shape), d)}
    terms = {
        'crps': probabilistic_losses.CRPS(),
        'pseudo_huber': probabilistic_losses.CRPS(delta=0.7),
    }
    loss = base.SumLoss(terms, term_weights={'pseudo_huber': 2.0})
    stats = base.compute_unique_statistics_for_all_metrics(
        {'loss': loss}, predictions, targets
    )
    self.assertLen(stats, 4)  # skill and spread of each kernel.
    values = loss.values_from_mean_statistics(stats)
    term_values = {
        name: term.values_from_mean_statistics(stats)
        for name, term in terms.items()
    }

    with self.subTest('values'):
      self.assertEqual(set(values), {'crps.x', 'pseudo_huber.x'})
      for name in terms:
        cx.testing.assert_fields_allclose(
            values[f'{name}.x'], term_values[name]['x']
        )

    with self.subTest('total'):
      expected_total = (
          term_values['crps']['x'] + 2.0 * term_values['pseudo_huber']['x']
      )
      cx.testing.assert_fields_allclose(loss.total(values), expected_total)

    with self.subTest('debug_terms'):
      debug_terms = loss.debug_terms(stats, values)
      for name, term in terms.items():
        term_debug_terms = term.debug_terms(stats, term_values[name])
        for key, expected in term_debug_terms.items():
          cx.testing.assert_fields_allclose(
              debug_terms[f'{name}.{key}'], expected
          )

    with self.subTest('missing_statistics'):
      incomplete_stats = dict(stats)
      del incomplete_stats['EnergySkill_ensemble_beta_1.0']
      with self.assertRaisesRegex(ValueError, 'do not contain all required'):
        loss.values_from_mean_statistics(incomplete_stats)

  @parameterized.parameters(dict(delta=0.0), dict(delta=0.7))
  def test_energy_spread_algorithms(self, delta):
    def make_stat(algorithm):
      return probabilistic_metrics.EnergySpread(
          algorithm=algorithm, delta=delta
      )

    algorithms = ['flip', 'all_pairs', 'ring_chain', 'auto']
    if delta == 0.0:
      algorithms.append('sort')
    else:
      with self.assertRaisesRegex(ValueError, 'sort algorithm only supports'):
        make_stat('sort')
    stats = {algorithm: make_stat(algorithm) for algorithm in algorithms}

    def expected_spread(pred_data):
      # Mean of ρ(x_i - x_j) over all ordered pairs of members i != j.
      n = pred_data.shape[0]
      diffs = pred_data[:, None] - pred_data[None, :]
      rho = np.sqrt(diffs**2 + delta**2) - delta
      return rho.sum(axis=(0, 1)) / (n * (n - 1))

    rng = np.random.default_rng(seed=0)
    d = cx.SizedAxis('d', 4)
    targets = {'x': cx.field(rng.random(d.shape), d)}

    for n in (2, 4):
      e = cx.SizedAxis('ensemble', n)
      pred_data = rng.random(e.shape + d.shape)
      predictions = {'x': cx.field(pred_data, e, d)}
      expected = cx.field(expected_spread(pred_data), d)
      for algorithm, stat in stats.items():
        if algorithm == 'flip' and n != 2:
          continue  # `flip` is only defined for 2 members.
        with self.subTest(n=n, algorithm=algorithm):
          spread = stat.compute(predictions, targets)['x']
          cx.testing.assert_fields_allclose(spread, expected, atol=1e-6)

  @parameterized.parameters(
      dict(n=2, beta=1.0, delta=0.0),
      dict(n=2, beta=1.0, delta=0.5),
      dict(n=4, beta=1.0, delta=0.0),
      dict(n=4, beta=1.5, delta=0.0),
      dict(n=4, beta=0.5, delta=0.0),
      dict(n=4, beta=1.0, delta=0.5),
      dict(n=4, beta=1.5, delta=0.5),
  )
  def test_energy_score_with_event_dims(self, n, beta, delta):
    """Energy score over event dims matches an explicit calculation."""
    e, x = cx.SizedAxis('ensemble', n), cx.SizedAxis('x', 3)
    p, q = cx.SizedAxis('p', 2), cx.SizedAxis('q', 4)
    rng = np.random.default_rng(seed=0)
    # Event dims are not adjacent on predictions, ordered differently on
    # targets and listed in yet another order in `event_dims`.
    pred_data = rng.normal(size=p.shape + e.shape + x.shape + q.shape)
    target_data = rng.normal(size=x.shape + q.shape + p.shape)
    predictions = {'a': cx.field(pred_data, p, e, x, q)}
    targets = {'a': cx.field(target_data, x, q, p)}
    event_dims = ('q', p)

    def rho(diff):  # diff has events along the last axis.
      squared_norm = np.sum(diff**2, axis=-1)
      return (squared_norm + delta**2) ** (beta / 2) - delta**beta

    events_pred = pred_data.transpose(1, 2, 0, 3).reshape(n, 3, 8)  # e, x, pq.
    events_target = target_data.transpose(0, 2, 1).reshape(3, 8)  # x, pq.
    expected_skill = rho(events_pred - events_target).mean(axis=0)
    pair_diffs = events_pred[:, None] - events_pred[None, :]
    expected_spread = rho(pair_diffs).sum(axis=(0, 1)) / (n * (n - 1))
    expected_score = cx.field(expected_skill - 0.5 * expected_spread, x)

    metric = probabilistic_metrics.EnergyScore(
        event_dims=event_dims, beta=beta, delta=delta
    )
    stats = base.compute_unique_statistics_for_all_metrics(
        {'es': metric}, predictions, targets
    )
    values = metric.values_from_mean_statistics(stats)
    with self.subTest('metric_values'):
      cx.testing.assert_fields_allclose(
          values['a'], expected_score, rtol=1e-5, atol=1e-6
      )

    with self.subTest('jit'):

      @jax.jit
      def compute(predictions, targets):
        stats = base.compute_unique_statistics_for_all_metrics(
            {'es': metric}, predictions, targets
        )
        return metric.values_from_mean_statistics(stats)

      cx.testing.assert_fields_allclose(
          compute(predictions, targets)['a'],
          expected_score,
          rtol=1e-5,
          atol=1e-6,
      )

    algorithms = ['all_pairs', 'ring_chain', 'auto']
    if n == 2:
      algorithms.append('flip')
    for algorithm in algorithms:
      with self.subTest(algorithm=algorithm):
        spread_stat = probabilistic_metrics.EnergySpread(
            beta=beta, algorithm=algorithm, delta=delta, event_dims=event_dims
        )
        spread = spread_stat.compute(predictions, targets)['a']
        cx.testing.assert_fields_allclose(
            spread, cx.field(expected_spread, x), rtol=1e-5, atol=1e-6
        )

    loss = probabilistic_losses.EnergyScore(
        event_dims=event_dims, beta=beta, delta=delta
    )
    with self.subTest('loss_total'):
      cx.testing.assert_fields_allclose(
          loss.total(values), expected_score, rtol=1e-5, atol=1e-6
      )

    with self.subTest('loss_debug_terms'):
      debug_terms = loss.debug_terms(stats, values)
      cx.testing.assert_fields_allclose(
          debug_terms['spread_skill_ratio_a'],
          cx.field(expected_spread / expected_skill, x),
          rtol=1e-5,
      )
      cx.testing.assert_fields_allclose(
          debug_terms['relative_energy_score_a'],
          cx.field(np.ones(x.shape), x),
          rtol=1e-5,
      )

  def test_energy_score_without_event_dims_is_crps(self):
    e, d = cx.SizedAxis('ensemble', 4), cx.SizedAxis('d', 5)
    rng = np.random.default_rng(seed=0)
    predictions = {'x': cx.field(rng.normal(size=e.shape + d.shape), e, d)}
    targets = {'x': cx.field(rng.normal(size=d.shape), d)}
    crps = probabilistic_metrics.CRPS()
    energy_score = probabilistic_metrics.EnergyScore()
    self.assertEqual(
        {k: v.unique_name for k, v in crps.statistics.items()},
        {k: v.unique_name for k, v in energy_score.statistics.items()},
    )
    stats = base.compute_unique_statistics_for_all_metrics(
        {'crps': crps, 'es': energy_score}, predictions, targets
    )
    cx.testing.assert_fields_allclose(
        energy_score.values_from_mean_statistics(stats)['x'],
        crps.values_from_mean_statistics(stats)['x'],
    )

  def test_energy_score_is_sensitive_to_dependence(self):
    """Energy score penalizes wrong dependence within events, CRPS does not."""
    e, s = cx.SizedAxis('ensemble', 8), cx.SizedAxis('sample', 20_000)
    p = cx.SizedAxis('p', 2)
    rng = np.random.default_rng(seed=0)
    # Both entries of each target event are equal. `dependent` predictions
    # share this structure, `independent` ones only match the marginals.
    target = np.repeat(rng.normal(size=s.shape + (1,)), 2, axis=-1)
    dependent = np.repeat(rng.normal(size=s.shape + e.shape + (1,)), 2, axis=-1)
    independent = rng.normal(size=s.shape + e.shape + p.shape)
    targets = {'x': cx.field(target, s, p)}
    metrics = {
        'crps': probabilistic_metrics.CRPS(),
        'es': probabilistic_metrics.EnergyScore(event_dims=('p',)),
    }

    def mean_scores(pred_data):
      predictions = {'x': cx.field(pred_data, s, e, p)}
      stats = base.compute_unique_statistics_for_all_metrics(
          metrics, predictions, targets
      )
      return {
          k: float(np.mean(m.values_from_mean_statistics(stats)['x'].data))
          for k, m in metrics.items()
      }

    dependent_scores = mean_scores(dependent)
    independent_scores = mean_scores(independent)
    # Expected CRPS is 1 / sqrt(pi) for both; expected energy scores are
    # sqrt(2 / pi) ~ 0.798 for `dependent` and ~0.857 for `independent`.
    expected_crps, expected_es = np.pi**-0.5, (2 / np.pi) ** 0.5
    self.assertAlmostEqual(dependent_scores['crps'], expected_crps, delta=0.02)
    self.assertAlmostEqual(
        independent_scores['crps'], expected_crps, delta=0.02
    )
    self.assertAlmostEqual(dependent_scores['es'], expected_es, delta=0.02)
    self.assertGreater(independent_scores['es'], dependent_scores['es'] + 0.03)

  def _total_energy_score(self, pred_data, target_data, **kwargs):
    """Sum of energy scores for events along the last axis of the data."""
    e = cx.SizedAxis('ensemble', pred_data.shape[0])
    d, p = cx.SizedAxis('d', 3), cx.SizedAxis('p', 2)
    predictions = {'x': cx.field(pred_data, e, d, p)}
    targets = {'x': cx.field(target_data, d, p)}
    metric = probabilistic_metrics.EnergyScore(event_dims=('p',), **kwargs)
    stats = base.compute_unique_statistics_for_all_metrics(
        {'es': metric}, predictions, targets
    )
    return metric.values_from_mean_statistics(stats)['x'].data.sum()

  @parameterized.parameters(dict(beta=1.0), dict(beta=0.5))
  def test_energy_score_gradient_at_zero_error(self, beta):
    """Gradients are finite when all members coincide with the target."""
    target_data = np.random.default_rng(seed=0).normal(size=(3, 2))
    pred_data = np.stack([target_data, target_data])
    grad = jax.grad(self._total_energy_score)(pred_data, target_data, beta=beta)
    np.testing.assert_array_equal(grad, np.zeros_like(grad))

  def test_energy_score_gradient_for_small_errors(self):
    """Gradients do not vanish when all errors are small."""
    noise = np.random.default_rng(seed=0).normal(size=(4, 3, 2))
    target_data = np.zeros((3, 2))
    # With beta=1 and delta=0 the score is homogeneous of degree 1 in the
    # errors, so its gradient does not depend on their scale.
    grad_fn = jax.grad(self._total_energy_score)
    grad = grad_fn(noise, target_data)
    self.assertGreater(np.linalg.norm(grad), 0.1)
    np.testing.assert_allclose(
        grad_fn(1e-5 * noise, target_data), grad, rtol=1e-4, atol=1e-5
    )

  def test_energy_statistics_event_dims_validation(self):
    e, d = cx.SizedAxis('ensemble', 4), cx.SizedAxis('d', 3)
    p = cx.SizedAxis('p', 2)
    predictions = {'x': cx.field(np.ones(e.shape + d.shape + p.shape), e, d, p)}
    targets = {'x': cx.field(np.ones(d.shape + p.shape), d, p)}

    for cls in (
        probabilistic_metrics.EnergySkill,
        probabilistic_metrics.EnergySpread,
        probabilistic_metrics.EnergyScore,
    ):
      with self.subTest('ensemble_dim_in_event_dims', cls=cls.__name__):
        with self.assertRaisesRegex(ValueError, 'must not include'):
          cls(event_dims=('p', 'ensemble'))

      with self.subTest('repeated_event_dims', cls=cls.__name__):
        with self.assertRaisesRegex(ValueError, 'must not repeat'):
          cls(event_dims=('p', 'p'))
        # A coordinate repeats the name of its dimension.
        with self.assertRaisesRegex(ValueError, 'must not repeat'):
          cls(event_dims=(p, 'd', 'p'))

      with self.subTest('string_event_dims', cls=cls.__name__):
        with self.assertRaisesRegex(TypeError, 'must be a sequence'):
          cls(event_dims='p')

      with self.subTest('one_shot_iterable_event_dims', cls=cls.__name__):
        event_dims = cls(event_dims=(dim for dim in ('p', 'd'))).event_dims
        self.assertEqual(event_dims, ('p', 'd'))

    with self.subTest('beta_out_of_range'):
      with self.assertRaisesRegex(ValueError, 'beta must be in'):
        probabilistic_metrics.EnergyScore(event_dims=('p',), beta=2.5)

    with self.subTest('unique_name_ignores_order_of_event_dims'):
      self.assertEqual(
          probabilistic_metrics.EnergySkill(event_dims=('p', 'd')).unique_name,
          probabilistic_metrics.EnergySkill(event_dims=('d', 'p')).unique_name,
      )

    with self.subTest('sort_algorithm'):
      with self.assertRaisesRegex(ValueError, 'sort algorithm only supports'):
        probabilistic_metrics.EnergySpread(algorithm='sort', event_dims=('p',))

    with self.subTest('missing_in_predictions'):
      spread = probabilistic_metrics.EnergySpread(event_dims=('q',))
      with self.assertRaisesRegex(ValueError, 'are not dimensions of'):
        spread.compute(predictions, targets)

    with self.subTest('missing_in_targets'):
      skill = probabilistic_metrics.EnergySkill(event_dims=('p',))
      targets_without_p = {'x': cx.field(np.ones(d.shape), d)}
      with self.assertRaisesRegex(ValueError, 'are not dimensions of'):
        skill.compute(predictions, targets_without_p)

  def test_energy_statistics_target_nans_with_event_dims(self):
    """A NaN target masks its whole event in both skill and spread."""
    e, d = cx.SizedAxis('ensemble', 2), cx.SizedAxis('d', 3)
    p = cx.SizedAxis('p', 2)
    predictions = {'x': cx.field(np.ones(e.shape + d.shape + p.shape), e, d, p)}
    target_data = np.ones(d.shape + p.shape)
    target_data[1, 0] = np.nan
    targets = {'x': cx.field(target_data, d, p)}
    expected = cx.field(np.array([0.0, np.nan, 0.0]), d)
    skill = probabilistic_metrics.EnergySkill(event_dims=('p',))
    spread = probabilistic_metrics.EnergySpread(
        broadcast_target_nans=True, event_dims=('p',)
    )
    for name, statistic in {'skill': skill, 'spread': spread}.items():
      with self.subTest(name):
        result = statistic.compute(predictions, targets)['x']
        cx.testing.assert_fields_equal(result, expected)

  @parameterized.parameters(
      dict(n=2),
      dict(n=4),
      dict(n=10),
  )
  def test_unbiased_mse_and_ssr(self, n):
    e, d = cx.SizedAxis('ensemble', n), cx.SizedAxis('d', 1)
    rng = np.random.default_rng(seed=0)
    # Sample predictions and targets.
    pred_data = rng.standard_normal((n, 1))
    target_data = rng.standard_normal((1,))
    predictions = {'x': cx.field(pred_data, e, d)}
    targets = {'x': cx.field(target_data, d)}

    # Explicit calculation.
    pred_mean = np.mean(pred_data)
    pred_var = np.var(pred_data, ddof=1)
    biased_mse = (pred_mean - target_data) ** 2
    expected_unbiased_mse = cx.field(
        np.full(d.shape, biased_mse - pred_var / n), d
    )
    expected_ssr = cx.field(
        np.full(d.shape, np.sqrt(pred_var / expected_unbiased_mse.data)), d
    )

    mse_stat = probabilistic_metrics.UnbiasedEnsembleMeanSquaredError()
    mse_val = mse_stat.compute(predictions, targets)['x']
    cx.testing.assert_fields_allclose(mse_val, expected_unbiased_mse, atol=1e-6)

    ssr_metric = probabilistic_metrics.UnbiasedSpreadSkillRatio()
    ssr_stats = base.compute_unique_statistics_for_all_metrics(
        {'ssr': ssr_metric}, predictions, targets
    )
    ssr_val = ssr_metric.values_from_mean_statistics(ssr_stats)['x']
    cx.testing.assert_fields_allclose(ssr_val, expected_ssr, atol=1e-6)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
