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
