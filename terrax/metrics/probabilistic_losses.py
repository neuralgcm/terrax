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

"""Defines probabilistic losses."""

from __future__ import annotations
import collections
import dataclasses
import coordax as cx
from terrax.metrics import base
from terrax.metrics import probabilistic_metrics


def _skill_spread_debug_terms(
    loss: CRPS | EnergyScore,
    mean_stats: dict[str, dict[str, cx.Field]],
    metric_values: dict[str, cx.Field],
    name: str,
) -> dict[str, cx.Field]:
  """Returns relative losses and spread-skill ratios of an energy-like loss.

  Args:
    loss: Loss that is computed from `skill` and `spread` statistics.
    mean_stats: Aggregated statistics keyed by their unique names.
    metric_values: Values of the loss for each variable.
    name: Name of the loss used in the keys of relative losses.

  Returns:
    Debug terms `relative_<name>_<variable>` and `spread_skill_ratio_<variable>`
    for each variable.
  """
  total = loss.total(metric_values)
  mean_stats = {
      k: mean_stats[v.unique_name] for k, v in loss.statistics.items()
  }
  skills = mean_stats['skill']
  spreads = mean_stats['spread']
  weighted_spreads = {
      k: loss.spread_term_weight * v for k, v in spreads.items()
  }
  ws = collections.defaultdict(lambda: 1.0) | (loss.variable_weights or {})
  relative_losses = {
      f'relative_{name}_{k}': ws[k] * (skills[k] - weighted_spreads[k]) / total
      for k in skills
  }
  skill_spread_ratio = {
      f'spread_skill_ratio_{k}': spreads[k] / skills[k] for k in skills
  }
  return relative_losses | skill_spread_ratio


@dataclasses.dataclass
class CRPS(probabilistic_metrics.CRPS, base.PerVariableLoss):
  """CRPS loss."""

  def debug_terms(
      self,
      mean_stats: dict[str, dict[str, cx.Field]],
      metric_values: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Returns relative per-variable losses and spread-skill ratios."""
    return _skill_spread_debug_terms(self, mean_stats, metric_values, 'crps')


@dataclasses.dataclass
class EnergyScore(probabilistic_metrics.EnergyScore, base.PerVariableLoss):
  """Energy score loss."""

  def debug_terms(
      self,
      mean_stats: dict[str, dict[str, cx.Field]],
      metric_values: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    """Returns relative per-variable losses and spread-skill ratios."""
    return _skill_spread_debug_terms(
        self, mean_stats, metric_values, 'energy_score'
    )
