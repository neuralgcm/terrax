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

"""Defines stability metrics that track NaN occurrences in predictions."""

from __future__ import annotations

import dataclasses
import enum
from typing import Sequence

import coordax as cx
import jax.numpy as jnp
from terrax.metrics import base


class NanMode(enum.Enum):
  """Mode for checking NaN values in predictions.

  Attributes:
    ANY: A simulation is considered unstable (NaN) if *any* element is NaN.
    ALL: A simulation is considered unstable (NaN) only if *all* elements are
      NaN.
  """

  ANY = 'any'
  ALL = 'all'


def _dims_suffix(event_dims: Sequence[cx.Coordinate | str] | None) -> str:
  """Returns a string suffix encoding event_dims for unique naming."""
  if event_dims is None:
    return 'all'
  dim_names = []
  for d in event_dims:
    d_dims = d.dims if cx.is_coord(d) else (d,)
    dim_names.extend(d_dims)
  return '_'.join(dim_names) or 'scalar'


def _is_finite_indicator(
    field: cx.Field,
    nan_mode: NanMode,
    event_dims: Sequence[cx.Coordinate | str] | None,
) -> cx.Field:
  """Computes a finite indicator for a single field.

  Args:
    field: Input field to compute indicator values for.
    nan_mode: Whether to check for any or all NaN values.
    event_dims: Dimensions over which to reduce the NaN check. These are the
      dimensions that constitute a single "event" (e.g. spatial dims). If None,
      all dimensions are reduced (the entire field is one event).

  Returns:
    A `cx.Field` with the same non-event dimensions as the input, containing
    1.0 (finite) or 0.0 (NaN) indicators.
  """
  nan_mask = cx.cmap(jnp.isnan)(field)
  if event_dims is None:
    dims_to_untag = tuple(field.dims)
  else:
    dims_to_untag = tuple(d for d in event_dims if cx.contains_dims(field, d))
  if dims_to_untag:
    nan_mask = nan_mask.untag(*dims_to_untag)
  if nan_mode == NanMode.ANY:
    has_nan = cx.cmap(jnp.any)(nan_mask)
  elif nan_mode == NanMode.ALL:
    has_nan = cx.cmap(jnp.all)(nan_mask)
  else:
    raise ValueError(f'Unknown nan_mode: {nan_mode}')
  return cx.cmap(lambda x: 1.0 - x.astype(jnp.float32))(has_nan)


@dataclasses.dataclass
class IsFinite(base.PerVariableStatistic):
  """Statistic that checks whether predictions are finite (non-NaN).

  For each variable, this computes an indicator:
    - 1.0 if the prediction is considered finite (no NaNs per `nan_mode`).
    - 0.0 if the prediction is considered NaN.

  The NaN check is reduced over `event_dims` (the dimensions that define a
  single simulation output, e.g. spatial dimensions). Any remaining dimensions
  (e.g. batch, ensemble) are preserved, producing one indicator per sample.
  Dimensions listed in `event_dims` that are not present on a given variable
  are silently skipped, which naturally handles variables with different
  dimensionality (e.g. surface vs pressure-level variables).

  Since `PerVariableStatistic` is a `Metric`, `IsFinite` can be used directly
  as a metric. When averaged over samples via the aggregation framework, the
  mean gives the fraction of finite predictions per variable.

  Attributes:
    nan_mode: How to determine whether a variable is NaN. See `NanMode`.
    event_dims: Dimensions over which to reduce the NaN check. If None, all
      dimensions are reduced (the entire field is treated as one event).
  """

  nan_mode: NanMode = NanMode.ANY
  event_dims: Sequence[cx.Coordinate | str] | None = None

  def __post_init__(self):
    if self.event_dims is not None:
      self.event_dims = tuple(self.event_dims)

  @property
  def unique_name(self) -> str:
    return f'IsFinite_{self.nan_mode.value}_{_dims_suffix(self.event_dims)}'

  def _compute_per_variable(
      self, predictions: cx.Field, targets: cx.Field
  ) -> cx.Field:
    del targets  # unused.
    return _is_finite_indicator(predictions, self.nan_mode, self.event_dims)


@dataclasses.dataclass
class AllFinite(base.Statistic):
  """Statistic that checks whether all variables are simultaneously finite.

  Unlike `IsFinite` which checks each variable independently, this statistic
  produces a single indicator that is 1.0 only when *every* variable in the
  predictions dict passes the NaN check. This correctly computes per-sample
  cross-variable stability before averaging (unlike taking the min of
  per-variable means, which gives incorrect results when NaN patterns differ
  across variables).

  Since `Statistic` is a `Metric`, `AllFinite` can be used directly as a
  metric. When averaged over samples via the aggregation framework, the mean
  gives the fraction of simulations where all variables are finite.

  Attributes:
    nan_mode: How to determine whether a variable is NaN. See `NanMode`.
    event_dims: Dimensions over which to reduce the NaN check per variable. If
      None, all dimensions of each variable are reduced.
  """

  nan_mode: NanMode = NanMode.ANY
  event_dims: Sequence[cx.Coordinate | str] | None = None

  def __post_init__(self):
    if self.event_dims is not None:
      self.event_dims = tuple(self.event_dims)

  @property
  def unique_name(self) -> str:
    return f'AllFinite_{self.nan_mode.value}_{_dims_suffix(self.event_dims)}'

  def compute(
      self,
      predictions: dict[str, cx.Field],
      targets: dict[str, cx.Field],
  ) -> dict[str, cx.Field]:
    del targets  # unused.
    all_finite = None
    for k in sorted(predictions.keys()):
      var_finite = _is_finite_indicator(
          predictions[k], self.nan_mode, self.event_dims
      )
      if all_finite is None:
        all_finite = var_finite
      else:
        all_finite = cx.cmap(jnp.minimum)(all_finite, var_finite)
    if all_finite is None:
      raise ValueError('AllFinite requires at least one variable.')
    return {'all_variables': all_finite}


# Convenience aliases.
FiniteFraction = IsFinite
SimulationStability = AllFinite
