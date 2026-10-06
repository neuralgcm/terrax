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

"""Defines classes that implement rescaling schemes for statistics."""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates


def _maybe_compile_time_eval(*values):
  """Returns ensure_compile_time_eval context if no values are JAX tracers."""
  if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(values)):
    return contextlib.nullcontext()
  return jax.ensure_compile_time_eval()


@dataclasses.dataclass
class ScaleFactor(abc.ABC):
  """Abstract class for scaling statistics.

  Scalers are applied to statistics before they are weighted and aggregated.
  This allows for controlling the scale of metric entrees that may depend on
  the their coordinate values, typically associated with spatial or temporal
  coordinates. Whereas `weighting.Weighting` keeps track of the relative
  statistical weights, `Scaler`s allow transforming the statistics. This is
  particularly useful for computing loss terms.

  `scales` are computed based on the input `field`, `field_name` and, in cases
  when the required coordinates are not present on the `field`, scalar
  coordinate values from the `context`. The latter indicates processing of
  statistics along dimensions one slice at a time.
  """

  @abc.abstractmethod
  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return scaling factor for a given field."""
    ...


@dataclasses.dataclass
class ConstantScaler(ScaleFactor):
  """ScaleFactor that returns scales equal to a user-provided constant.

  Attributes:
    constant: A `cx.Field` containing the scaling factor. Its coordinates should
      be alignable with the field being scaled.
    skip_missing: If True, inputs without a matching coordinates will return a
      scale of 1.0, otherwise an error is raised.
  """

  constant: cx.Field
  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Returns the user-provided constant field for scaling."""
    del field_name, context  # unused.
    if all(d in field.dims for d in self.constant.dims):
      return self.constant
    if self.skip_missing:
      return cx.field(1.0)
    else:
      raise ValueError(
          f'{field=} does not have all coordinates in {self.constant=}.'
      )


@dataclasses.dataclass
class PerVariableScaler(ScaleFactor):
  """ScaleFactor that returns scales from `scalers_by_name[field_name]`."""

  scalers_by_name: dict[str, ScaleFactor]
  default_scaler: ScaleFactor | None = None

  def scales(  # pyrefly: ignore[bad-override]
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field | float:
    """Return scales for `field` computed by a scaler for `field_name`."""
    if field_name is None:
      raise ValueError('PerVariableScaler requires a `field_name`.')

    scaler = self.scalers_by_name.get(field_name)
    if scaler is not None:
      return scaler.scales(field, field_name, context)

    if self.default_scaler is not None:
      return self.default_scaler.scales(field, field_name, context)

    raise KeyError(
        f'"{field_name}" not found in scalers_by_name and no default_scaler is'
        ' set.'
    )

  @classmethod
  def from_constants(
      cls,
      variable_weights: dict[str, float | cx.Field],
      default_scaler: ScaleFactor | None = None,
  ) -> PerVariableScaler:
    """Returns a PerVariableScaler with ConstantScalers."""
    scalers = {
        name: ConstantScaler(constant=w if cx.is_field(w) else cx.field(w))
        for name, w in variable_weights.items()
    }
    return cls(scalers_by_name=scalers, default_scaler=default_scaler)  # pyrefly: ignore[bad-argument-type]


@dataclasses.dataclass
class GridAreaScaler(ScaleFactor):
  """ScaleFactor that returns scales proportional to the area of grid cells.

  This weighting works with both `LonLatGrid` and `SphericalHarmonicGrid`.

  For `LonLatGrid`, weights are approximated by cos(lat), which are proportional
  to the proper quadrature weights of Gaussian grids. This ensures that grid
  cells near the poles have smaller weights than those near the equator.

  For `SphericalHarmonicGrid`, the basis functions are orthonormal, so uniform
  weights (1.0) are returned.

  If skip_missing attribute is set to True, fields without a grid will return
  a weight of 1.0, otherwise an error is raised.
  """

  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    del context  # unused.
    lon_lat_dims = ('longitude', 'latitude')
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if cx.contains_dims(field, *lon_lat_dims):
      grid = cx.coords.extract(field.coordinate, coordinates.LonLatGrid)

      def get_weight(x):
        # Latitudes are in degrees, convert to radians for cosine.
        lat = jnp.deg2rad(x)
        pi_over_2 = jnp.array([np.pi / 2])
        lat_cell_bounds = jnp.concatenate(
            [-pi_over_2, (lat[:-1] + lat[1:]) / 2, pi_over_2]
        )
        upper = lat_cell_bounds[1:]
        lower = lat_cell_bounds[:-1]
        return jnp.sin(upper) - jnp.sin(lower)

      get_weight = cx.cmap(get_weight)
      lats = grid.fields['latitude']
      lat_ax = lats.coordinate
      with jax.ensure_compile_time_eval():
        weights = get_weight(grid.fields['latitude'].untag(lat_ax)).tag(lat_ax)
        weights = weights.broadcast_like(grid)
    elif cx.contains_dims(field, *ylm_dims):
      ylm_grid = cx.coords.extract(
          field.coordinate, coordinates.SphericalHarmonicGrid
      )
      # avoid counting padding towards overall weight by using mask.
      weights = ylm_grid.fields['mask'].astype(jnp.float32)
    else:
      if self.skip_missing:
        weights = cx.field(1.0)
      else:
        raise ValueError(f'No LonLatGrid or SphericalHarmonicGrid on {field=}')
    return weights


@dataclasses.dataclass
class PressureLevelAtmosphericMassScaler(ScaleFactor):
  """ScaleFactor that returns scales proportional to pressure level thickness.

  This scaling results in statistics that upon the sum would represent a
  discrete integral over pressure (or approximate mass intergral) of the
  corresponding quantity.
  """

  standard_pressure: float = 1013.25  # standard pressure in hPa.

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Return weights extracted from the pressure level coordinate."""
    del field_name, context  # unused.
    with jax.ensure_compile_time_eval():
      if 'pressure' not in field.dims:
        return cx.field(1.0)

      pressure = field.axes['pressure']
      padded = np.concatenate([
          np.asarray([0.0]),
          pressure.centers,  # pyrefly: ignore[missing-attribute]
          np.asarray([self.standard_pressure]),
      ])
      # thickness is estimated as 0.5 * |p_{k+1} - p_{k-1}|.
      thickness = (np.roll(padded, -1) - np.roll(padded, 1))[1:-1] / 2
      return cx.field(thickness, pressure)


@dataclasses.dataclass
class WavenumberScaler(ScaleFactor):
  """ScaleFactor that returns fixed scale equal to "number of ylm_modes" / 4pi.

  For fields with `SphericalHarmonicGrid` dimension, this scaler rescales the
  statistics by the number of spherical harmonic modes divided by 4pi. This
  rescaling modifies the statistics from computing per-mode mean that is
  resolution-dependent to spatial mean that is resolution-invariant.

  Attributes:
    skip_missing: If True, fields without a grid will return a scale of 1.0.
  """

  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    del field_name, context  # unused.
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if not cx.contains_dims(field, *ylm_dims):
      if self.skip_missing:
        return cx.field(1.0)
      raise ValueError(f'No SphericalHarmonicGrid on {field=}')

    ylm_grid = cx.coords.extract(
        field.coordinate, coordinates.SphericalHarmonicGrid
    )
    with jax.ensure_compile_time_eval():
      return cx.field(ylm_grid.fields['mask'].data.sum() / (4 * np.pi))


@dataclasses.dataclass
class SigmoidWavenumberScaler(ScaleFactor):
  """ScaleFactor that returns wavenumber weights following a sigmoid profile.

  For fields with a `SphericalHarmonicGrid` coordinate, this scaler returns
  total-wavenumber-dependent weights following a smooth low-pass sigmoid profile
  that starts at 1.0 at l = 0, transitions around inflection_factor * l_cutoff
  with width width_factor * l_cutoff, and strictly terminates (0.0) for
  l >= l_cutoff, while masking out padded modes via `ylm_grid.fields['mask']`.

  The cutoff wavenumber l_cutoff is determined by `cutoff_wavenumber` and
  `cutoff_fraction`, at least one of which must be set:
    * `cutoff_wavenumber` as a number sets l_cutoff directly.
    * `cutoff_wavenumber` as a dict maps `SphericalHarmonicGrid`s to l_cutoff.
      Grids are matched by resolution, ignoring padding and the spherical
      harmonics method. If the grid is not in the dict, l_cutoff falls back to
      `cutoff_fraction` if set, otherwise an error is raised.
    * `cutoff_fraction` alone sets l_cutoff as a fraction of the grid's maximum
      wavenumber.

  Attributes:
    cutoff_wavenumber: Total wavenumber at and above which weights are zero, or
      a mapping from `SphericalHarmonicGrid` to such cutoff.
    cutoff_fraction: Cutoff as a fraction of the grid's maximum wavenumber. Used
      when `cutoff_wavenumber` is None or does not contain the grid.
    inflection_factor: Inflection point l_0 as a fraction of l_cutoff.
    width_factor: Transition width w as a fraction of l_cutoff.
    skip_missing: If True, fields without a SphericalHarmonicGrid get scale 1.0.
  """

  cutoff_wavenumber: (
      float | dict[coordinates.SphericalHarmonicGrid, float] | None
  ) = None
  cutoff_fraction: float | None = None
  inflection_factor: float = 0.7
  width_factor: float = 0.1
  skip_missing: bool = True

  def __post_init__(self):
    if self.cutoff_wavenumber is None and self.cutoff_fraction is None:
      raise ValueError(
          'At least one of `cutoff_wavenumber` or `cutoff_fraction` must be'
          ' set.'
      )
    if self.cutoff_fraction is not None and not isinstance(
        self.cutoff_wavenumber, (dict, type(None))
    ):
      raise ValueError(
          '`cutoff_fraction` is only used as a fallback for a dict'
          f' `cutoff_wavenumber`, got {self.cutoff_wavenumber=} and'
          f' {self.cutoff_fraction=}.'
      )

  def _get_cutoff(self, ylm_grid: coordinates.SphericalHarmonicGrid) -> float:
    """Returns the cutoff wavenumber for `ylm_grid`."""
    resolution = lambda g: (g.longitude_wavenumbers, g.total_wavenumbers)
    if isinstance(self.cutoff_wavenumber, dict):
      for key_grid, cutoff in self.cutoff_wavenumber.items():
        if resolution(key_grid) == resolution(ylm_grid):
          return float(cutoff)
      if self.cutoff_fraction is None:
        raise ValueError(
            f'{ylm_grid=} not found in {self.cutoff_wavenumber=} and'
            ' `cutoff_fraction` is not set.'
        )
    elif self.cutoff_wavenumber is not None:
      return float(self.cutoff_wavenumber)
    grid_max_wavenumber = ylm_grid.total_wavenumbers - 2
    return float(self.cutoff_fraction * grid_max_wavenumber)  # pyrefly: ignore[unsupported-operation]

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    del field_name, context  # unused.
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if not cx.contains_dims(field, *ylm_dims):
      if self.skip_missing:
        return cx.field(1.0)
      raise ValueError(f'No SphericalHarmonicGrid on {field=}')

    ylm_grid = cx.coords.extract(
        field.coordinate, coordinates.SphericalHarmonicGrid
    )
    l_cutoff = self._get_cutoff(ylm_grid)
    with jax.ensure_compile_time_eval():
      l_0 = self.inflection_factor * l_cutoff
      width = self.width_factor * l_cutoff
      sigmoid = lambda l: 1.0 / (1.0 + jnp.exp((l - l_0) / width))
      s_0, s_cutoff = sigmoid(0.0), sigmoid(l_cutoff)

      def _profile(l):
        profile = jnp.maximum(0.0, sigmoid(l) - s_cutoff) / (s_0 - s_cutoff)
        return jnp.where(l < l_cutoff, profile, 0.0)

      ls = ylm_grid.fields['total_wavenumber'].astype(jnp.float32)
      return ylm_grid.fields['mask'].astype(jnp.float32) * cx.cmap(_profile)(ls)


@dataclasses.dataclass
class CoordinateMaskScaler(ScaleFactor):
  """ScaleFactor that that returns masked/unmasked scales based on masking.

  This scaler is parameterized by a `mask_coord`. For each dimension in
  `mask_coord.dims`, it checks for a matching coordinate in the input `field`
  or, if not present on the `field`, a scalar value in the `context`.

  If a matching coordinate is found on the `field`, it returns `masked_value`
  for each value in that coordinate that is present in the `mask_coord`, and
  `unmasked_value` otherwise.

  If no matching coordinate is found of the `field`, the `context` is searched
  for scalar value using the dimension name, indicating in-context processing
  of `field` slices. If found, it returns `masked_value` if the value from the
  context (context[dim_name]) is present in `mask_coord`, and `unmasked_value`
  otherwise.

  If coordinates for multiple dimensions are found, the resulting masks are
  multiplied.

  If `skip_missing` is True, dimensions from `mask_coord` not found in the
  `field` or `context` are ignored (effectively resulting in `unmasked_value`).
  Otherwise, a ValueError is raised.
  """

  mask_coord: cx.Coordinate
  masked_value: float = 0.0
  unmasked_value: float = 1.0
  skip_missing: bool = True

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Computes scales based on coordinate values."""
    del field_name  # unused.
    all_masks = []
    for dim_name in self.mask_coord.dims:
      in_context = context and dim_name in context
      in_field = dim_name in field.axes

      if in_context and in_field:
        raise ValueError(
            f'Coordinate for {dim_name!r} found both in context and field.'
        )

      if not in_context and not in_field:
        if self.skip_missing:
          continue
        raise ValueError(
            f'Coordinate for {dim_name!r} not found on {field=} or in context.'
        )

      mask_values_field = self.mask_coord.fields[dim_name]  # pyrefly: ignore[bad-index]
      if in_context:
        current_value = context[dim_name]  # pyrefly: ignore[bad-index, unsupported-operation]
        if current_value.ndim != 0:
          raise ValueError(
              f'Expected scalar {dim_name!r} in context, got '
              f'{current_value.shape=}'
          )
        mask_values = mask_values_field.untag(dim_name)  # pyrefly: ignore[bad-argument-type]
        with _maybe_compile_time_eval(current_value, mask_values):
          is_present = (current_value == mask_values).data.any()
          mask = cx.field(is_present)
        all_masks.append(mask)
      elif in_field:
        coord_from_field = field.axes[dim_name]  # pyrefly: ignore[bad-index]
        field_values = coord_from_field.fields[dim_name]  # pyrefly: ignore[bad-index]
        mask_values_data = mask_values_field.untag(dim_name)  # pyrefly: ignore[bad-argument-type]
        with jax.ensure_compile_time_eval():
          is_present_broadcasted = field_values == mask_values_data
          mask_for_dim = cx.cmap(lambda x: x.any())(is_present_broadcasted)
        all_masks.append(mask_for_dim)

    if not all_masks:
      return cx.field(self.unmasked_value)

    with _maybe_compile_time_eval(*all_masks):
      final_mask = functools.reduce(lambda x, y: x & y, all_masks)
      masked_v, unmasked_v = self.masked_value, self.unmasked_value
      where_fn = lambda x: (
          jnp.where(x, masked_v, unmasked_v).astype(jnp.float32)
      )
      return cx.cmap(where_fn)(final_mask)


@dataclasses.dataclass
class LeadTimeScaler(ScaleFactor):
  """ScaleFactor that returns scales equal to 1/(std of a random walk spread).

  It computes scales that are inversely proportional to the anticipated standard
  deviation of errors at a given lead time, assuming random-walk-like error
  growth. The weights are derived from the `TimeDelta` coordinate of the input
  field or `context` when producing weights for statistics slices along the
  `timedelta` dimension.

  Attributes:
    base_squared_error_in_hours: Number of hours before assumed variance starts
      growing (almost) linearly.
    asymptotic_squared_error_in_hours: Number of hours before assumed variance
      slows its growth. Set to None (the default) if variance grows
      indefinitely.
    normalize_weights: Whether to normalizing scaling factors such that the
      square of all of the weights add up to 1.
    skip_missing: If True, fields without a matching coordinate will return a
      scale of 1.0, otherwise an error is raised.
    weights_power: Optional power to which to raise the weights. Can be used to
      scale statistics that grow faster with leadtime (e.g. SquaredError).
  """

  base_squared_error_in_hours: float
  asymptotic_squared_error_in_hours: float | None = None
  normalize_weights: bool = True
  skip_missing: bool = True
  weights_power: float | None = None

  def _compute_inv_variance(self, t):
    """Computes the unnormalized 1/std_dev weights."""
    if self.asymptotic_squared_error_in_hours is not None:
      t = t / (1 + t / self.asymptotic_squared_error_in_hours)

    # Variance is assumed to grow linearly with our transformed time `t`.
    # weight ~ 1 / std_dev ~ 1 / sqrt(variance)
    return 1 / (1 + t / self.base_squared_error_in_hours)

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Computes scale factors for statistics."""
    del field_name  # unused.
    time_coord = field.axes.get('timedelta', None)
    from_context = time_coord is None and context and 'timedelta' in context

    if time_coord is None and not from_context:
      if self.skip_missing:
        return cx.field(1.0)
      raise ValueError(f'TimeDelta coord not found on {field=} or in context')

    one_hr_delta = np.timedelta64(1, 'h')
    if from_context:
      assert isinstance(context, dict)  # make pytype happy.
      all_timedeltas = None  # only used when normalize_weights is True.
      if self.normalize_weights:
        if 'times' not in context:
          raise ValueError(
              'Both "timedelta" and "times" must be present in the context'
              f' when normalize_weights is True, but got: {context.keys()=}'
          )
        all_timedeltas = context['times'].data / one_hr_delta  # pyrefly: ignore[unsupported-operation]
      timedelta_now = context['timedelta'].data
      if timedelta_now.ndim != 0:
        raise ValueError(
            f'Expected scalar timedelta in context, got {timedelta_now.shape=}'
        )
      t = timedelta_now / one_hr_delta  # pyrefly: ignore[unsupported-operation]
    else:
      t = time_coord.deltas / one_hr_delta  # pyrefly: ignore[missing-attribute]
      all_timedeltas = t

    if self.normalize_weights:
      with _maybe_compile_time_eval(all_timedeltas):
        norm_const = self._compute_inv_variance(all_timedeltas).sum()
    else:
      norm_const = 1.0

    with _maybe_compile_time_eval(t, norm_const):
      inv_variance = self._compute_inv_variance(t) / norm_const
      inv_variance_sqrt = jnp.sqrt(inv_variance)
      if self.weights_power is not None:
        inv_variance_sqrt = inv_variance_sqrt**self.weights_power
      if from_context:
        return cx.field(inv_variance_sqrt)
      return cx.field(inv_variance_sqrt, time_coord)


@dataclasses.dataclass
class GeneralizedLeadTimeScaler(ScaleFactor):
  """ScaleFactor for reweighting loss based on lead-time error growth.

  Computes scales inversely proportional to the anticipated growth of error
  (assuming random-walk-like behavior). The scaling is normalized, with default
  mean scale being 1 or a value that varies from 1 to `asymptotic_norm` as a
  function of total lead-time. The rate at which longer lead-time is discounted
  can be adjusted via `weights_power` argument. This scaler can be used with
  both in-context and single pass evaluation. When used in-context, requires
  `timedelta` and `times` that present the current in context lead-time and the
  full time-series.

  Attributes:
    base_squared_error_in_hours: Hours before ~linear variance growth starts.
      Can be a scalar float or a `cx.Field` aligned with non-temporal dimensions
      of the input field (e.g. `pressure` or `total_wavenumber`).
    asymptotic_squared_error_in_hours: Hours before variance starts to plateau.
      Can be a scalar float, a `cx.Field`, or None.
    skip_missing: If True, fields without a timedelta coordinate get scale 1.0.
    weights_power: Optional power to raise the weights. Can be a scalar float or
      a `cx.Field` to scale statistics that grow at different rates across
      coordinates.
    asymptotic_norm: If set, the normalization of the mean scale is set to (1 +
      asymptotic_norm * ratio) / (1 + ratio), where ratio is the ratio of the
      total lead-time to the `norm_transition_timescale_in_hours`.
    norm_transition_power: Power to raise the ratio in norm calculation.
    norm_transition_timescale_in_hours: Number of hours at which norm crosses
      `(1 + asymptotic_norm) / 2` value.
  """

  base_squared_error_in_hours: float | cx.Field
  asymptotic_squared_error_in_hours: float | cx.Field | None = None
  skip_missing: bool = True
  weights_power: float | cx.Field | None = None
  asymptotic_norm: float | None = None
  norm_transition_power: float = 1.0
  norm_transition_timescale_in_hours: float | None = None

  def _compute_raw_weights(self, t: cx.Field) -> cx.Field:
    """Computes the unnormalized 1/std_dev weights."""
    if self.asymptotic_squared_error_in_hours is not None:
      t = t / (1 + t / self.asymptotic_squared_error_in_hours)

    # Variance is assumed to grow linearly with our transformed time `t`.
    # weight ~ 1 / std_dev ~ 1 / sqrt(variance)
    inv_variance = 1 / (1 + t / self.base_squared_error_in_hours)
    weights = cx.cmap(jnp.sqrt)(inv_variance)

    if self.weights_power is not None:
      weights = weights**self.weights_power
    return weights

  def _compute_normalization_scale(
      self, max_t: cx.Field
  ) -> float | cx.Field:
    """Computes the target normalization scale."""
    if self.asymptotic_norm is None:
      return 1.0
    if self.norm_transition_timescale_in_hours is None:
      raise ValueError(
          '`norm_transition_timescale_in_hours` must be provided '
          'when `asymptotic_norm` is set.'
      )
    ratio = (
        max_t / self.norm_transition_timescale_in_hours
    ) ** self.norm_transition_power
    return (1.0 + self.asymptotic_norm * ratio) / (1.0 + ratio)

  def scales(
      self,
      field: cx.Field,
      field_name: str | None = None,
      context: dict[str, cx.Field] | None = None,
  ) -> cx.Field:
    """Computes scale factors for statistics."""
    del field_name  # unused.

    time_coord = field.axes.get('timedelta', None)
    from_context = time_coord is None and context and 'timedelta' in context

    if time_coord is None and not from_context:
      if self.skip_missing:
        return cx.field(1.0)
      raise ValueError(f'TimeDelta coord not found on {field=} or in context')

    params = (
        self.base_squared_error_in_hours,
        self.asymptotic_squared_error_in_hours,
        self.weights_power,
    )
    for param in params:
      if cx.is_field(param) and not set(param.dims).issubset(field.dims):
        raise ValueError(
            f'Parameter {param=} has dimensions not present in {field=}.'
        )

    one_hr_delta = np.timedelta64(1, 'h')
    if from_context:
      assert isinstance(context, dict)  # make pytype happy.
      if 'times' not in context:
        raise ValueError(
            'Both "timedelta" and "times" must be present in the context, but'
            f' got: {context.keys()=}'
        )
      timedelta_now = context['timedelta']
      all_timedeltas = context['times']
      if timedelta_now.ndim != 0:
        raise ValueError(
            f'Expected scalar timedelta in context, got {timedelta_now.shape=}'
        )
      if all_timedeltas.ndim != 1:
        raise ValueError(
            f'Expected 1D times in context, got {all_timedeltas.shape=}'
        )
      with _maybe_compile_time_eval(timedelta_now):
        t = timedelta_now / one_hr_delta  # pyrefly: ignore[unsupported-operation]
      with _maybe_compile_time_eval(all_timedeltas):
        all_times = all_timedeltas / one_hr_delta  # pyrefly: ignore[unsupported-operation]
    else:
      all_timedeltas = time_coord.fields['timedelta']  # pyrefly: ignore[missing-attribute]
      if all_timedeltas.ndim != 1:
        raise ValueError(
            f'Expected 1D timedelta coordinate, got {all_timedeltas.shape=}'
        )
      with jax.ensure_compile_time_eval():
        all_times = all_timedeltas / one_hr_delta  # pyrefly: ignore[unsupported-operation]
      t = all_times

    with _maybe_compile_time_eval(all_times, *params):
      t_coord = all_times.coordinate
      max_t = cx.cmap(jnp.max)(all_times.untag(t_coord))
      norm_scale = self._compute_normalization_scale(max_t)
      raw_weights = self._compute_raw_weights(all_times)
      norm_const = cx.cmap(jnp.mean)(raw_weights.untag(t_coord))

    with _maybe_compile_time_eval(t, norm_scale, norm_const, *params):
      raw_weight = (
          self._compute_raw_weights(t) if from_context else raw_weights
      )
      return raw_weight * norm_scale / norm_const
