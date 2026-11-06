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

"""Modules that interpolate data from one coordinate system to another."""

import abc
import dataclasses
import functools
from typing import Literal, Protocol, overload

import coordax as cx
from dinosaur import horizontal_interpolation
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates
from terrax.core import patch_geometry
from terrax.core import typing


class BaseRegridder(Protocol):
  """Base class for regridders."""

  @overload
  def __call__(self, inputs: cx.Field) -> cx.Field:
    ...

  @overload
  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def __call__(
      self, inputs: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    ...


@dataclasses.dataclass(frozen=True)
class SpectralRegridder:
  """Regrid between spherical harmonic grids with truncation or zero padding."""

  target_coords: coordinates.SphericalHarmonicGrid

  def truncate_to_target_wavenumbers(
      self,
      field: cx.Field,
  ) -> cx.Field:
    """Interpolates to lower resolution spherical harmonic by truncation."""
    # TODO(dkochkov) consider using coordinate values to inform truncation.
    target_lon_wavenumbers = self.target_coords.sizes['longitude_wavenumber']
    target_total_wavenumbers = self.target_coords.sizes['total_wavenumber']
    lon_slice = slice(0, target_lon_wavenumbers)
    total_slice = slice(0, target_total_wavenumbers)
    field = field.untag('longitude_wavenumber', 'total_wavenumber')
    result = cx.cpmap(lambda x: x[lon_slice, total_slice])(field)
    return result.tag(self.target_coords)

  def pad_to_target_wavenumbers(
      self,
      field: cx.Field,
  ) -> cx.Field:
    """Interpolates to higher resolution spherical harmonic by zero-padding."""
    # TODO(dkochkov) use `sizes` on coords to carry shape of dims info.
    input_lon_k = field.coord_fields['longitude_wavenumber'].shape[0]
    input_total_k = field.coord_fields['total_wavenumber'].shape[0]
    target_lon_k = self.target_coords.sizes['longitude_wavenumber']
    target_total_k = self.target_coords.sizes['total_wavenumber']
    pad_lon = (0, target_lon_k - input_lon_k)
    pad_total = (0, target_total_k - input_total_k)
    pad_fn = lambda x: jnp.pad(x, pad_width=(pad_lon, pad_total))
    field = field.untag('longitude_wavenumber', 'total_wavenumber')
    result = cx.cpmap(pad_fn)(field)
    return result.tag(self.target_coords)

  def interpolate_field(self, field: cx.Field) -> cx.Field:
    """Interpolates a single field."""
    # TODO(dkochkov) Check that inputs.coords includes SphericalHarmonicGrid.
    input_lon_k = field.coord_fields['longitude_wavenumber'].shape[0]
    input_total_k = field.coord_fields['total_wavenumber'].shape[0]
    target_lon_k = self.target_coords.sizes['longitude_wavenumber']
    target_total_k = self.target_coords.sizes['total_wavenumber']
    if (input_total_k < target_total_k) and (input_lon_k <= target_lon_k):
      return self.pad_to_target_wavenumbers(field)
    elif (input_total_k >= target_total_k) and (input_lon_k >= target_lon_k):
      return self.truncate_to_target_wavenumbers(field)
    else:
      raise ValueError(
          'Incompatible horizontal coordinates with shapes '
          f'{field.dims=} with {field.shape=}, '
          f'{self.target_coords.dims=} with {self.target_coords.shape=}, '
      )

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    """Interpolates fields in `inputs`."""
    is_field = lambda x: isinstance(x, cx.Field)
    return jax.tree.map(
        self.interpolate_field,
        inputs,
        is_leaf=is_field,
    )


@dataclasses.dataclass(frozen=True)
class LonLatRegridder(abc.ABC):
  """Base class for regridders between two LonLatGrids.

  Regridded values are weighted averages of source values along longitude and
  latitude, with weights of shape (target, source) provided by subclasses. The
  weights must only depend on the source and target grids, since they are
  computed once per grid pair and reused across calls.
  """

  target_grid: coordinates.LonLatGrid
  skipna: bool = False

  @abc.abstractmethod
  def lat_weights(self, source_grid: coordinates.LonLatGrid) -> typing.Array:
    """Regridding weights for the latitude dimension."""

  @abc.abstractmethod
  def lon_weights(self, source_grid: coordinates.LonLatGrid) -> typing.Array:
    """Regridding weights for the longitude dimension."""

  def _weighted_mean(
      self, data: jax.Array, source_grid: coordinates.LonLatGrid
  ) -> jax.Array:
    """Applies regridding weights to a 2D (lon, lat) array."""
    lon_w, lat_w = _cached_weights(self, source_grid)
    return jnp.einsum(
        'ab,cd,bd->ac', lon_w, lat_w, data, precision='float32'
    )

  def _regrid_2d(
      self, array: jax.Array, source_grid: coordinates.LonLatGrid
  ) -> jax.Array:
    """Applies horizontal regridding to a 2D (lon, lat) array."""
    not_nulls = jnp.logical_not(jnp.isnan(array))
    mean = self._weighted_mean(jnp.where(not_nulls, array, 0), source_grid)
    not_null_fraction = self._weighted_mean(not_nulls, source_grid)
    if self.skipna:
      is_valid = not_null_fraction > 0.0
    else:
      # The tolerance absorbs float32 round-off in weights of grids with
      # coinciding cell edges, which would otherwise spread NaNs to neighbors.
      is_valid = jnp.isclose(not_null_fraction, 1.0, rtol=1e-3)
    safe_fraction = jnp.where(is_valid, not_null_fraction, 1.0)
    return jnp.where(is_valid, mean / safe_fraction, jnp.nan)

  @overload
  def __call__(self, inputs: cx.Field) -> cx.Field:
    ...

  @overload
  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def __call__(
      self, inputs: cx.Field | dict[str, cx.Field]
  ) -> cx.Field | dict[str, cx.Field]:
    """Regrids a Field or dictionary of fields to the target grid."""
    if isinstance(inputs, dict):
      return {k: self(v) for k, v in inputs.items()}

    source_grid = cx.coords.extract(inputs.coordinate, coordinates.LonLatGrid)
    if (
        source_grid.lon_lat_padding != (0, 0)
        or self.target_grid.lon_lat_padding != (0, 0)
    ):
      raise ValueError(
          'LonLatRegridder requires unpadded source and target grids, got '
          f'{source_grid.lon_lat_padding=} and '
          f'{self.target_grid.lon_lat_padding=}.'
      )
    x = inputs.untag(source_grid)
    regrid_fn = cx.cpmap(self._regrid_2d)
    x = regrid_fn(x, source_grid)
    return x.tag(self.target_grid)


@functools.cache
def _cached_weights(
    regridder: LonLatRegridder, source_grid: coordinates.LonLatGrid
) -> tuple[jax.Array, jax.Array]:
  """Returns (lon, lat) weights of `regridder`, computed once per grid pair.

  Reusing the same weight arrays avoids recomputing them whenever a regridder is
  traced and lets JAX keep a single copy of the weights in programs that regrid
  multiple fields, which reduces compilation time for large grids.

  Args:
    regridder: Regridder that computes the weights.
    source_grid: Grid of the regridded inputs.
  """
  with jax.ensure_compile_time_eval():
    lon_weights = jnp.asarray(regridder.lon_weights(source_grid), jnp.float32)
    lat_weights = jnp.asarray(regridder.lat_weights(source_grid), jnp.float32)
  return lon_weights, lat_weights


def _linear_interp_weights(
    target: np.ndarray, source: np.ndarray, period: float | None = None
) -> np.ndarray:
  """Returns weights `w` such that `w @ f == np.interp(target, source, f)`.

  Weights are computed with NumPy, which avoids compiling interpolation code
  for accelerators when weights are evaluated at trace time.

  Args:
    target: Points at which to interpolate.
    source: Increasing points at which values `f` are given; `np.interp`
      silently returns incorrect values otherwise.
    period: Optional period of a periodic coordinate.

  Returns:
    Weights with shape `(target.size, source.size)`.
  """
  # np.interp is linear in `f`, so interpolating unit vectors yields `w`.
  unit_vectors = np.eye(source.size)
  return np.stack(
      [np.interp(target, source, f, period=period) for f in unit_vectors],
      axis=1,
  )


def _linear_interp_neighbors(
    target: np.ndarray, source: np.ndarray, period: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
  """Returns indices and weights of the source neighbors of `target` points.

  Interpolated values `(f[indices] * weights).sum(-1)` are equal to
  `np.interp(target, source, f, period=period)`, i.e. values are clamped to the
  outermost source points unless `period` is provided.

  Args:
    target: Points at which to interpolate.
    source: Increasing points at which values `f` are given.
    period: Optional period of a periodic coordinate. `source` points must span
      less than one period.

  Returns:
    Integer indices and weights, both with shape `(target.size, 2)`.
  """
  size = source.size
  if period is None:
    target = np.clip(target, source[0], source[-1])
  else:
    # Map targets into [source[0], source[0] + period) and add the first source
    # point shifted by `period` to cover the interval that wraps around.
    target = source[0] + np.mod(target - source[0], period)
    source = np.append(source, source[0] + period)
  upper = np.searchsorted(source, target, side='right')
  upper = np.clip(upper, 1, source.size - 1)
  lower = upper - 1
  upper_weight = (target - source[lower]) / (source[upper] - source[lower])
  indices = np.stack([lower, upper % size], axis=-1)
  weights = np.stack([1 - upper_weight, upper_weight], axis=-1)
  return indices, weights


@functools.cache
def _cached_bilinear_neighbors(
    target_grid: coordinates.LonLatGrid, source_grid: coordinates.LonLatGrid
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns neighbor indices and weights along longitude and latitude.

  Like `_cached_weights`, arrays are computed once per grid pair, which lets JAX
  keep a single copy of them in programs that regrid multiple fields.

  Args:
    target_grid: Grid to interpolate to.
    source_grid: Grid of the interpolated inputs.

  Returns:
    Longitude indices and weights with shape `(target_lon, 2)`, followed by
    latitude indices and weights with shape `(target_lat, 2)`.
  """
  degrees = lambda grid, dim: np.asarray(grid.fields[dim].data, np.float64)
  lon_indices, lon_weights = _linear_interp_neighbors(
      degrees(target_grid, 'longitude'),
      degrees(source_grid, 'longitude'),
      period=360.0,
  )
  lat_indices, lat_weights = _linear_interp_neighbors(
      degrees(target_grid, 'latitude'), degrees(source_grid, 'latitude')
  )
  with jax.ensure_compile_time_eval():
    return (
        jnp.asarray(lon_indices, jnp.int32),
        jnp.asarray(lon_weights, jnp.float32),
        jnp.asarray(lat_indices, jnp.int32),
        jnp.asarray(lat_weights, jnp.float32),
    )


@dataclasses.dataclass(frozen=True)
class ConservativeRegridder(LonLatRegridder):
  """Regrids fields between two LonLatGrids using conservative interpolation."""

  def lat_weights(self, source_grid: coordinates.LonLatGrid) -> jax.Array:
    """Conservative regridding weights for the latitude dimension."""
    s_lat = jnp.deg2rad(source_grid.fields['latitude'].data)
    t_lat = jnp.deg2rad(self.target_grid.fields['latitude'].data)
    return horizontal_interpolation.conservative_latitude_weights(s_lat, t_lat)

  def lon_weights(self, source_grid: coordinates.LonLatGrid) -> jax.Array:
    """Conservative regridding weights for the longitude dimension."""
    s_lon = jnp.deg2rad(source_grid.fields['longitude'].data)
    t_lon = jnp.deg2rad(self.target_grid.fields['longitude'].data)
    return horizontal_interpolation.conservative_longitude_weights(s_lon, t_lon)


@dataclasses.dataclass(frozen=True)
class BilinearRegridder(LonLatRegridder):
  """Regrids fields between two LonLatGrids using bilinear interpolation.

  Interpolates periodically in longitude (period 360 degrees) and linearly in
  latitude, clamping values at the northernmost and southernmost source
  latitudes for target latitudes outside the source range (e.g. at the poles
  when regridding from a Gaussian grid to an equiangular grid with poles).

  Attributes:
    method: How interpolation weights are applied. ``'matmul'`` uses dense
      matrix products, which are efficient on TPUs. ``'gather'`` combines the
      two neighboring source values of each target point, which avoids the
      quadratic cost of dense products on CPUs and GPUs. ``'auto'`` selects
      ``'matmul'`` on TPUs and ``'gather'`` on other platforms.
  """

  method: Literal['auto', 'matmul', 'gather'] = 'auto'

  def __post_init__(self):
    if self.method not in ('auto', 'matmul', 'gather'):
      raise ValueError(
          f"Unsupported {self.method=}, expected 'auto', 'matmul' or 'gather'."
      )

  def lat_weights(self, source_grid: coordinates.LonLatGrid) -> np.ndarray:
    """Bilinear regridding weights for the latitude dimension."""
    return _linear_interp_weights(
        self.target_grid.fields['latitude'].data,
        source_grid.fields['latitude'].data,
    )

  def lon_weights(self, source_grid: coordinates.LonLatGrid) -> np.ndarray:
    """Bilinear regridding weights for the longitude dimension."""
    return _linear_interp_weights(
        self.target_grid.fields['longitude'].data,
        source_grid.fields['longitude'].data,
        period=360.0,
    )

  def _weighted_mean(
      self, data: jax.Array, source_grid: coordinates.LonLatGrid
  ) -> jax.Array:
    """Applies bilinear weights using the implementation set by `method`."""
    matmul = functools.partial(super()._weighted_mean, source_grid=source_grid)
    gather = functools.partial(self._gather_mean, source_grid=source_grid)
    if self.method == 'matmul':
      return matmul(data)
    if self.method == 'gather':
      return gather(data)
    # The platform is resolved when the computation is lowered, so that only
    # the implementation for the platform it runs on is compiled.
    return jax.lax.platform_dependent(data, tpu=matmul, default=gather)

  def _gather_mean(
      self, data: jax.Array, source_grid: coordinates.LonLatGrid
  ) -> jax.Array:
    """Applies bilinear weights by gathering neighbors of target points."""
    lon_indices, lon_weights, lat_indices, lat_weights = (
        _cached_bilinear_neighbors(self.target_grid, source_grid)
    )
    data = jnp.sum(lon_weights[:, :, jnp.newaxis] * data[lon_indices], axis=1)
    return jnp.sum(lat_weights * data[:, lat_indices], axis=-1)


def _jnp_linear_interp_neighbors(
    target: jax.Array,
    source: jax.Array,
    period: float | None = None,
    uniform_step: float | None = None,
) -> tuple[jax.Array, jax.Array]:
  """JAX-traceable 1D linear interpolation neighbor indices and weights.

  Args:
    target: Query coordinates of arbitrary shape (caller should sanitize NaNs
      when outputs are used in scatter-adds).
    source: 1D ascending array of source grid coordinates.
    period: Optional domain period (e.g. 360.0 for longitude). If None, queries
      outside `[source[0], source[-1]]` are clamped to boundary values.
    uniform_step: Optional uniform spacing between consecutive `source` nodes.
      When provided, computes neighbors in `O(1)` arithmetic without binary
      search.

  Returns:
    Tuple `(indices, weights)` each of shape `(*target.shape, 2)`.
  """
  n_src = len(source)
  if n_src == 1:
    indices = jnp.zeros((*target.shape, 2), dtype=jnp.int32)
    weights = jnp.broadcast_to(
        jnp.array([1.0, 0.0], dtype=target.dtype), (*target.shape, 2)
    )
    return indices, weights

  if uniform_step is not None:
    origin = source[0]
    if period is not None:
      u = jnp.mod(target - origin, period) / uniform_step
      lower = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_src - 1)
      upper = jnp.mod(lower + 1, n_src)
      upper_weight = jnp.clip(u - lower.astype(u.dtype), 0.0, 1.0)
    else:
      u = jnp.clip((target - origin) / uniform_step, 0.0, float(n_src - 1))
      lower = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_src - 2)
      upper = lower + 1
      upper_weight = jnp.clip(u - lower.astype(u.dtype), 0.0, 1.0)
    indices = jnp.stack([lower, upper], axis=-1)
    weights = jnp.stack([1.0 - upper_weight, upper_weight], axis=-1)
    return indices, weights

  source = jnp.asarray(source)

  if period is not None:
    target = jnp.mod(target - source[0], period) + source[0]
    upper = jnp.searchsorted(source, target, side='right')
    lower = jnp.mod(upper - 1, source.size)
    lower_val = source[lower]
    safe_upper = jnp.minimum(upper, source.size - 1)
    upper_val = jnp.where(
        upper < source.size, source[safe_upper], source[0] + period
    )
    upper = jnp.mod(upper, source.size)
  else:
    upper = jnp.clip(
        jnp.searchsorted(source, target, side='right'), 1, source.size - 1
    )
    lower = upper - 1
    lower_val = source[lower]
    upper_val = source[upper]
    target = jnp.clip(target, lower_val, upper_val)

  upper_weight = (target - lower_val) / (upper_val - lower_val)
  indices = jnp.stack([lower, upper], axis=-1)
  weights = jnp.stack([1.0 - upper_weight, upper_weight], axis=-1)
  return indices, weights


def _splat_2d(
    f_flat: jax.Array,
    valid: jax.Array,
    w_flat: jax.Array,
    idx0: jax.Array,
    w0: jax.Array,
    idx1: jax.Array,
    w1: jax.Array,
    target_shape: tuple[int, int],
    method: Literal['bilinear', 'nearest'],
) -> jax.Array:
  """Accumulates 1D flattened footprints onto a 2D target grid."""
  safe_f = jnp.where(valid, f_flat, 0.0)
  safe_w_mult = jnp.where(valid, w_flat, 0.0)
  val_sum = jnp.zeros(target_shape, dtype=f_flat.dtype)
  w_sum = jnp.zeros(target_shape, dtype=f_flat.dtype)

  if method == 'nearest':
    i0 = jnp.where(w0[:, 1] >= 0.5, idx0[:, 1], idx0[:, 0])
    i1 = jnp.where(w1[:, 1] >= 0.5, idx1[:, 1], idx1[:, 0])
    val_sum = val_sum.at[i0, i1].add(safe_f * safe_w_mult)
    w_sum = w_sum.at[i0, i1].add(safe_w_mult)
  else:
    raw_w2d = w0[:, :, None] * w1[:, None, :]
    safe_w2d = safe_w_mult[:, None, None] * raw_w2d
    m = f_flat.size
    i0_2d = jnp.broadcast_to(idx0[:, :, None], (m, 2, 2))
    i1_2d = jnp.broadcast_to(idx1[:, None, :], (m, 2, 2))
    val_sum = val_sum.at[i0_2d, i1_2d].add(safe_f[:, None, None] * safe_w2d)
    w_sum = w_sum.at[i0_2d, i1_2d].add(safe_w2d)

  has_obs = w_sum > 0.0
  safe_w_sum = jnp.where(has_obs, w_sum, 1.0)
  return jnp.where(has_obs, val_sum / safe_w_sum, jnp.nan)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LonLatToSwathInterpolator:
  """Interpolates fields from a LonLatGrid onto PatchCoordinate query fields.

  Supports both `'bilinear'` and `'nearest'` interpolation with periodic
  longitude wrapping (360 degrees) and pole clamping in latitude. Footprints
  with `NaN` query coordinates (`longitude` or `latitude`) produce `NaN`
  outputs.

  Attributes:
    method: Interpolation method, either `'bilinear'` or `'nearest'`.
    skipna: If True and `method='bilinear'`, renormalizes bilinear weights over
      non-NaN corner neighbors when a query footprint straddles NaN grid cells.
    skip_missing: If True, returns fields without a `LonLatGrid` unchanged
      instead of raising an error.
    lon_query_key: Key used to extract longitude query field when called on a
      dict without explicit `lon_query`.
    lat_query_key: Key used to extract latitude query field when called on a
      dict without explicit `lat_query`.
    include_query_fields_in_output: Whether to keep `lon_query_key` and
      `lat_query_key` in the returned dict when called on a dict.
  """

  method: Literal['bilinear', 'nearest'] = 'bilinear'
  skipna: bool = False
  skip_missing: bool = False
  lon_query_key: str = 'longitude'
  lat_query_key: str = 'latitude'
  include_query_fields_in_output: bool = False

  def __post_init__(self):
    if self.method not in ('bilinear', 'nearest'):
      raise ValueError(
          f"method must be 'bilinear' or 'nearest', got {self.method=!r}."
      )

  def _validate_queries(
      self, lon_query: cx.Field, lat_query: cx.Field
  ) -> None:
    patch = cx.coords.extract(
        lon_query.coordinate, coordinates.PatchCoordinate
    )
    lat_patch = cx.coords.extract(
        lat_query.coordinate, coordinates.PatchCoordinate
    )
    if patch != lat_patch or lon_query.dims != lat_query.dims:
      raise ValueError(
          'lon_query and lat_query must have matching PatchCoordinate and dims,'
          f' got {lon_query.coordinate=} vs {lat_query.coordinate=}.'
      )

  def _precompute_neighbors(
      self,
      source_grid: coordinates.LonLatGrid,
      lon_query: cx.Field,
      lat_query: cx.Field,
  ) -> tuple[cx.Field, ...]:
    """Precomputes neighbor indices and weights once per `source_grid`."""
    if source_grid.lon_lat_padding != (0, 0):
      raise ValueError(f'Padded source_grid is not supported: {source_grid=}.')
    n_lon = source_grid.shape[0]
    lon_src = source_grid.fields['longitude'].data
    lat_src = source_grid.fields['latitude'].data

    def _point_neighbors(lon_q: jax.Array, lat_q: jax.Array):
      q_valid = jnp.isfinite(lon_q) & jnp.isfinite(lat_q)
      safe_lon = jnp.where(q_valid, lon_q, 0.0)
      safe_lat = jnp.where(q_valid, lat_q, 0.0)
      lon_idx, lon_w = _jnp_linear_interp_neighbors(
          safe_lon, lon_src, period=360.0, uniform_step=360.0 / n_lon
      )
      lat_idx, lat_w = _jnp_linear_interp_neighbors(
          safe_lat, lat_src, period=None
      )
      if self.method == 'nearest':
        i_lon = jnp.where(lon_w[1] >= 0.5, lon_idx[1], lon_idx[0])
        i_lat = jnp.where(lat_w[1] >= 0.5, lat_idx[1], lat_idx[0])
        return q_valid, i_lon, i_lat
      return (
          q_valid,
          lon_idx[0],
          lon_idx[1],
          lat_idx[0],
          lat_idx[1],
          lon_w[0] * lat_w[0],
          lon_w[0] * lat_w[1],
          lon_w[1] * lat_w[0],
          lon_w[1] * lat_w[1],
      )

    return cx.cmap(_point_neighbors)(lon_query, lat_query)

  def _interpolate_field_with_cache(
      self,
      f: cx.Field,
      lon_query: cx.Field,
      lat_query: cx.Field,
      neighbors_cache: dict[coordinates.LonLatGrid, tuple[cx.Field, ...]],
  ) -> cx.Field:
    """Interpolates `f` onto `(lon_query, lat_query)` using cached neighbors."""
    try:
      source_grid = cx.coords.extract(f.coordinate, coordinates.LonLatGrid)
    except ValueError:
      if self.skip_missing:
        return f
      raise
    self._validate_queries(lon_query, lat_query)
    if source_grid not in neighbors_cache:
      neighbors_cache[source_grid] = self._precompute_neighbors(
          source_grid, lon_query, lat_query
      )
    neighbors = neighbors_cache[source_grid]

    query_axes = [
        ax for ax in lon_query.coordinate.axes if not cx.contains_dims(f, ax)
    ]
    out_coord = cx.coords.replace_axes(
        f.coordinate, source_grid, cx.coords.compose(*query_axes)
    )

    if self.method == 'nearest':
      q_valid, i_lon, i_lat = neighbors

      def _gather_nearest(grid_2d: jax.Array, q_ok, il, jl) -> jax.Array:
        val = jnp.asarray(grid_2d)[il, jl]
        return jnp.where(q_ok, val, jnp.nan)

      return cx.cmap(_gather_nearest, out_axes=out_coord.dims)(
          f.untag(source_grid), q_valid, i_lon, i_lat
      )

    q_valid, i0, i1, j0, j1, w00, w01, w10, w11 = neighbors

    def _gather_bilinear(
        grid_2d: jax.Array, q_ok, il0, il1, jl0, jl1, a00, a01, a10, a11
    ) -> jax.Array:
      g = jnp.asarray(grid_2d)
      c00 = g[il0, jl0]
      c01 = g[il0, jl1]
      c10 = g[il1, jl0]
      c11 = g[il1, jl1]
      if self.skipna:
        corners = jnp.stack([c00, c01, c10, c11])
        weights = jnp.stack([a00, a01, a10, a11])
        valid_c = jnp.isfinite(corners)
        safe_c = jnp.where(valid_c, corners, 0.0)
        safe_w = jnp.where(valid_c, weights, 0.0)
        w_sum = jnp.sum(safe_w)
        has_valid = q_ok & (w_sum > 0.0)
        safe_w_sum = jnp.where(has_valid, w_sum, 1.0)
        return jnp.where(
            has_valid, jnp.sum(safe_c * safe_w) / safe_w_sum, jnp.nan
        )
      val = a00 * c00 + a01 * c01 + a10 * c10 + a11 * c11
      return jnp.where(q_ok, val, jnp.nan)

    return cx.cmap(_gather_bilinear, out_axes=out_coord.dims)(
        f.untag(source_grid), q_valid, i0, i1, j0, j1, w00, w01, w10, w11
    )

  def interpolate_field(
      self,
      f: cx.Field,
      lon_query: cx.Field,
      lat_query: cx.Field,
  ) -> cx.Field:
    """Interpolates Field `f` on LonLatGrid onto `(lon_query, lat_query)`."""
    return self._interpolate_field_with_cache(f, lon_query, lat_query, {})

  def __call__(
      self,
      inputs: cx.Field | dict[str, cx.Field],
      lon_query: cx.Field | None = None,
      lat_query: cx.Field | None = None,
  ) -> cx.Field | dict[str, cx.Field]:
    if isinstance(inputs, cx.Field):
      if lon_query is None or lat_query is None:
        raise ValueError(
            'lon_query and lat_query must be provided when inputs is a'
            ' cx.Field.'
        )
      return self.interpolate_field(inputs, lon_query, lat_query)

    inputs_copy = inputs.copy()
    lon_q = inputs_copy.pop(self.lon_query_key, None)
    lat_q = inputs_copy.pop(self.lat_query_key, None)
    if lon_query is None:
      lon_query = lon_q
    if lat_query is None:
      lat_query = lat_q
    if lon_query is None or lat_query is None:
      raise ValueError(
          f'Could not find {self.lon_query_key!r} and {self.lat_query_key!r} in'
          ' inputs and lon_query/lat_query were not provided explicitly.'
      )

    neighbors_cache: dict[coordinates.LonLatGrid, tuple[cx.Field, ...]] = {}
    outputs = {
        k: self._interpolate_field_with_cache(
            v, lon_query, lat_query, neighbors_cache
        )
        for k, v in inputs_copy.items()
    }
    if self.include_query_fields_in_output:
      outputs[self.lon_query_key] = lon_query
      outputs[self.lat_query_key] = lat_query
    return outputs


LonLatToPatchInterpolator = LonLatToSwathInterpolator


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SwathToLonLatInterpolator:
  """Projects/splats PatchCoordinate fields onto a target LonLatGrid.

  Accumulates patch footprint values onto `target_grid` using inverse-distance
  bilinear weights (`method='bilinear'`) or cell-bin averaging
  (`method='nearest'`), optionally compositing across `reduction_dims` and
  weighting by per-footprint blending weights (`weights`). Grid cells that
  receive no valid footprints are filled with `jnp.nan`.

  Attributes:
    target_grid: Target `coordinates.LonLatGrid` to project onto.
    method: Projection method, either `'bilinear'` or `'nearest'`.
    reduction_dims: Optional tuple of dimension names to composite together with
      the 2D patch dimensions onto a single `target_grid`.
    skip_missing: If True, returns fields without a `PatchCoordinate` unchanged
      instead of raising an error.
    lon_query_key: Key used to extract longitude field when called on a dict.
    lat_query_key: Key used to extract latitude field when called on a dict.
    include_query_fields_in_output: Whether to keep `lon_query_key` and
      `lat_query_key` in the returned dict when called on a dict.
  """

  target_grid: coordinates.LonLatGrid
  method: Literal['bilinear', 'nearest'] = 'bilinear'
  reduction_dims: tuple[str, ...] = ()
  skip_missing: bool = False
  lon_query_key: str = 'longitude'
  lat_query_key: str = 'latitude'
  include_query_fields_in_output: bool = False

  def __post_init__(self):
    object.__setattr__(self, 'reduction_dims', tuple(self.reduction_dims))
    if self.target_grid.lon_lat_padding != (0, 0):
      raise ValueError(
          f'Padded target_grid is not supported: {self.target_grid=}.'
      )
    if self.method not in ('bilinear', 'nearest'):
      raise ValueError(
          f"method must be 'bilinear' or 'nearest', got {self.method=!r}."
      )

  def interpolate_field(
      self,
      f: cx.Field,
      lon_query: cx.Field,
      lat_query: cx.Field,
      *,
      weights: cx.Field | None = None,
  ) -> cx.Field:
    """Projects Field `f` on a PatchCoordinate onto `self.target_grid`."""
    try:
      patch = cx.coords.extract(f.coordinate, coordinates.PatchCoordinate)
    except ValueError:
      if self.skip_missing:
        return f
      raise
    lon_patch = cx.coords.extract(
        lon_query.coordinate, coordinates.PatchCoordinate
    )
    lat_patch = cx.coords.extract(
        lat_query.coordinate, coordinates.PatchCoordinate
    )
    if patch != lon_patch or patch != lat_patch:
      raise ValueError(
          'f, lon_query, and lat_query must share the same PatchCoordinate, got'
          f' {patch=}, {lon_patch=}, {lat_patch=}.'
      )

    dims_to_reduce = (*self.reduction_dims, *patch.dims)
    f_u = f.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)
    lon_u = lon_query.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)
    lat_u = lat_query.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)
    if weights is None:
      w_field = cx.field(jnp.ones(patch.shape, dtype=jnp.float32), patch)
    else:
      w_field = weights
    w_bcast = w_field.broadcast_like(lon_query.coordinate)
    w_u = w_bcast.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)

    lon_src = self.target_grid.fields['longitude'].data
    lat_src = self.target_grid.fields['latitude'].data
    n_lon, n_lat = self.target_grid.shape

    def _splat_to_grid(
        f_vals: jax.Array,
        lon_vals: jax.Array,
        lat_vals: jax.Array,
        w_vals: jax.Array,
    ) -> jax.Array:
      f_flat = f_vals.ravel()
      lon_flat = lon_vals.ravel()
      lat_flat = lat_vals.ravel()
      w_flat = w_vals.ravel()
      coord_valid = jnp.isfinite(lon_flat) & jnp.isfinite(lat_flat)
      safe_lon = jnp.where(coord_valid, lon_flat, 0.0)
      safe_lat = jnp.where(coord_valid, lat_flat, 0.0)
      lon_idx, lon_w = _jnp_linear_interp_neighbors(
          safe_lon, lon_src, period=360.0, uniform_step=360.0 / n_lon
      )
      lat_idx, lat_w = _jnp_linear_interp_neighbors(
          safe_lat, lat_src, period=None
      )
      valid = (
          jnp.isfinite(f_flat)
          & coord_valid
          & jnp.isfinite(w_flat)
          & (w_flat > 0.0)
      )
      return _splat_2d(
          f_flat,
          valid,
          w_flat,
          lon_idx,
          lon_w,
          lat_idx,
          lat_w,
          (n_lon, n_lat),
          self.method,
      )

    two_positional = cx.coords.compose(
        cx.DummyAxis(None, 1), cx.DummyAxis(None, 1)
    )
    out_template = cx.coords.replace_axes(f.coordinate, patch, two_positional)
    for d in self.reduction_dims:
      out_template = cx.coords.replace_axes(
          out_template, cx.get_coordinate_part(f, d), cx.Scalar()
      )
    return cx.cmap(_splat_to_grid, out_axes=out_template.dims)(
        f_u, lon_u, lat_u, w_u
    ).tag(self.target_grid)

  def __call__(
      self,
      inputs: cx.Field | dict[str, cx.Field],
      lon_query: cx.Field | None = None,
      lat_query: cx.Field | None = None,
      *,
      weights: cx.Field | None = None,
  ) -> cx.Field | dict[str, cx.Field]:
    if isinstance(inputs, cx.Field):
      if lon_query is None or lat_query is None:
        raise ValueError(
            'lon_query and lat_query must be provided when inputs is a'
            ' cx.Field.'
        )
      return self.interpolate_field(
          inputs, lon_query, lat_query, weights=weights
      )

    inputs_copy = inputs.copy()
    lon_q = inputs_copy.pop(self.lon_query_key, None)
    lat_q = inputs_copy.pop(self.lat_query_key, None)
    if lon_query is None:
      lon_query = lon_q
    if lat_query is None:
      lat_query = lat_q
    if lon_query is None or lat_query is None:
      raise ValueError(
          f'Could not find {self.lon_query_key!r} and {self.lat_query_key!r} in'
          ' inputs and lon_query/lat_query were not provided explicitly.'
      )

    outputs = {
        k: self.interpolate_field(v, lon_query, lat_query, weights=weights)
        for k, v in inputs_copy.items()
    }
    if self.include_query_fields_in_output:
      outputs[self.lon_query_key] = lon_query
      outputs[self.lat_query_key] = lat_query
    return outputs


PatchToLonLatInterpolator = SwathToLonLatInterpolator


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class PatchToLocalPatchInterpolator:
  """Co-projects fields from any source PatchCoordinate onto a `LocalGridPatch`.

  Maps source footprint coordinates `(lon_query, lat_query)` into the target
  `LocalGridPatch`'s rotated-sphere frame `(patch_x, patch_y)` centered at
  `(center_lon, center_lat)`, masks out footprints outside the target patch
  bounds, and accumulates in-bounds footprints via bilinear splatting or nearest
  cell-bin averaging.

  Attributes:
    target_patch: Target `coordinates.LocalGridPatch` to project onto.
    method: Projection method, either `'bilinear'` or `'nearest'`.
    reduction_dims: Optional tuple of source dimension names (e.g.
      `('satellite', 'patch_index')`) to composite together onto `target_patch`.
    azimuth_deg: Tangent-plane rotation angle in degrees of `target_patch`.
    skip_missing: If True, returns fields without a `PatchCoordinate` unchanged.
    return_coverage_mask: If True and called on a dict, adds a `'coverage_mask'`
      field (`1.0` where observed, `0.0` where unobserved) to the output dict.
    lon_query_key: Key used to extract source longitude field from a dict.
    lat_query_key: Key used to extract source latitude field from a dict.
  """

  target_patch: coordinates.LocalGridPatch
  method: Literal['bilinear', 'nearest'] = 'bilinear'
  reduction_dims: tuple[str, ...] = ()
  azimuth_deg: float = 0.0
  skip_missing: bool = False
  return_coverage_mask: bool = False
  lon_query_key: str = 'longitude'
  lat_query_key: str = 'latitude'

  def __post_init__(self):
    object.__setattr__(self, 'reduction_dims', tuple(self.reduction_dims))
    if self.method not in ('bilinear', 'nearest'):
      raise ValueError(
          f"method must be 'bilinear' or 'nearest', got {self.method=!r}."
      )

  def _interpolate_field_impl(
      self,
      f: cx.Field,
      lon_query: cx.Field,
      lat_query: cx.Field,
      center_lon: cx.Field,
      center_lat: cx.Field,
      *,
      weights: cx.Field | None = None,
      local_coords: tuple[cx.Field, cx.Field] | None = None,
  ) -> cx.Field:
    """Projects a single field onto `target_patch` using cached local coords."""
    try:
      src_patch = cx.coords.extract(f.coordinate, coordinates.PatchCoordinate)
    except ValueError:
      if self.skip_missing:
        return f
      raise
    lon_patch = cx.coords.extract(
        lon_query.coordinate, coordinates.PatchCoordinate
    )
    lat_patch = cx.coords.extract(
        lat_query.coordinate, coordinates.PatchCoordinate
    )
    if src_patch != lon_patch or src_patch != lat_patch:
      raise ValueError(
          'f, lon_query, and lat_query must share the same PatchCoordinate, got'
          f' {src_patch=}, {lon_patch=}, {lat_patch=}.'
      )

    if local_coords is None:
      xi_deg, eta_deg = patch_geometry.lon_lat_to_local_patch(
          lon_query,
          lat_query,
          center_lon,
          center_lat,
          azimuth_deg=self.azimuth_deg,
      )
    else:
      xi_deg, eta_deg = local_coords

    dims_to_reduce = (*self.reduction_dims, *src_patch.dims)
    f_u = f.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)
    xi_u = xi_deg.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)
    eta_u = eta_deg.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)
    if weights is None:
      ones = jnp.ones(src_patch.shape, dtype=jnp.float32)
      w_field = cx.field(ones, src_patch)
    else:
      w_field = weights
    w_bcast = w_field.broadcast_like(lon_query.coordinate)
    w_u = w_bcast.order_as(..., *dims_to_reduce).untag(*dims_to_reduce)

    x_src = jnp.asarray(self.target_patch.fields['patch_x'].data)
    y_src = jnp.asarray(self.target_patch.fields['patch_y'].data)
    nx, ny = self.target_patch.shape
    tol_x = 0.5 * self.target_patch.dx_deg
    tol_y = 0.5 * self.target_patch.dy_deg

    def _splat_to_local(
        f_vals: jax.Array,
        xi_vals: jax.Array,
        eta_vals: jax.Array,
        w_vals: jax.Array,
    ) -> jax.Array:
      f_flat = f_vals.ravel()
      xi_flat = xi_vals.ravel()
      eta_flat = eta_vals.ravel()
      w_flat = w_vals.ravel()
      in_bounds_coord = (
          jnp.isfinite(xi_flat)
          & jnp.isfinite(eta_flat)
          & (xi_flat >= x_src[0] - tol_x)
          & (xi_flat <= x_src[-1] + tol_x)
          & (eta_flat >= y_src[0] - tol_y)
          & (eta_flat <= y_src[-1] + tol_y)
      )
      safe_xi = jnp.where(in_bounds_coord, xi_flat, 0.0)
      safe_eta = jnp.where(in_bounds_coord, eta_flat, 0.0)
      x_idx, x_w = _jnp_linear_interp_neighbors(
          safe_xi, x_src, period=None, uniform_step=self.target_patch.dx_deg
      )
      y_idx, y_w = _jnp_linear_interp_neighbors(
          safe_eta, y_src, period=None, uniform_step=self.target_patch.dy_deg
      )
      valid = (
          jnp.isfinite(f_flat)
          & in_bounds_coord
          & jnp.isfinite(w_flat)
          & (w_flat > 0.0)
      )
      return _splat_2d(
          f_flat,
          valid,
          w_flat,
          x_idx,
          x_w,
          y_idx,
          y_w,
          (nx, ny),
          self.method,
      )

    extra_center_axes = [
        c
        for c in cx.coords.canonicalize(center_lon.coordinate)
        if not cx.contains_dims(f, c) and c.dims[0] not in self.reduction_dims
    ]
    replacement = cx.coords.compose(
        *extra_center_axes, cx.DummyAxis(None, 1), cx.DummyAxis(None, 1)
    )
    out_template = cx.coords.replace_axes(f.coordinate, src_patch, replacement)
    for d in self.reduction_dims:
      if d in out_template.dims:
        out_template = cx.coords.replace_axes(
            out_template, cx.get_coordinate_part(f, d), cx.Scalar()
        )
    return cx.cmap(_splat_to_local, out_axes=out_template.dims)(
        f_u, xi_u, eta_u, w_u
    ).tag(self.target_patch)

  def interpolate_field(
      self,
      f: cx.Field,
      lon_query: cx.Field,
      lat_query: cx.Field,
      center_lon: cx.Field,
      center_lat: cx.Field,
      *,
      weights: cx.Field | None = None,
  ) -> cx.Field:
    """Projects `f` onto `self.target_patch` at `(center_lon, center_lat)`."""
    return self._interpolate_field_impl(
        f, lon_query, lat_query, center_lon, center_lat, weights=weights
    )

  def __call__(
      self,
      inputs: cx.Field | dict[str, cx.Field],
      center_lon: cx.Field,
      center_lat: cx.Field,
      lon_query: cx.Field | None = None,
      lat_query: cx.Field | None = None,
      *,
      weights: cx.Field | None = None,
  ) -> cx.Field | dict[str, cx.Field]:
    if isinstance(inputs, cx.Field):
      if lon_query is None or lat_query is None:
        raise ValueError(
            'lon_query and lat_query must be provided when inputs is a'
            ' cx.Field.'
        )
      return self.interpolate_field(
          inputs,
          lon_query,
          lat_query,
          center_lon,
          center_lat,
          weights=weights,
      )

    inputs_copy = inputs.copy()
    lon_q = inputs_copy.pop(self.lon_query_key, None)
    lat_q = inputs_copy.pop(self.lat_query_key, None)
    if lon_query is None:
      lon_query = lon_q
    if lat_query is None:
      lat_query = lat_q
    if lon_query is None or lat_query is None:
      raise ValueError(
          f'Could not find {self.lon_query_key!r} and {self.lat_query_key!r} in'
          ' inputs and lon_query/lat_query were not provided explicitly.'
      )

    local_coords = patch_geometry.lon_lat_to_local_patch(
        lon_query,
        lat_query,
        center_lon,
        center_lat,
        azimuth_deg=self.azimuth_deg,
    )
    outputs = {
        k: self._interpolate_field_impl(
            v,
            lon_query,
            lat_query,
            center_lon,
            center_lat,
            weights=weights,
            local_coords=local_coords,
        )
        for k, v in inputs_copy.items()
    }
    if self.return_coverage_mask:
      unit_footprint = cx.cmap(
          lambda lon, lat: jnp.where(
              jnp.isfinite(lon) & jnp.isfinite(lat), jnp.float32(1.0), jnp.nan
          )
      )(lon_query, lat_query)
      cov_splat = self._interpolate_field_impl(
          unit_footprint,
          lon_query,
          lat_query,
          center_lon,
          center_lat,
          weights=weights,
          local_coords=local_coords,
      )
      outputs['coverage_mask'] = cx.cmap(
          lambda a: jnp.isfinite(a).astype(jnp.float32)
      )(cov_splat)
    return outputs


SwathToLocalPatchInterpolator = PatchToLocalPatchInterpolator
