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
