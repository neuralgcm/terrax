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
from typing import Protocol, overload

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

  def _regrid_2d(
      self, array: jax.Array, source_grid: coordinates.LonLatGrid
  ) -> jax.Array:
    """Applies horizontal regridding to a 2D (lon, lat) array."""
    lon_w, lat_w = _cached_weights(self, source_grid)

    def _mean(data: jax.Array) -> jax.Array:
      return jnp.einsum(
          'ab,cd,bd->ac',
          lon_w,
          lat_w,
          data,
          precision='float32',
      )

    not_nulls = jnp.logical_not(jnp.isnan(array))
    mean = _mean(jnp.where(not_nulls, array, 0))
    not_null_fraction = _mean(not_nulls)
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

  Weights are applied as dense matrix products, which is efficient on TPUs.
  """

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
