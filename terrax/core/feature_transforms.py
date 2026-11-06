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

"""Transforms that specialize in generating features."""

from __future__ import annotations

import functools
from typing import Any, Literal, Sequence

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import jax_solar
import numpy as np
from terrax.core import coordinates
from terrax.core import diagnostics
from terrax.core import dynamic_io
from terrax.core import orographies
from terrax.core import parallelism
from terrax.core import random_processes
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units
from terrax.core import xarray_utils
import xarray

TransformParams = transforms.TransformParams


@nnx.dataclass
class RadiationFeatures(transforms.PytreeTransformABC):
  """Returns incident radiation flux Field."""

  grid: coordinates.LonLatGrid

  @property
  def lon(self) -> typing.Array:
    return self.grid.fields['longitude'].broadcast_like(self.grid).data

  @property
  def lat(self) -> typing.Array:
    return self.grid.fields['latitude'].broadcast_like(self.grid).data

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    features = {}
    features['radiation'] = cx.field(
        jax_solar.normalized_radiation_flux(
            time=inputs['time'].data, longitude=self.lon, latitude=self.lat  # pyrefly: ignore[bad-argument-type]
        ),
        self.grid,
    )
    return features


@nnx.dataclass
class LatitudeFeatures(transforms.PytreeTransformABC):
  """Returns cos and sin of latitude features."""

  grid: coordinates.LonLatGrid

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    latitudes = jnp.deg2rad(
        self.grid.fields['latitude'].broadcast_like(self.grid).data
    )
    features = {
        'cos_latitude': cx.field(jnp.cos(latitudes), self.grid),
        'sin_latitude': cx.field(jnp.sin(latitudes), self.grid),
    }
    return features


@nnx.dataclass
class DiagnosticValueFeatures(transforms.TransformABC):
  """Returns diagnostic values to be used as features."""

  diagnostic_module: diagnostics.DiagnosticModule

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused
    return self.diagnostic_module.diagnostic_values()


@nnx.dataclass
class ObservationValueFeatures(transforms.TransformABC):
  """Returns values observed by `operator` for a fixed `query`.

  Adapts an observation operator to the transform interface, which makes it
  possible to use predictions of an existing operator (e.g. as a reference for
  another operator) anywhere a transform is expected.

  Attributes:
    operator: Observation operator used to compute the returned values.
    query: Query passed to ``operator.observe``.
  """

  operator: typing.ObservationOperator
  query: dict[str, cx.Coordinate]

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.operator.observe(inputs, self.query)


@nnx.dataclass
class RandomnessFeatures(transforms.TransformABC):
  """Returns values of a random process evaluated on a coordinate.

  The evaluation coordinate is either the fixed `coord` or, if `like` is set,
  the coordinate of ``inputs[like]`` at call time, so that a single transform
  serves inputs of varying structure, e.g. queries at different sets of points
  (the random process must then support evaluation on arbitrary coordinates,
  e.g. `random_processes.NormalUncorrelated`). In both cases `sample_axis`, if
  provided, is appended to the evaluation coordinate.

  Attributes:
    random_process: Random process evaluated via `state_values(coord)`.
    coord: Fixed coordinate on which the random process is evaluated. Exactly
      one of `coord` and `like` must be set.
    feature_name: Key under which the random values are returned.
    like: Key of the input field whose coordinate is used for evaluation.
    sample_axis: Optional axis of independent samples per entry of the
      evaluation coordinate.
  """

  random_process: random_processes.RandomProcessModule
  coord: cx.Coordinate | None = None
  feature_name: str = 'randomness'
  like: str | None = None
  sample_axis: cx.Coordinate | None = None

  def __post_init__(self):
    if (self.coord is None) == (self.like is None):
      raise ValueError(
          'Exactly one of `coord` and `like` must be set, got'
          f' {self.coord=} and {self.like=}.'
      )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    if self.like is not None:
      coord = inputs[self.like].coordinate
    else:
      assert self.coord is not None  # validated in __post_init__.
      coord = self.coord
    if self.sample_axis is not None:
      coord = cx.coords.compose(coord, self.sample_axis)
    return {self.feature_name: self.random_process.state_values(coord)}


@nnx.dataclass
class DynamicInputFeatures(transforms.TransformABC):
  """Returns subset of dynamic input values.

  Attributes:
    keys: Names of the dynamic input variables to return.
    dynamic_input_module: Module that supplies time-indexed data.
    time_offsets: Offset(s) relative to the input time at which values are
      retrieved. A sequence of offsets retrieves values once per offset and
      stacks them along `window_axis`, which can then be reduced or indexed into
      with standard transforms.
    window_axis: Axis along which values retrieved at multiple `time_offsets`
      are stacked. Required if `time_offsets` is a sequence, in which case its
      size must match the number of offsets.
  """

  keys: Sequence[str]
  dynamic_input_module: dynamic_io.DynamicInputSlice
  time_offsets: np.timedelta64 | Sequence[np.timedelta64] | None = None
  window_axis: cx.Coordinate | None = None

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    time_offsets = self.time_offsets if self.time_offsets is not None else (
        np.timedelta64(0, 's')
    )
    if isinstance(time_offsets, np.timedelta64):
      # pyrefly: ignore[bad-argument-type]
      time = cx.field(jdt.to_timedelta(time_offsets)) + inputs['time']
      data_features = self.dynamic_input_module(time)
      return {k: data_features[k] for k in self.keys}
    if self.window_axis is None:
      raise ValueError(
          f'`window_axis` is required to stack values at {time_offsets=}.'
      )
    if self.window_axis.shape != (len(time_offsets),):
      raise ValueError(
          f'{self.window_axis=} must be one dimensional with size'
          f' {len(time_offsets)} to match {time_offsets=}.'
      )
    features_per_offset = []
    for offset in time_offsets:
      # pyrefly: ignore[bad-argument-type]
      offset_time = inputs['time'] + cx.field(jdt.to_timedelta(offset))
      features_per_offset.append(self.dynamic_input_module(offset_time))
    stack = cx.cmap(jnp.stack)
    return {
        k: stack([f[k] for f in features_per_offset]).tag(self.window_axis)
        for k in self.keys
    }


@nnx.dataclass
class OrographyFeatures(transforms.TransformABC):
  """Returns elevation values in real space representing orography."""

  orography_module: orographies.ModalOrography | orographies.Orography

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {'orography': self.orography_module.nodal_orography}

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      **kwargs,
  ):
    """Updates the underlying orography module from an xarray dataset."""
    self.orography_module.update_from_xarray(dataset, **kwargs)


@nnx.dataclass
class OrographyWithGradsFeatures(transforms.PytreeTransformABC):
  """Returns orography features and their gradients."""

  orography_module: orographies.ModalOrography
  compute_gradients_transform: transforms.ToModalWithDerivatives
  include_raw_orography: bool = True

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    ylm_map = self.orography_module.ylm_map
    grid = ylm_map.lon_lat_grid
    modal_features = {'orography': self.orography_module.modal_orography}
    modal_features = {
        typing.KeyWithCosLatFactor(k, 0): v for k, v in modal_features.items()
    }
    modal_gradient_features = self.compute_gradients_transform(modal_features)
    sh_grid = ylm_map.dinosaur_grid
    sec_lat = 1 / sh_grid.cos_lat
    sec2_lat = sh_grid.sec2_lat
    lat_axis = grid.axes[1]
    sec_lat_scales = {
        0: 1,
        1: cx.field(sec_lat, lat_axis),
        2: cx.field(sec2_lat, lat_axis),
    }
    features = {}
    if self.include_raw_orography:
      all_modal_features = modal_gradient_features | modal_features
    else:
      all_modal_features = modal_gradient_features
    for k, v in all_modal_features.items():
      sec_lat_scale = sec_lat_scales[k.factor_order]
      features[k.name] = ylm_map.to_nodal(v) * sec_lat_scale
    return features

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      **kwargs,
  ):
    """Updates the underlying orography module from an xarray dataset."""
    self.orography_module.update_from_xarray(dataset, **kwargs)


class ParamFeatures(transforms.PytreeTransformABC):
  """Returns, potentially learnable, features on fixed coordinates."""

  def __init__(
      self,
      coords: dict[str, cx.Coordinate],
      *,
      param_type: TransformParams | nnx.Param = TransformParams,
      initializer: nnx.Initializer = nnx.initializers.truncated_normal(),
      rngs: nnx.Rngs,
  ):
    self.coords = coords
    self.features = nnx.data({
        k: param_type(initializer(rngs.params(), c.shape))
        for k, c in coords.items()
    })

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    del inputs  # unused.
    return {
        k: cx.field(v[...], self.coords[k]) for k, v in self.features.items()
    }

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      **kwargs,
  ):
    """Updates `self.features` with data from dataset."""
    sim_units = kwargs['sim_units']
    for key, feature in self.features.items():
      if key in dataset:
        da = dataset[key]
        data_units = units.parse_units(da.attrs['units'])
        da = da.copy(data=sim_units.nondimensionalize(da.values * data_units))
        candidate = xarray_utils.from_xarray(da)
        value = candidate.order_as(self.coords[key]).data
        feature.set_value(parallelism.shard_like(value, feature.get_value()))


class ClimatologyFeatures(transforms.PytreeTransformABC):
  """Returns climatological feature fields dynamically sliced by time."""

  def __init__(
      self,
      coords: dict[str, cx.Coordinate],
      *,
      param_type: TransformParams | nnx.Param = TransformParams,
      initializer: nnx.Initializer = nnx.initializers.zeros,
      time_offset: np.timedelta64 = np.timedelta64(0, 's'),
      rngs: nnx.Rngs,
  ):
    self.coords = coords
    self.time_offset = time_offset
    self.features = nnx.data({
        k: param_type(initializer(rngs.params(), (366,) + c.shape))
        for k, c in coords.items()
    })

  def _get_day_index(self, time: jdt.Datetime) -> typing.Array:
    """Computes dayofyear index from Datetime."""
    year = time.delta.days / 365.25  # estimate day of year.
    day_idx = jnp.floor((year - jnp.floor(year)) * 365.25).astype(jnp.int32)
    return day_idx

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    time = inputs['time']
    if self.time_offset != np.timedelta64(0, 's'):
      time = time + cx.field(jdt.to_timedelta(self.time_offset))  # pyrefly: ignore[bad-argument-type]

    day_idx = cx.cpmap(self._get_day_index)(time)
    slice_array = functools.partial(
        jax.lax.dynamic_index_in_dim, axis=0, keepdims=False
    )
    slice_fn = cx.cmap(slice_array, out_axes='leading')

    outputs = {}
    for k, v in self.features.items():
      outputs[k] = slice_fn(v[...], day_idx).tag(self.coords[k])
    return outputs

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      *,
      reduce_hour_method: Literal['zero', 'mean'] = 'zero',
      **kwargs,
  ):
    """Updates `self.features` with data from dataset."""
    sim_units = kwargs['sim_units']
    day_axis = cx.LabeledAxis('dayofyear', 1 + np.arange(366))
    for key, feature in self.features.items():
      if key in dataset:
        da = dataset[key]
        if 'hour' in da.dims:
          if reduce_hour_method == 'zero':
            da = da.sel(hour=0)
          elif reduce_hour_method == 'mean':
            da = da.mean('hour')
          else:
            raise ValueError(f'Unsupported {reduce_hour_method=}')
        data_units = units.parse_units(da.attrs['units'])
        da = da.copy(data=sim_units.nondimensionalize(da.values * data_units))
        candidate = xarray_utils.from_xarray(da, (cx.LabeledAxis,))
        full_coord = cx.coords.compose(day_axis, self.coords[key])
        value = candidate.order_as(full_coord).data
        feature.set_value(parallelism.shard_like(value, feature.get_value()))


@nnx.dataclass
class PointCyclicCovariates(transforms.PytreeTransformABC):
  """Computes solar/seasonal harmonic covariates (and latitude) at query points.

  Evaluates unit-bounded covariates along `axis` (size
  `2 * include_latitude + 4 * num_harmonics`, 10 by default):
    - `sin(lat)`, `cos(lat)` if `include_latitude`;
    - harmonics `1..H` of local solar time, from `time`, `time_offset` and
      `longitude`;
    - harmonics `1..H` of day-of-year, from `time` and `time_offset`.

  Attributes:
    lon_key: Key in `inputs` providing query longitudes in degrees.
    lat_key: Key in `inputs` providing query latitudes in degrees.
    time_key: Key in `inputs` providing `jdt.Datetime` timestamps. Required.
    time_offset_key: Key in `inputs` providing time offsets in seconds. Zero
      offsets are assumed if not present.
    output_key: Key under which the cyclic covariate field is returned.
    num_harmonics: Number of diurnal and seasonal harmonics (`>= 0`).
    include_latitude: Whether to include `sin(lat)` and `cos(lat)` entries.
    axis: Coordinate (or axis name string) of shape
      `(2 * include_latitude + 4 * num_harmonics,)`. Defaults to
      `'cyclic_covariates_at_loc'`.
  """

  lon_key: str = 'longitude'
  lat_key: str = 'latitude'
  time_key: str = 'time'
  time_offset_key: str = 'time_offset'
  output_key: str = 'cyclic_covariates'
  num_harmonics: int = 2
  include_latitude: bool = True
  axis: cx.Coordinate | str = 'cyclic_covariates_at_loc'

  def __post_init__(self):
    if self.num_harmonics < 0:
      raise ValueError(
          f'`num_harmonics` must be >= 0, got {self.num_harmonics}.'
      )
    expected_size = 2 * self.include_latitude + 4 * self.num_harmonics
    if expected_size == 0:
      raise ValueError(
          'PointCyclicCovariates produces no covariates with'
          ' `include_latitude=False` and `num_harmonics=0`.'
      )
    if isinstance(self.axis, str):
      self.axis = cx.SizedAxis(self.axis, expected_size)
    if self.axis.shape != (expected_size,):
      raise ValueError(
          f'`axis` must have shape ({expected_size},), got {self.axis.shape=}.'
      )

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    missing = {self.lon_key, self.lat_key, self.time_key} - set(inputs)
    if missing:
      raise ValueError(f'Inputs are missing required keys {sorted(missing)}.')
    lon = inputs[self.lon_key]
    lat = inputs[self.lat_key]
    assert isinstance(self.axis, cx.Coordinate)
    time_field = inputs[self.time_key]
    time_data: Any = time_field.data
    days = cx.field(
        jax.lax.stop_gradient(time_data.delta.days.astype(jnp.float32)),
        time_field.coordinate,
    )
    seconds = cx.field(
        jax.lax.stop_gradient(time_data.delta.seconds.astype(jnp.float32)),
        time_field.coordinate,
    )
    if self.time_offset_key in inputs:
      time_offset = inputs[self.time_offset_key]
    else:
      time_offset = cx.cmap(jnp.zeros_like)(lon)

    def _compute(lon_deg, lat_deg, t_off, d, s):
      # Fraction of the day is computed from seconds to avoid float32 round-off
      # of the (large) day count; days only enter the slow seasonal cycle.
      local_seconds = s + t_off
      day_frac = jnp.mod(local_seconds, 86400.0) / 86400.0
      solar_angle = 2.0 * jnp.pi * (day_frac + lon_deg / 360.0)
      year_frac = (d + local_seconds / 86400.0) / 365.2422
      season_angle = 2.0 * jnp.pi * (year_frac - jnp.floor(year_frac))
      terms = []
      if self.include_latitude:
        lat_rad = jnp.deg2rad(lat_deg)
        terms.extend([jnp.sin(lat_rad), jnp.cos(lat_rad)])
      for k in range(1, self.num_harmonics + 1):
        terms.extend([jnp.sin(k * solar_angle), jnp.cos(k * solar_angle)])
      for k in range(1, self.num_harmonics + 1):
        terms.extend([jnp.sin(k * season_angle), jnp.cos(k * season_angle)])
      return jnp.stack(terms, axis=-1)

    out = cx.cmap(_compute, out_axes='leading')(
        lon, lat, time_offset, days, seconds
    ).tag(self.axis)
    return {self.output_key: out}
