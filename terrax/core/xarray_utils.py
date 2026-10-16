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

"""Utilities for converting between xarray objects and coordax Fields.

Main entry points:

* `to_xarray` / `from_xarray`: convert `cx.Field` objects and (nested) dicts of
  them to xarray objects and back. `from_xarray` infers Terrax coordinates
  (e.g. `LonLatGrid`, `PressureLevels`, `TimeDelta`) from the data.
* `read_from_xarray`: read data according to a schema defined using
  `data_specs`, which maps data keys and variable names to expected coordinates
  (`cx.Coordinate`, `data_specs.CoordSpec` or `data_specs.OptionalSpec`).
  `read_sharded_from_xarray` does the same for shards of data in distributed
  data loading.
* `model_inputs_from_xarray` / `model_dynamic_inputs_from_xarray`: read the
  inputs supported by a model and the dynamic inputs it requires, using the
  schemas `model.inputs_spec` and `model.dynamic_inputs_spec` respectively.
* `ensure_timedelta_axis` / `ensure_time_axis`: index datasets by relative
  `timedelta`, as expected by Terrax models, or by absolute `time`.
"""

import collections
from collections.abc import Mapping, Sequence
import datetime
import functools
from typing import Any, TypeAlias, TypeVar, cast, overload

import coordax as cx
import jax
import numpy as np
from terrax.core import api
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import parallelism
from terrax.core import typing
import xarray


CORE_COORD_TYPES = (
    coordinates.TimeDelta,
    coordinates.LonLatGrid,
    coordinates.SphericalHarmonicGrid,
    coordinates.HybridLevels,
    coordinates.PressureLevels,
    coordinates.SigmaLevels,
    coordinates.SigmaBoundaries,
    coordinates.LayerLevels,
    cx.DummyAxis,
)


@overload
def to_xarray(fields: cx.Field) -> xarray.DataArray:
  ...


@overload
def to_xarray(fields: Mapping[str, cx.Field]) -> xarray.Dataset:
  ...


@overload
def to_xarray(
    fields: Mapping[str, Mapping[str, cx.Field]],
) -> dict[str, xarray.Dataset]:
  ...


@overload
def to_xarray(fields: Mapping[str, Any]) -> dict[str, Any]:
  ...


def to_xarray(
    fields: cx.Field | Mapping[str, Any],
) -> xarray.DataArray | xarray.Dataset | dict[str, Any]:
  """Converts `fields` to xarray objects, preserving their nesting structure.

  Conversion rules, applied recursively:

  * ``cx.Field`` -> ``xarray.DataArray``
  * ``dict[str, cx.Field]`` -> ``xarray.Dataset``
  * any other ``dict`` -> ``dict`` with the same keys and converted values.

  For example, ``to_xarray({'era5': {'u': u}, 'ocean': {'sst': sst}})`` returns
  a dictionary that maps ``'era5'`` and ``'ocean'`` to ``xarray.Dataset``s.

  Args:
    fields: field, dictionary of fields or nested dictionary of fields.

  Returns:
    xarray objects with the same nesting structure as `fields`.
  """
  if cx.is_field(fields):
    return fields.to_xarray()
  if not isinstance(fields, Mapping):
    raise TypeError(
        f'Expected a cx.Field or a dictionary of them, got {type(fields)}'
    )
  if all(cx.is_field(v) for v in fields.values()):
    return xarray.Dataset({k: v.to_xarray() for k, v in fields.items()})
  return {k: to_xarray(v) for k, v in fields.items()}


@overload
def from_xarray(
    data: xarray.DataArray,
    additional_coord_types: Sequence[type[cx.Coordinate]] = ...,
) -> cx.Field:
  ...


@overload
def from_xarray(
    data: xarray.Dataset,
    additional_coord_types: Sequence[type[cx.Coordinate]] = ...,
) -> dict[str, cx.Field]:
  ...


@overload
def from_xarray(
    data: Mapping[str, xarray.Dataset],
    additional_coord_types: Sequence[type[cx.Coordinate]] = ...,
) -> dict[str, dict[str, cx.Field]]:
  ...


@overload
def from_xarray(
    data: Mapping[str, Any],
    additional_coord_types: Sequence[type[cx.Coordinate]] = ...,
) -> dict[str, Any]:
  ...


def from_xarray(
    data: xarray.DataArray | xarray.Dataset | Mapping[str, Any],
    additional_coord_types: Sequence[type[cx.Coordinate]] = (),
) -> cx.Field | dict[str, Any]:
  """Converts xarray objects to fields, inverse of `to_xarray`.

  Conversion rules, applied recursively:

  * ``xarray.DataArray`` -> ``cx.Field``
  * ``xarray.Dataset`` -> ``dict[str, cx.Field]`` with an entry for each data
    variable.
  * ``dict`` -> ``dict`` with the same keys and converted values.

  Unlike `cx.from_xarray`, coordinates are inferred using Terrax coordinate
  types (see `CORE_COORD_TYPES`). Datasets indexed by absolute ``time`` should
  be converted with `ensure_timedelta_axis` first. To read data that should
  match an inputs spec, use `read_from_xarray` instead.

  Args:
    data: data array, dataset or (nested) dictionary of datasets to convert.
    additional_coord_types: coordinate types to try in addition to
      `CORE_COORD_TYPES` when inferring coordinates from the data.

  Returns:
    Fields with the same nesting structure as `data`.
  """
  coord_types = CORE_COORD_TYPES + tuple(additional_coord_types)
  if isinstance(data, xarray.DataArray):
    return cx.from_xarray(data, coord_types)
  if isinstance(data, xarray.Dataset):
    data_arrays = cast(dict[str, xarray.DataArray], dict(data))
    return {k: cx.from_xarray(v, coord_types) for k, v in data_arrays.items()}
  if isinstance(data, Mapping):
    return {k: from_xarray(v, additional_coord_types) for k, v in data.items()}
  raise TypeError(
      'Expected an xarray.DataArray, xarray.Dataset or a dictionary of them, '
      f'got {type(data)}'
  )


_DatasetOrNestedDataset = TypeVar(
    '_DatasetOrNestedDataset', xarray.Dataset, dict[str, xarray.Dataset]
)
_TimedeltaOrNested: TypeAlias = np.timedelta64 | dict[str, np.timedelta64]


def ensure_timedelta_axis(
    ds: _DatasetOrNestedDataset,
    timedelta_origin: _TimedeltaOrNested = np.timedelta64(0),
) -> _DatasetOrNestedDataset:
  """Returns `ds` indexed by `timedelta` rather than absolute `time`.

  Terrax models expect data with a `timedelta` dimension and absolute times
  stored in a `time` variable along it. If `ds` has a `time` dimension, it is
  replaced by ``timedelta = time - time[0] + timedelta_origin``. A `time`
  coordinate is converted to a data variable, so that it can be read as a
  field. Datasets that are already indexed by `timedelta` are returned as is.

  Args:
    ds: dataset or dictionary of datasets.
    timedelta_origin: value of `timedelta` at the first time step. If `ds` is a
      dictionary, can be a dictionary with the same keys.

  Returns:
    `ds` indexed by `timedelta`.
  """
  if not isinstance(ds, xarray.Dataset):
    if not isinstance(timedelta_origin, dict):
      timedelta_origin = {k: timedelta_origin for k in ds.keys()}
    return jax.tree.map(
        ensure_timedelta_axis,
        ds,
        timedelta_origin,
        is_leaf=lambda x: isinstance(x, xarray.Dataset),
    )
  if 'time' in ds.dims:
    ds = ds.assign_coords(timedelta=ds.time - ds.time[0] + timedelta_origin)
    ds = ds.swap_dims({'time': 'timedelta'})
  if 'time' in ds.coords:
    ds = ds.reset_coords('time')
  return ds


def ensure_time_axis(
    ds: _DatasetOrNestedDataset,
    reference_time: np.datetime64 | datetime.datetime | None = None,
) -> _DatasetOrNestedDataset:
  """Returns `ds` indexed by absolute `time` rather than `timedelta`.

  Inverse of `ensure_timedelta_axis`, e.g. for comparing model outputs from
  `to_xarray` with data indexed by `time`. The `time` variable along
  `timedelta` becomes the dimension coordinate and `timedelta` is kept as a
  non-dimension coordinate. Datasets without a `timedelta` dimension are
  returned as is.

  Args:
    ds: dataset or dictionary of datasets.
    reference_time: if provided, absolute times are computed as
      ``reference_time + timedelta``, replacing any `time` variable in `ds`.

  Returns:
    `ds` indexed by `time`.

  Raises:
    ValueError: if `ds` has a `timedelta` dimension, but neither a `time`
      variable nor `reference_time` is provided.
  """
  if not isinstance(ds, xarray.Dataset):
    return jax.tree.map(
        functools.partial(ensure_time_axis, reference_time=reference_time),
        ds,
        is_leaf=lambda x: isinstance(x, xarray.Dataset),
    )
  if 'timedelta' not in ds.dims:
    return ds
  if reference_time is not None:
    ds = ds.drop_vars('time', errors='ignore')
    ds = ds.assign_coords(time=ds['timedelta'] + np.datetime64(reference_time))
  elif 'time' not in ds:
    raise ValueError(
        'Cannot index dataset by `time` because it has no `time` variable, '
        f'consider passing `reference_time`. Got dataset: {ds}'
    )
  return ds.swap_dims({'timedelta': 'time'})


def read_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    in_spec: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]],
    strict_matches: bool = True,
) -> typing.NestedFields:
  """Reads fields specified by `in_spec` from `nested_data`.

  Unlike `from_xarray`, which converts all variables, this function only reads
  the variables listed in `in_spec`, checks that their dimensions match the
  spec and returns them tagged with coordinates finalized from the spec, see
  `data_specs.finalize_spec`. Datasets indexed by absolute `time` are first
  converted using `ensure_timedelta_axis`.

  Args:
    nested_data: datasets keyed by data key, e.g. ``{'era5': dataset}``.
    in_spec: coordinates or coordinate specs keyed by data key and variable
      name, e.g. `model.inputs_spec`. Variables wrapped in
      `data_specs.OptionalSpec` are only read if present in `nested_data`.
    strict_matches: if True, all coordinates must be inferred from the data
      using coordinate types from `in_spec` and `CORE_COORD_TYPES`, and axes
      with `AxisMatchRules.REPLACED` match rules must already match `in_spec`.
      If False, dimensions that cannot be inferred fall back to
      `cx.LabeledAxis` and `REPLACED` axes are replaced by those from `in_spec`.

  Returns:
    Fields keyed by data key and variable name. Data keys for which no
    variables are present are omitted.

  Raises:
    ValueError: if a variable that is not optional is missing from the data or
      if the dimensions of the data do not match `in_spec`.
  """

  def _get_field(
      da: xarray.DataArray, spec: cx.Coordinate | data_specs.CoordSpec
  ) -> cx.Field:
    coord_types = data_specs.get_coord_types(spec)
    if not strict_matches:
      coord_types += (cx.LabeledAxis,)
    return from_xarray(da, coord_types)

  nested_data = ensure_timedelta_axis(nested_data)
  result = {}
  for data_key, data_spec in in_spec.items():
    dataset = nested_data.get(data_key, {})
    specs, fields, missing_vars = {}, {}, []
    for k, v in data_spec.items():
      spec, is_optional = data_specs.unwrap_optional(v)
      if k in dataset:
        if isinstance(spec, data_specs.CoordSpec):
          specs[k] = spec
        else:
          specs[k] = data_specs.CoordSpec(spec)
        fields[k] = _get_field(dataset[k], spec)
      elif not is_optional:
        missing_vars.append(k)

    if missing_vars:
      raise ValueError(
          f'Specs for {data_key!r} contains {missing_vars=} that are '
          f'not in the corresponding dataset with keys {list(dataset.keys())}'
      )
    if not fields:
      continue

    result[data_key] = {}
    target_coords = data_specs.finalize_nested_spec(specs, fields)
    for k, v in fields.items():
      target_coord = target_coords[k]
      if strict_matches:
        v = v.untag(target_coord).tag(target_coord)
      else:
        v = v.untag(*target_coord.dims).tag(target_coord)
      result[data_key][k] = v
  return result


def read_sharded_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    in_spec: dict[str, dict[str, cx.Coordinate | data_specs.CoordLikeSpec]],
    mesh_shape: collections.OrderedDict[str, int],
    dim_partitions: parallelism.DimPartitions,
) -> typing.NestedFields:
  """Returns a `specs`-like structure of coordax.Fields from a `dataset` shard.

  This is a helpful function for annotating coordax.Field with full coordinates
  while reading shards of dataset in a distributed setting. By providing the
  mesh shape and how different dimensions are partitioned we can include full
  coordinate information by tagging the data with CoordinateShard objects. This
  can later be dropped once the data is converted to jax arrays and sharded
  across devices.

  Args:
    nested_data: dict of xarray datasets from which to read data.
    in_spec: nested dictionary that associates variables with coordinates.
    mesh_shape: shape of the sharding mesh indicating number of devices in each
      axis.
    dim_partitions: mapping from dimension names to labels of device axes that
      the dimension is partitioned across.

  Returns:
    A dictionary of dictionaries of coordax.Fields tagged with CoordinateShard
    coordinates.
  """

  def _wrap_axis(ax: cx.Coordinate) -> cx.Coordinate:
    return parallelism.CoordinateShard(ax, mesh_shape, dim_partitions)

  def wrap_coordinate_shard(coord_or_spec):
    if isinstance(coord_or_spec, data_specs.OptionalSpec):
      return data_specs.OptionalSpec(wrap_coordinate_shard(coord_or_spec.spec))
    elif isinstance(coord_or_spec, data_specs.CoordSpec):
      coord = coord_or_spec.coord
      coord = cx.coords.compose(*[_wrap_axis(ax) for ax in coord.axes])
      exact = data_specs.AxisMatchRules.EXACT
      replaced = data_specs.AxisMatchRules.REPLACED
      replace_dict = {exact: replaced}
      exact_to_replaced = lambda v: replace_dict.get(v, v)
      new_dim_match_rules = {
          dim: exact_to_replaced(coord_or_spec.dim_match_rules.get(dim, exact))  # pyrefly: ignore[no-matching-overload]
          for dim in coord_or_spec.coord.dims
      }
      return data_specs.CoordSpec(
          coord,
          new_dim_match_rules,  # pyrefly: ignore[bad-argument-type]
          coord_or_spec.optional_dims,
      )
    elif isinstance(coord_or_spec, cx.Coordinate):
      return wrap_coordinate_shard(data_specs.CoordSpec(coord_or_spec))
    else:
      raise ValueError(f'Unsupported coord or spec type: {coord_or_spec}')

  if_leaf = lambda x: not isinstance(x, dict)  # in_spec must be a nested dict.
  shard_specs = jax.tree.map(wrap_coordinate_shard, in_spec, is_leaf=if_leaf)
  return read_from_xarray(nested_data, shard_specs, strict_matches=False)


def model_inputs_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    model: api.InferenceModel | api.Model,
) -> typing.NestedFields:
  """Returns subset of `nested_data` supported by the model for assimilation."""
  return read_from_xarray(
      nested_data=nested_data,
      in_spec=model.inputs_spec,
      strict_matches=False,
  )


def model_dynamic_inputs_from_xarray(
    nested_data: dict[str, xarray.Dataset],
    model: api.InferenceModel | api.Model,
) -> typing.NestedFields:
  """Returns subset of `nested_data` required by the model."""
  return read_from_xarray(
      nested_data=nested_data,
      in_spec=model.dynamic_inputs_spec,
      strict_matches=False,
  )
