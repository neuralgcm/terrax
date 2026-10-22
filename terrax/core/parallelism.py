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
"""Utilities for implementing SPMD model parallelism."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import contextvars
import copy
import dataclasses
import math
from typing import Any, Self, TypeAlias

import coordax as cx
import einops
import fiddle as fdl
from fiddle import selectors
from fiddle import tagging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import pytree_utils
from terrax.core import typing
import xarray

P = jax.sharding.PartitionSpec

# Type aliases for improved readability.
Schema = str
DimensionName = str
AxisPartition = str | tuple[str, ...] | None
ArrayPartition = tuple[AxisPartition, ...]
DimPartitions = dict[DimensionName, AxisPartition]

ArrayPartitions = dict[Schema, ArrayPartition]
FieldPartitions = dict[Schema, DimPartitions]

# Axis partitions specify how to partition a single dimension of an array, which
# can be either a single axis along the device mesh, specified as a `str`,
# a `tuple[str, ...]` if the axis is partitioned acorss multiple device axes
# or `None` if the dimension is not partitioned (i.e. replicated).

# Array partitions specify an AxisPartition per axis.
#
# For example, if we have a 3D array and we want to partition it across
# a device mesh `('x', 'y')`, along the first dimension, we would specify
#   `array_partition = (('x', 'y'), None, None)`.

# DimPartitions determines how each dimension of a field is split (partitioned)
# across multiple devices.
#
# You define this partitioning by associating each dimension with a device mesh.
# For example, if you have a device mesh defined as ('x', 'y', 'z') and you set
#   `dim_partitions = {'level': ('x', 'z'), 'lat': 'y'}`
# it means: (1) The 'level' dimension will be divided across the 'x' and 'z'
# devices of the mesh. (2) The 'lat' dimension will be divided across the 'y'
# device of the mesh. (3) Any dimensions not mentioned will be fully replicated
# on all devices.


def get_partition_spec(
    dims: tuple[str | None, ...], dim_partitions: DimPartitions
) -> P:
  """Returns partition spec for `field`."""
  return P(
      *[dim_partitions.get(d, None) if d is not None else None for d in dims]
  )


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CoordinateShard(cx.Coordinate):
  """Coordinate that represents a shard of a full coordinate system.

  This class helps tag array shards that represent a subset of values of a
  `full_coordinare`. The most common use case is during data ingestion where
  shards are read from disk and are not yet converted to a sharded array.
  """

  coordinate: cx.Coordinate
  spmd_mesh_shape: collections.OrderedDict[str, int]
  dimension_partitions: dict[str, None | str | tuple[str, ...]]

  def __hash__(self):
    """Hash implementation that avoids un-hashable mesh and partitions attrs."""
    spmd_mesh_tuple = tuple(self.spmd_mesh_shape.items())
    partitions_tuple = tuple(self.dimension_partitions.items())
    return hash((self.coordinate, spmd_mesh_tuple, partitions_tuple))

  def __eq__(self, other):
    return (
        isinstance(other, CoordinateShard)
        and self.coordinate == other.coordinate
        and self.spmd_mesh_shape == other.spmd_mesh_shape
        and self.dimension_partitions == other.dimension_partitions
    )

  @property
  def dims(self) -> tuple[str, ...]:
    return self.coordinate.dims  # sharding only affects the shape.  # pyrefly: ignore[bad-return]

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    dims, full_shape = self.dims, self.coordinate.shape
    shard_shape = []
    for dim, full_size in zip(dims, full_shape, strict=True):
      axes = self.dimension_partitions.get(dim, None)
      if not isinstance(axes, tuple):
        axes = (axes,)
      n_shards = math.prod(self.spmd_mesh_shape.get(axis, 1) for axis in axes)  # pyrefly: ignore[no-matching-overload]
      size, reminder = divmod(full_size, n_shards)
      if reminder:
        raise ValueError(
            f'Dimension {dim} has size {full_size} which is not divisible by'
            f' the number of shards {n_shards}.'
        )
      shard_shape.append(size)
    return tuple(shard_shape)

  @property
  def fields(self) -> dict[str, cx.Field]:
    """A maps from field names to their values."""
    return {}

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    return cx.coords.NoCoordinateMatch(
        'CoordinateShard cannot be deterministically inferred from xarray.'
    )


def get_unsharded(coordinate: cx.Coordinate) -> cx.Coordinate:
  """Returns the unsharded coordinate of a CoordinateShard."""
  unsharded = []
  for c in cx.coords.canonicalize(coordinate):
    unsharded.append(c.coordinate if isinstance(c, CoordinateShard) else c)
  return cx.coords.compose(*unsharded)


def rearrange_spmd_mesh(
    spmd_mesh: jax.sharding.Mesh,
    dim_partitions: DimPartitions,
    dims_to_spmd_mesh_axes: dict[str, str],
) -> jax.sharding.Mesh:
  """Returns rearranged & renamed spmd mesh representing the same partitions.

  This transformation helps interface with existing codes that rely on
  spmd mesh with specific axis names. It is parameterized by a mapping from
  dimension names that are to be aligned with spmd_mesh axes in the output to
  the desired spmd_mesh axis names.

  Args:
    spmd_mesh: input spmd mesh specifying the devices and axis names.
    dim_partitions: mapping from dimension names to partition axes in spmd_mesh.
    dims_to_spmd_mesh_axes: mapping from dimension names to be aligned with the
      trailing axes of the output spmd_mesh to their desired axis names.

  Returns:
    A new spmd mesh with trailing axes renamed to spmd_mesh_axis_names and
    rearranged to correspond to data partitioning that is consistent with
    dim_partitions and input spmd_mesh.
  """
  partitions = tuple(
      dim_partitions.get(k, None) for k in dims_to_spmd_mesh_axes
  )
  # Replace None with empty tuple to assign no partitioning to these dimensions.
  partitions = tuple(p if p is not None else tuple() for p in partitions)
  used_axes = sum(map(tuple, partitions), start=())
  retained_axes = tuple(
      axis for axis in spmd_mesh.axis_names if axis not in used_axes
  )

  def _as_rearrangement_string(x: str | tuple[str, ...]) -> str:
    """Converts partition description to rearrangement string."""
    if isinstance(x, str):
      return x
    if not x:
      return '1'  # effectively expands axis in the rearrangement.
    else:
      return '(' + ' '.join(x) + ')'  # groups multiple axes into one.

  rearrange_schema = (
      ' '.join(spmd_mesh.axis_names)  # current mesh axes.
      + '->'
      + ' '.join(retained_axes)
      + ' '
      + ' '.join(map(_as_rearrangement_string, partitions))
  )
  devices = einops.rearrange(spmd_mesh.devices, rearrange_schema)
  spmd_mesh_axis_names = tuple(dims_to_spmd_mesh_axes.values())
  return jax.sharding.Mesh(devices, retained_axes + spmd_mesh_axis_names)


# TODO(dkochkov): drop array_partitions specification when partitions can be
# specified using field_partitions.


@dataclasses.dataclass(frozen=True)
class Mesh:
  """Holds the device mesh and array/field partitioning strategies.

  Attributes:
    spmd_mesh: Jax sharding mesh object.
    array_partitions: Schemas for partitioning arrays. An array partition
      specifies how to partition a an array of a specific rank. For example, to
      partition a 3d array across a device mesh `('x', 'y')`, along the first
      dimension, we would use a `(('x', 'y'), None, None)` schema.
    field_partitions: Schemas for partitioning cx.Field. Each schema provides
      partitioning rules for array axes specified by the dimension names. For
      example, schema `{'vertical': {'level': ('z', 'x', 'y'), 'layer': 'z'}}`
      specifies that all `level` dimensions of a field will be partitioned
      across the `z`, `x`, and `y` axes of the device mesh and all `layer`
      dimensions will be partitioned only across the `z` axis.
    memory_kind: Optional memory kind to use when placing arrays on the mesh.
      Default is None, which means that the memory kind will be inherited from
      the devices.
  """

  spmd_mesh: jax.sharding.Mesh | None = None
  array_partitions: dict[Schema, ArrayPartition] = dataclasses.field(
      default_factory=dict
  )
  field_partitions: dict[Schema, DimPartitions] = dataclasses.field(
      default_factory=dict
  )
  memory_kind: str | None = None

  def __post_init__(self):
    self._validate_partitions()

  def __hash__(self) -> int:
    # Convert dicts to frozenset of items to make them hashable.
    array_partitions_hashable = frozenset(self.array_partitions.items())
    field_partitions_hashable = frozenset(
        (k, frozenset(v.items())) for k, v in self.field_partitions.items()
    )
    return hash((
        self.spmd_mesh,
        array_partitions_hashable,
        field_partitions_hashable,
        self.memory_kind,
    ))

  def _validate_partitions(self):
    """Validates that partitioning options are compatible with the mesh."""
    if self.spmd_mesh is not None:
      for k, v in self.array_partitions.items():
        partition_axes = jax.tree.leaves(v)
        if not set(partition_axes).issubset(set(self.axis_names)):
          raise ValueError(f'Spec {k, v} use axes not in {self.axis_names}')
        if np.unique(partition_axes).size != len(partition_axes):
          raise ValueError(f'Encountered duplicate in spec {k, v}')

      for _, dim_partitions in self.field_partitions.items():
        for k, v in dim_partitions.items():
          partition_axes = jax.tree.leaves(v)
          if not set(partition_axes).issubset(set(self.axis_names)):
            raise ValueError(f'Spec {k, v} use axes not in {self.axis_names}')
          if np.unique(partition_axes).size != len(partition_axes):
            raise ValueError(f'Encountered duplicate in spec {k, v}')

  @property
  def shape(self) -> collections.OrderedDict[str, int]:
    """Shape of the mesh."""
    return self.spmd_mesh.shape if self.spmd_mesh else collections.OrderedDict()

  @property
  def axis_names(self) -> tuple[str, ...]:
    """Names of the mesh axes."""
    return self.spmd_mesh.axis_names if self.spmd_mesh else ()

  def rearrange_spmd_mesh(
      self,
      schema_name: Schema | None,
      dims_to_spmd_mesh_axes: dict[str, str],
  ) -> jax.sharding.Mesh | None:
    """Returns rearranged/renamed spmd mesh representing the same partitions."""
    if self.spmd_mesh is None or schema_name is None:
      return None
    if schema_name not in self.field_partitions:
      raise ValueError(
          f'Schema {schema_name} not found in field_partitions.'
          f' Available schemas: {self.field_partitions.keys()}'
      )
    return rearrange_spmd_mesh(
        self.spmd_mesh,
        self.field_partitions[schema_name],
        dims_to_spmd_mesh_axes=dims_to_spmd_mesh_axes,
    )

  def with_sharding_constraint(
      self, inputs: typing.PyTreeState, schema: str | tuple[str, ...] | None
  ) -> typing.PyTreeState:
    """Applies `schema` sharding constraint to `inputs`."""
    if schema is None:
      return inputs
    if isinstance(schema, tuple):
      if not schema:  # case when we processed all constraints
        return inputs
      constrained = self.with_sharding_constraint(inputs, schema[1:])
      return self.with_sharding_constraint(constrained, schema[0])

    is_field = lambda x: isinstance(x, cx.Field)
    all_leaves, tree_def = jax.tree.flatten(inputs, is_leaf=is_field)
    array_leaves = [x for x in all_leaves if not is_field(x)]
    if np.unique([x.ndim for x in array_leaves]).size > 1:
      raise ValueError(
          'All arrays in the pytree must have the same rank. Got:'
          f' {[x.ndim for x in array_leaves]=}'
      )
    if self.spmd_mesh is None:
      return inputs
    leaves = []
    for x in all_leaves:
      if is_field(x):
        leaves.append(self._with_sharding_constraint_field(x, schema))
      else:
        leaves.append(self._with_sharding_constraint_array(x, schema))
    return jax.tree.unflatten(tree_def, leaves)

  def get_sharding(
      self,
      dims: tuple[str | None, ...],
      schema: Schema | None = None,
  ) -> jax.sharding.NamedSharding | None:
    """Returns `NamedSharding` for field with `dims` using sharding `schema`.

    Args:
      dims: Dimension names of a `cx.Field`.
      schema: Partitioning schema name in `self.field_partitions`, or `None` for
        replicated (`PartitionSpec()`).

    Returns:
      The resolved `NamedSharding`, or `None` if `self.spmd_mesh` is `None`.
    """
    if self.spmd_mesh is None:
      return None
    if schema is None:
      pspec = P()
    else:
      if schema not in self.field_partitions:
        raise ValueError(
            f'Schema {schema!r} not found in field_partitions.'
            f' Available schemas: {list(self.field_partitions.keys())}'
        )
      pspec = get_partition_spec(dims, self.field_partitions[schema])
    return jax.sharding.NamedSharding(
        self.spmd_mesh, pspec, memory_kind=self.memory_kind
    )

  def _with_sharding_constraint_array(
      self, array: jax.Array, schema: str
  ) -> jax.Array:
    """Applies sharding constraint to `array`.

    Args:
      array: array to apply sharding constraint to.
      schema: key in `self.array_partitions` indicating sharding schema.

    Returns:
      `inputs` with sharding constraint(s) applied.
    """
    p_specs = P(*self.array_partitions[schema])
    sharding = jax.sharding.NamedSharding(
        self.spmd_mesh, p_specs, memory_kind=self.memory_kind  # pyrefly: ignore[bad-argument-type]
    )
    return jax.lax.with_sharding_constraint(array, sharding)

  def _with_sharding_constraint_field(
      self, field: cx.Field, schema: str
  ) -> cx.Field:
    """Applies sharding constraint to `field`.

    Args:
      field: field to apply sharding constraint to.
      schema: key in `self.field_partitions` indicating sharding schema.

    Returns:
      Field with sharding constraint applied.
    """
    sharding = self.get_sharding(field.dims, schema)
    return jax.lax.with_sharding_constraint(field, sharding)  # pyrefly: ignore[bad-argument-type]

  def unshard(
      self, field_shards: dict[jax.Device, cx.Field], schema: str
  ) -> cx.Field:
    """Convert sharded fields to a single unsharded field.

    Args:
      field_shards: mapping from JAX device to coordax.Field indicating data to
        load on to that device.
      schema: key in `self.field_partitions` indicating sharding schema.

    Returns:
      A single coordax.Field with data from `field_shards` as a single JAX
      array with sharded inputs.
    """

    example_field = next(iter(field_shards.values()))
    sharding = self.get_sharding(example_field.dims, schema)
    assert sharding is not None

    coord = cx.get_coordinate(example_field)
    unsharded_coord = get_unsharded(coord)

    single_device_arrays = [
        jax.device_put(field.data, device)
        for device, field in field_shards.items()
    ]

    def make_array(*arrays: jax.Array) -> jax.Array:
      return jax.make_array_from_single_device_arrays(
          unsharded_coord.shape, sharding, list(arrays)
      )

    # We use jax.tree.map to handle the case where cx.Field.data contains a
    # pytree of JAX arrays (e.g., jax_datetime.Datetime), which are not
    # supported by jax.make_array_from_single_device_arrays.
    array = jax.tree.map(make_array, *single_device_arrays)
    return cx.Field(array).tag(unsharded_coord)


_REQUIRE_MESH = contextvars.ContextVar('require_mesh', default=False)


class _RequireMeshContext:
  """Context manager to enforce explicit mesh instantiation."""

  def __enter__(self):
    self._token = _REQUIRE_MESH.set(True)

  def __exit__(self, exc_type, exc_val, exc_tb):
    _REQUIRE_MESH.reset(self._token)


require_mesh = _RequireMeshContext()


def default_mesh() -> Mesh:
  """Returns a default Mesh, or raises an error if require_mesh is active."""
  if _REQUIRE_MESH.get():
    raise ValueError(
        'Implicit instantiation of default Mesh is prohibited in this context. '
        'Please provide an explicit mesh.'
    )
  return Mesh()


# TODO(dkochkov): Remove these temporary functions once we can rely on imposing
# sharding constraints on coordax.Field leaves.
def with_dycore_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies even sharding variants to inputs depending on input shape."""

  def f(y: jax.Array) -> jax.Array:
    assert isinstance(mesh, Mesh)  # make pytype happy.
    if y.ndim == 1 and y.dtype == jnp.uint32:
      return y
    elif y.ndim == 3 and y.shape[0] != 1:
      return mesh.with_sharding_constraint(y, 'dycore_3d')
    elif y.ndim == 3 and y.shape[0] == 1:
      return mesh.with_sharding_constraint(y, 'dycore_3d_surface')
    elif y.ndim == 2:
      return mesh.with_sharding_constraint(y, 'dycore_2d')
    else:
      raise ValueError(f'Unsupported array shape: {y.shape}')

  if mesh is None or mesh.spmd_mesh is None:
    return inputs
  return pytree_utils.tree_map_over_nonscalars(f, inputs)  # pyrefly: ignore[bad-argument-type]


def with_physics_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies horizontal sharding variants depending on input shape."""

  def f(y: jax.Array) -> jax.Array:
    assert isinstance(mesh, Mesh)  # make pytype happy.
    if y.ndim == 1 and y.dtype == jnp.uint32:
      return y
    elif y.ndim == 3:
      return mesh.with_sharding_constraint(y, 'physics_3d')
    elif y.ndim == 2:
      return mesh.with_sharding_constraint(y, 'physics_2d')
    else:
      raise ValueError(f'Unsupported array shape: {y.shape}')

  if mesh is None or mesh.spmd_mesh is None:
    return inputs
  return pytree_utils.tree_map_over_nonscalars(f, inputs)  # pyrefly: ignore[bad-argument-type]


def with_dycore_to_physics_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies dycore sharding followed by physics sharding constraints."""
  return with_physics_sharding(mesh, with_dycore_sharding(mesh, inputs))


def with_physics_to_dycore_sharding(
    mesh: Mesh | None,
    inputs: typing.PyTreeState,
) -> typing.PyTreeState:
  """Applies physics sharding followed by dycore sharding constraints."""
  return with_dycore_sharding(mesh, with_physics_sharding(mesh, inputs))


#
# Helper functions for updating model state and Fiddle configs with Mesh.
#


def _apply_leaf_sharding(
    x: Any,
    sharding: jax.sharding.NamedSharding | None,
) -> Any:
  """Applies `sharding` to a single array leaf `x`."""
  # Inside JIT-compiled functions, leaves are abstract Tracers that cannot be
  # copied to host via `jax.device_get` or have concrete sharding inspected;
  # sharding is instead constrained symbolically via `with_sharding_constraint`.
  if isinstance(x, jax.core.Tracer):
    arr = jnp.asarray(x)
    return (
        jax.lax.with_sharding_constraint(arr, sharding)
        if sharding is not None
        else arr
    )
  if sharding is None:
    if isinstance(x, jax.Array):
      if not x.is_fully_addressable and not x.is_fully_replicated:
        x = jax.lax.with_sharding_constraint(
            x, x.sharding.update(spec=P())  # pyrefly: ignore[missing-attribute]
        )
      x = jax.device_get(x)
    return jnp.asarray(x)
  if isinstance(x, jax.Array):
    if x.sharding == sharding:
      return x
    # In eager mode, `with_sharding_constraint` requires the input array to be
    # uncommitted or on the same device set as `sharding`. Moving `x` to host
    # first allows resharding across different device sets (e.g., from a single
    # device to a multi-device mesh).
    if isinstance(sharding.mesh, jax.sharding.AbstractMesh) or (
        x.sharding.device_set != sharding.device_set
    ):
      x = jax.device_get(x)
  return jax.lax.with_sharding_constraint(jnp.asarray(x), sharding)


def shard_like(value: Any, like: Any) -> Any:
  """Returns `value` with array leaves sharded like matching leaves of `like`.

  Use this to update variables in place without changing their placement, e.g.
  `var.set_value(shard_like(new_value, var.get_value()))`. Leaves of `like`
  that are not `jax.Array`s with a `NamedSharding` (e.g. NumPy arrays, Python
  scalars or single-device arrays) leave the matching leaves of `value`
  unchanged.

  Args:
    value: Pytree of array leaves to shard.
    like: Pytree with the same structure as `value` whose leaves provide the
      target shardings.

  Returns:
    `value` with each array leaf placed with the sharding of the corresponding
    leaf of `like`.
  """

  def _place(x: Any, reference: Any) -> Any:
    if isinstance(reference, jax.Array) and isinstance(
        reference.sharding, jax.sharding.NamedSharding
    ):
      return _apply_leaf_sharding(x, reference.sharding)
    return x

  return jax.tree.map(_place, value, like)


def _find_unique_module_mesh(module: nnx.Module) -> Mesh | None:
  """Returns the unique `Mesh` attribute on `module`, or `None` if absent."""
  meshes = {v for v in vars(module).values() if isinstance(v, Mesh)}
  if len(meshes) > 1:
    raise ValueError(
        f'Module {type(module).__name__} defines multiple distinct Mesh '
        f'attributes: {meshes}.'
    )
  return next(iter(meshes)) if meshes else None


def _nearest_enclosing_mesh(
    path: tuple[Any, ...],
    meshes_by_path: Mapping[tuple[Any, ...], Mesh],
    default: Mesh | None = None,
) -> Mesh | None:
  """Returns the `Mesh` of the nearest module enclosing `path`, or `default`."""
  for depth in reversed(range(len(path))):  # `path[:0] == ()` is the root.
    if (mesh := meshes_by_path.get(path[:depth])) is not None:
      return mesh
  return default


class _UseModuleMesh:
  """Sentinel indicating that module should use its own `Mesh` attribute."""


USE_MODULE_MESH = _UseModuleMesh()


def _shard_variable(
    var: nnx.Variable,
    mesh: Mesh | None,
    *,
    use_context_mesh: bool = True,
    sim_state_partition_schema: str | None = None,
    param_partition_schema: str | None = None,
    param_partition_spec: P | None = None,
    check_no_conflict: bool = False,
) -> nnx.Variable:
  """Returns a copy of `var` with sharding constraints applied to its value."""
  metadata = var.get_metadata()
  meta_field_schema = metadata.get('field_partitions')
  is_sim_var = isinstance(var, typing.SimulationVariable)

  requested_schema = (
      sim_state_partition_schema if is_sim_var else param_partition_schema
  )
  if requested_schema is not None:
    if (
        check_no_conflict
        and meta_field_schema is not None
        and meta_field_schema != requested_schema
    ):
      raise ValueError(
          f'Variable has field_partitions={meta_field_schema!r}, which '
          f'conflicts with requested schema {requested_schema!r}.'
      )
    field_schema = requested_schema
  else:
    field_schema = meta_field_schema

  var_pspec = nnx.get_partition_spec(var).get_raw_value() or P()
  if param_partition_spec is not None:
    if (
        check_no_conflict
        and var_pspec != P()
        and var_pspec != param_partition_spec
    ):
      raise ValueError(
          f'Variable has partition spec {var_pspec}, which conflicts with '
          f'param_partition_spec={param_partition_spec}.'
      )
    pspec = param_partition_spec
  else:
    pspec = var_pspec

  # Resolve sharding for non-`cx.Field` array leaves, using `mesh.spmd_mesh`
  # first and falling back to NNX variable metadata or ambient JAX mesh when no
  # explicit `mesh` override is provided.
  if mesh is not None and mesh.spmd_mesh is not None:
    array_sharding = jax.sharding.NamedSharding(
        mesh.spmd_mesh, pspec, memory_kind=mesh.memory_kind
    )
  elif use_context_mesh and metadata.get('mesh') is not None:
    array_sharding = jax.sharding.NamedSharding(metadata['mesh'], pspec)
  elif use_context_mesh and not jax.sharding.get_abstract_mesh().empty:
    array_sharding = jax.sharding.NamedSharding(
        jax.sharding.get_abstract_mesh(), pspec
    )
  else:
    array_sharding = None

  def _shard_item(item: Any) -> Any:
    if cx.is_field(item):
      sharding = (
          mesh.get_sharding(item.dims, field_schema)
          if mesh is not None
          else None
      )
      return jax.tree.map(lambda x: _apply_leaf_sharding(x, sharding), item)
    if meta_field_schema is not None:
      raise ValueError(
          '`field_partitions` metadata requires `cx.Field` values, '
          f'got {type(item)}.'
      )
    return jax.tree.map(lambda x: _apply_leaf_sharding(x, array_sharding), item)

  sharded_val = jax.tree.map(_shard_item, var.get_value(), is_leaf=cx.is_field)
  return var.replace(sharded_val)


def shard_module_states(
    module: nnx.Module,
    mesh: Mesh | None | _UseModuleMesh = USE_MODULE_MESH,
    *,
    fallback_mesh: Mesh | None = None,
    sim_state_partition_schema: str | None = None,
    param_partition_schema: str | None = None,
    param_partition_spec: P | None = None,
    check_no_conflict: bool = False,
    filter: nnx.filterlib.Filter = ...,  # pylint: disable=redefined-builtin
) -> nnx.State:
  """Returns `nnx.state(module)` with sharding constraints applied.

  By default each `nnx.Variable` is sharded with the `Mesh` attribute of its
  nearest enclosing module: the module that owns it, or else the closest
  ancestor module defining a `Mesh`. Generic layers (e.g. `nnx.Linear`) and
  containers (e.g. `nnx.Dict`) hence inherit the `Mesh` of the component they
  belong to. `fallback_mesh` is used only when no enclosing module, including
  `module` itself, defines a `Mesh`. An explicit `mesh` overrides all module
  `Mesh` attributes; `mesh=None` (or a `Mesh` with `spmd_mesh=None`) unshards
  all state arrays onto the local default device.

  Variables are sharded as follows:
    - `typing.SimulationVariable`s wrapping `cx.Field`s use
      `sim_state_partition_schema`.
    - `cx.Field` parameters use `param_partition_schema`.
    - Array-valued variables (`nnx.Param`, etc.) use `param_partition_spec`.
  For each of these, `None` (the default) uses the sharding configuration from
  variable metadata (`field_partitions` or `out_sharding`) if present, and
  defaults to replication (`PartitionSpec()`) otherwise.

  Args:
    module: `nnx.Module` whose state variables should be sharded.
    mesh: Optional `parallelism.Mesh` (or `None` to unshard onto a single local
      device) overriding all module `Mesh` attributes. Defaults to using the
      `Mesh` of each variable's nearest enclosing module.
    fallback_mesh: `parallelism.Mesh` used for variables without a `Mesh` on
      any enclosing module. If `None`, such variables raise a `ValueError`.
    sim_state_partition_schema: Directive field partition schema name in
      `mesh.field_partitions` for `typing.SimulationVariable`s, or `None` to
      use variable metadata / replicate.
    param_partition_schema: Directive field partition schema name in
      `mesh.field_partitions` for `cx.Field` parameters, or `None` to use
      variable metadata / replicate.
    param_partition_spec: Directive `PartitionSpec` for non-`cx.Field` array
      variables (`nnx.Param`, etc.), or `None` to use variable metadata /
      replicate with `PartitionSpec()`.
    check_no_conflict: If `True`, raises `ValueError` when an explicit directive
      (`sim_state_partition_schema`, `param_partition_schema`, or
      `param_partition_spec`) conflicts with non-empty variable metadata.
    filter: Optional NNX filter passed to `nnx.state(module, filter)` to select
      which state variables to shard. Defaults to `...` (all variables).

  Returns:
    Sharded `nnx.State` suitable for updating `module` via
    `nnx.update(module, parallelism.shard_module_states(module, ...))`.
  """
  use_module_mesh = isinstance(mesh, _UseModuleMesh)
  meshes_by_path = {}
  if use_module_mesh:
    meshes_by_path = {
        path: module_mesh
        for path, submodule in nnx.iter_modules(module)
        if (module_mesh := _find_unique_module_mesh(submodule)) is not None
    }

  def _shard_path_var(path: tuple[Any, ...], var: nnx.Variable) -> nnx.Variable:
    if use_module_mesh:
      target_mesh = _nearest_enclosing_mesh(path, meshes_by_path, fallback_mesh)
      if target_mesh is None:
        raise ValueError(
            f'No module enclosing the variable at path {path} defines a `Mesh`'
            ' attribute and no `fallback_mesh` was provided.'
        )
    else:
      target_mesh = mesh
    return _shard_variable(
        var,
        target_mesh,
        use_context_mesh=use_module_mesh,
        sim_state_partition_schema=sim_state_partition_schema,
        param_partition_schema=param_partition_schema,
        param_partition_spec=param_partition_spec,
        check_no_conflict=check_no_conflict,
    )

  return nnx.map_state(_shard_path_var, nnx.state(module, filter))


MeshUpdates: TypeAlias = Mesh | None | Mapping[tagging.TagType, Mesh | None]


def update_mesh_properties(
    fiddle_config: fdl.Config,
    mesh: MeshUpdates = None,
    attrs_to_update: Sequence[str] = ('spmd_mesh',),
) -> fdl.Config:
  """Returns a copy of `fiddle_config` with updated `Mesh` properties.

  Args:
    fiddle_config: Input configuration that will be copied and updated.
    mesh: Reference `Mesh` whose `attrs_to_update` attributes are copied onto
      all `Mesh` configs in `fiddle_config`, or a mapping from Fiddle tags to
      reference meshes applied only to the `Mesh` configs carrying that tag. A
      `None` reference unsets `attrs_to_update`, restoring `Mesh` defaults.
    attrs_to_update: Names of `Mesh` attributes to update, applied uniformly to
      all selected `Mesh` configs; call repeatedly to update differently tagged
      meshes with different attributes. Defaults to `('spmd_mesh',)`, which
      updates the device mesh while preserving the partition schemas configured
      in `fiddle_config`.

  Returns:
    A copy of `fiddle_config` with updated `Mesh` properties.
  """
  mesh_attrs = {field.name for field in dataclasses.fields(Mesh)}
  if unknown_attrs := set(attrs_to_update) - mesh_attrs:
    raise ValueError(
        f'Unsupported Mesh attributes {unknown_attrs}; expected a subset of'
        f' {sorted(mesh_attrs)}.'
    )
  fiddle_config = copy.deepcopy(fiddle_config)
  if isinstance(mesh, Mapping):
    updates = [
        (selectors.select(fiddle_config, tag=tag), tagged_mesh)
        for tag, tagged_mesh in mesh.items()
    ]
  else:
    updates = [(selectors.select(fiddle_config, Mesh), mesh)]
  for selection, reference_mesh in updates:
    for mesh_cfg in selection:
      for attr in attrs_to_update:
        if reference_mesh is not None:
          setattr(mesh_cfg, attr, getattr(reference_mesh, attr))
        elif attr in mesh_cfg.__arguments__:
          # Unset rather than assign `Mesh()` defaults: Fiddle compares an
          # explicit `{}` as unequal to an unset `default_factory` argument.
          delattr(mesh_cfg, attr)
  return fiddle_config
