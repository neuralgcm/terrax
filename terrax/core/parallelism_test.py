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
"""Tests distributed parallelism helper module Mesh and its methods."""

import collections
import re

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
import fiddle as fdl
from flax import nnx
import jax
from jax import config  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np
from terrax.core import parallelism
from terrax.core import typing


class MeshTest(parameterized.TestCase):
  """Tests Mesh class and its sharding constraint methods."""

  def test_constructor(self):
    with self.subTest('no_sharding'):
      no_sharding_mesh = parallelism.default_mesh()
      self.assertIsNone(no_sharding_mesh.spmd_mesh)
      self.assertEqual(no_sharding_mesh.axis_names, ())
      self.assertEqual(no_sharding_mesh.shape, collections.OrderedDict())

    with self.subTest('222_mesh'):
      spmd_mesh = jax.sharding.Mesh(
          np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
      )
      mesh = parallelism.Mesh(spmd_mesh=spmd_mesh)
      self.assertEqual(mesh.spmd_mesh, spmd_mesh)
      self.assertEqual(mesh.axis_names, ('z', 'x', 'y'))
      self.assertEqual(
          mesh.shape, collections.OrderedDict([('z', 2), ('x', 2), ('y', 2)])
      )

  def test_partitions_checked_on_init(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )

    with self.subTest('valid_array_partitions'):
      array_partitions = {
          'vertical': (('z', 'x', 'y'), None, None),
          'horizontal': (None, ('z', 'x'), 'y'),
      }
      parallelism.Mesh(spmd_mesh, array_partitions=array_partitions)

    with self.subTest('invalid_array_partitions_unknown_name'):
      array_partitions = {'unknownB': (('B', 'x', 'y'), None, 'z')}
      with self.assertRaisesRegex(ValueError, re.escape(r'use axes not in')):
        parallelism.Mesh(spmd_mesh, array_partitions=array_partitions)

    with self.subTest('invalid_array_partitions_duplicate_axis'):
      array_partitions = {'p': (('x', 'x', 'y'), None, 'z')}
      with self.assertRaisesRegex(
          ValueError, re.escape(r'Encountered duplicate')
      ):
        parallelism.Mesh(spmd_mesh, array_partitions=array_partitions)

    with self.subTest('valid_field_partitions'):
      field_partitions = {
          'vertical': {'level': ('z', 'x', 'y'), 'layer': 'z'},
          'horizontal': {'lon': ('z', 'x'), 'lat': 'y'},
      }
      parallelism.Mesh(spmd_mesh, field_partitions=field_partitions)

    with self.subTest('invalid_field_partitions_unknown_name'):
      field_partitions = {'unknownB': {'level': ('B', 'x', 'y')}}
      with self.assertRaisesRegex(ValueError, re.escape(r'use axes not in')):
        parallelism.Mesh(spmd_mesh, field_partitions=field_partitions)

    with self.subTest('invalid_field_partitions_duplicate_axis'):
      field_partitions = {'p': {'level': ('z', 'z', 'x')}}
      with self.assertRaisesRegex(
          ValueError, re.escape(r'Encountered duplicate')
      ):
        parallelism.Mesh(spmd_mesh, field_partitions=field_partitions)

  def test_require_mesh_context(self):
    with self.subTest('outside_context'):
      # Instantiation should succeed.
      mesh = parallelism.default_mesh()
      self.assertIsInstance(mesh, parallelism.Mesh)

    with self.subTest('inside_context_raises'):
      with parallelism.require_mesh:
        with self.assertRaisesRegex(ValueError, 'Implicit instantiation'):
          parallelism.default_mesh()

    with self.subTest('inside_context_explicit_ok'):
      with parallelism.require_mesh:
        mesh = parallelism.Mesh()
        self.assertIsInstance(mesh, parallelism.Mesh)

  def test_array_constraint(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )
    array_partitions = {
        'vertical': (('z', 'x', 'y'), None, None),
        'horizontal': (None, ('z', 'x'), 'y'),
        'replicated': (None, None, None),
    }
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh, array_partitions=array_partitions
    )
    shape = (16, 8, 14)
    input_array = np.ones(shape)
    array_vertical = mesh.with_sharding_constraint(input_array, 'vertical')
    actual_shard_shape = array_vertical.sharding.shard_shape(shape)
    self.assertEqual(actual_shard_shape, (16 // (2 * 2 * 2), 8, 14))
    array_horizontal = mesh.with_sharding_constraint(input_array, 'horizontal')
    actual_shard_shape = array_horizontal.sharding.shard_shape(shape)
    self.assertEqual(actual_shard_shape, (16, 8 // (2 * 2), 14 // 2))
    array_replicated = mesh.with_sharding_constraint(input_array, 'replicated')
    actual_shard_shape = array_replicated.sharding.shard_shape(shape)
    self.assertEqual(actual_shard_shape, shape)

  def test_field_constraint(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )
    field_partitions = {
        'vertical': {'level': ('z', 'x', 'y'), 'layer': 'z'},
        'horizontal': {'lon': ('z', 'x'), 'lat': 'y'},
    }
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh, field_partitions=field_partitions
    )
    shape = (16, 8, 14)
    input_level_field = cx.field(np.ones(shape), 'level', 'lon', 'lat')
    input_layer_field = cx.field(np.ones(shape), 'layer', 'lon', 'lat')

    field_vertical = mesh.with_sharding_constraint(
        input_level_field, 'vertical'
    )
    actual_shard_shape = field_vertical.data.sharding.shard_shape(shape)
    self.assertEqual(actual_shard_shape, (16 // (2 * 2 * 2), 8, 14))

    field_vertical = mesh.with_sharding_constraint(
        input_layer_field, 'vertical'
    )
    actual_shard_shape = field_vertical.data.sharding.shard_shape(shape)
    self.assertEqual(actual_shard_shape, (16 // 2, 8, 14))

    field_horizontal = mesh.with_sharding_constraint(
        input_layer_field, 'horizontal'
    )
    actual_shard_shape = field_horizontal.data.sharding.shard_shape(shape)
    self.assertEqual(actual_shard_shape, (16, 8 // (2 * 2), 14 // 2))

  def test_mixed_pytree_constraint(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )
    array_partitions = {
        'vertical': (('z', 'x', 'y'), None),
    }
    field_partitions = {
        'vertical': {'level': ('z', 'x', 'y'), 'layer': 'z'},
    }
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        array_partitions=array_partitions,
        field_partitions=field_partitions,
    )
    input_pytree = {
        'a': np.ones((16, 7)),
        'b': cx.field(np.ones((16, 7)), 'level', 'y'),
        'c': np.zeros((32, 7)),
        'd': cx.field(np.zeros((2, 16, 7)), 'b', 'level', 'x'),
        'e': cx.field(np.arange(32), 'layer'),
    }
    sharded_pytree = mesh.with_sharding_constraint(input_pytree, 'vertical')
    a_shard_shape = sharded_pytree['a'].sharding.shard_shape((16, 7))
    self.assertEqual(a_shard_shape, (16 // (2 * 2 * 2), 7))
    b_shard_shape = sharded_pytree['b'].data.sharding.shard_shape((16, 7))
    self.assertEqual(b_shard_shape, (16 // (2 * 2 * 2), 7))
    c_shard_shape = sharded_pytree['c'].sharding.shard_shape((32, 7))
    self.assertEqual(c_shard_shape, (32 // (2 * 2 * 2), 7))
    d_shard_shape = sharded_pytree['d'].data.sharding.shard_shape((2, 16, 7))
    self.assertEqual(d_shard_shape, (2, 16 // (2 * 2 * 2), 7))
    e_shard_shape = sharded_pytree['e'].data.sharding.shard_shape((32,))
    self.assertEqual(e_shard_shape, (32 // 2,))

  def test_raises_if_arrays_have_different_ranks(self):
    array_partitions = {'vertical': (('z', 'x', 'y'), None)}
    mesh = parallelism.Mesh(array_partitions=array_partitions)
    with self.assertRaisesRegex(
        ValueError,
        re.escape('All arrays in the pytree must have the same rank.'),
    ):
      input_pytree = {'rank_2': np.ones((16, 7)), 'rank_3': np.ones((16, 7, 9))}
      mesh.with_sharding_constraint(input_pytree, 'vertical')


class _DycoreMeshTag(fdl.Tag):
  """Tag for dycore mesh in tests."""


class _PhysicsMeshTag(fdl.Tag):
  """Tag for physics mesh in tests."""


class ModuleWithShardingMock(nnx.Module):
  """Configurable mock module for testing `shard_module_states`."""

  def __init__(
      self,
      mesh: parallelism.Mesh | None = None,
      *,
      mesh_instances: dict[str, parallelism.Mesh] | None = None,
      include_prognostic: bool = False,
      prognostic_schema: str | None = None,
      include_field_param: bool = False,
      field_param_schema: str | None = None,
      include_nested_field_param: bool = False,
      nested_field_param_schema: str | None = None,
      include_array_param: bool = False,
      array_param_pspec: tuple[str | None, ...] | None = None,
      array_param_tag: str | None = None,
      include_single_device_param: bool = False,
      include_rng_key: bool = False,
  ):
    if mesh is not None:
      self.mesh = mesh
    for name, m in (mesh_instances or {}).items():
      setattr(self, name, m)
    if include_prognostic:
      self.temperature = typing.Prognostic(
          cx.field(jnp.ones((4, 8)), 'lon', 'lat'),
          field_partitions=prognostic_schema,
      )
    if include_field_param:
      self.orography = nnx.Param(
          cx.field(jnp.ones((4, 8)), 'lon', 'lat'),
          field_partitions=field_param_schema,
      )
    if include_nested_field_param:
      self.nested_field_param = nnx.Param(
          {'f': cx.field(jnp.ones((4, 8)), 'lon', 'lat')},
          field_partitions=nested_field_param_schema,
      )
    if include_array_param:
      init_fn = lambda: jnp.ones((4, 8))
      if array_param_pspec is not None:
        init_fn = nnx.with_partitioning(init_fn, array_param_pspec)
      kwargs = {'tag': array_param_tag} if array_param_tag else {}
      self.weights = nnx.Param(init_fn(), **kwargs)
    if include_single_device_param:
      self.committed_single_device = nnx.Param(
          jax.device_put(
              jnp.ones((4, 8)),
              jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
          )
      )
    if include_rng_key:
      self.rng_key = nnx.Variable(jax.random.key(0))


class CoupledModuleMock(nnx.Module):
  """Mock module combining components with and without their own Mesh."""

  def __init__(
      self,
      mesh_a: parallelism.Mesh,
      mesh_b: parallelism.Mesh,
      root_mesh: parallelism.Mesh | None = None,
  ):
    if root_mesh is not None:
      self.mesh = root_mesh
    self.comp_a = ModuleWithShardingMock(
        mesh_a, include_field_param=True, field_param_schema='schema'
    )
    self.comp_b = ModuleWithShardingMock(
        mesh_b, include_field_param=True, field_param_schema='schema'
    )
    self.comp_b.nested_without_mesh = ModuleWithShardingMock(
        None, include_field_param=True, field_param_schema='schema'
    )
    self.comp_b.container = nnx.Dict({
        'in_dict': nnx.Param(
            cx.field(jnp.ones((4, 8)), 'lon', 'lat'),
            field_partitions='schema',
        )
    })
    self.linear_without_mesh = nnx.Linear(4, 8, rngs=nnx.Rngs(0))


class HelperFunctionsTest(parameterized.TestCase):
  """Tests helper functions for dycore and physics constraints on arrays."""

  def test_rearrange_spmd_simple_rename(self):
    devices = np.array(jax.devices()).reshape((1, 2, 2, 2))
    spmd_mesh = jax.sharding.Mesh(devices, ('a', 'b', 'c', 'd'))
    partition = {'level': 'b', 'longitude': 'c', 'latitude': 'd'}
    dim_to_ax = {'level': 'z', 'longitude': 'x', 'latitude': 'y'}
    actual = parallelism.rearrange_spmd_mesh(spmd_mesh, partition, dim_to_ax)
    expected = jax.sharding.Mesh(devices, ('a', 'z', 'x', 'y'))
    self.assertEqual(actual, expected)

  def test_rearrange_spmd_reshape(self):
    devices = np.array(jax.devices())
    spmd_mesh = jax.sharding.Mesh(
        devices.reshape((1, 2, 2, 2)), ('a', 'b', 'c', 'd')
    )
    partition = {'longitude': ('b', 'c'), 'latitude': 'd'}
    dim_to_ax = {'level': 'z', 'longitude': 'x', 'latitude': 'y'}
    actual = parallelism.rearrange_spmd_mesh(spmd_mesh, partition, dim_to_ax)
    expected = jax.sharding.Mesh(
        devices.reshape((1, 1, 4, 2)), ('a', 'z', 'x', 'y')
    )
    self.assertEqual(actual, expected)

  def test_rearrange_spmd_many_axes_reshape_and_transpose(self):
    devices = np.array(jax.devices())
    spmd_mesh = jax.sharding.Mesh(
        devices.reshape((1, 1, 2, 2, 2)), ('e', 'a', 'b', 'c', 'd')
    )
    partition = {'longitude': ('c', 'd'), 'latitude': 'b'}
    dim_to_ax = {'level': 'z', 'longitude': 'x', 'latitude': 'y'}
    actual = parallelism.rearrange_spmd_mesh(spmd_mesh, partition, dim_to_ax)
    expected = jax.sharding.Mesh(
        np.swapaxes(devices.reshape((1, 1, 1, 2, 4)), 3, 4),
        axis_names=('e', 'a', 'z', 'x', 'y'),
    )
    self.assertEqual(actual, expected)

  def test_dycore_sharding(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )
    array_partitions = {
        'dycore_3d': ('z', 'x', 'y'),
        'dycore_3d_surface': (None, 'x', 'y'),
        'dycore_2d': ('x', 'y'),
    }
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh, array_partitions=array_partitions
    )
    inputs = {
        'volume': np.ones((8, 32, 16)),
        'surface': np.ones((1, 32, 16)),
        '2d_surface': np.ones((32, 16)),
        'rng': jax.random.PRNGKey(0),
    }
    with mesh.spmd_mesh:
      sharded_inputs = parallelism.with_dycore_sharding(mesh, inputs)
      volume_shard_shape = sharded_inputs['volume'].sharding.shard_shape(
          (8, 32, 16)
      )
      self.assertEqual(volume_shard_shape, (8 // 2, 32 // 2, 16 // 2))
      surface_shard_shape = sharded_inputs['surface'].sharding.shard_shape(
          (1, 32, 16)
      )
      self.assertEqual(surface_shard_shape, (1, 32 // 2, 16 // 2))
      two_d_shard_shape = sharded_inputs['2d_surface'].sharding.shard_shape(
          (32, 16)
      )
      self.assertEqual(two_d_shard_shape, (32 // 2, 16 // 2))
      rng_shard_shape = sharded_inputs['rng'].sharding.shard_shape((2,))
      self.assertEqual(rng_shard_shape, (2,))

  def test_physics_sharding(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )
    array_partitions = {
        'physics_3d': (None, ('x', 'z'), 'y'),
        'physics_2d': (('x', 'z'), 'y'),
    }
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh, array_partitions=array_partitions
    )
    inputs = {
        'volume': np.ones((8, 32, 16)),
        'surface': np.ones((1, 32, 16)),
        '2d_surface': np.ones((32, 16)),
        'rng': jax.random.PRNGKey(0),
    }
    with mesh.spmd_mesh:
      sharded_inputs = parallelism.with_physics_sharding(mesh, inputs)
      volume_shard_shape = sharded_inputs['volume'].sharding.shard_shape(
          (8, 32, 16)
      )
      self.assertEqual(volume_shard_shape, (8, 32 // (2 * 2), 16 // 2))
      surface_shard_shape = sharded_inputs['surface'].sharding.shard_shape(
          (1, 32, 16)
      )
      self.assertEqual(surface_shard_shape, (1, 32 // (2 * 2), 16 // 2))
      two_d_shard_shape = sharded_inputs['2d_surface'].sharding.shard_shape(
          (32, 16)
      )
      self.assertEqual(two_d_shard_shape, (32 // (2 * 2), 16 // 2))
      rng_shard_shape = sharded_inputs['rng'].sharding.shard_shape((2,))
      self.assertEqual(rng_shard_shape, (2,))

  def test_update_mesh_properties_roundtrip(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2, 2)), axis_names=['z', 'x', 'y']
    )
    with self.subTest('default_spmd_mesh_only_preserves_schemas'):
      base_cfg = fdl.Config(
          parallelism.Mesh,
          field_partitions={'physics': {'lon': 'x'}},
      )
      ref_mesh = parallelism.Mesh(spmd_mesh=spmd_mesh)
      updated_cfg = parallelism.update_mesh_properties(base_cfg, mesh=ref_mesh)
      self.assertEqual(
          fdl.build(updated_cfg),
          parallelism.Mesh(
              spmd_mesh=spmd_mesh,
              field_partitions={'physics': {'lon': 'x'}},
          ),
      )

      stripped_cfg = parallelism.update_mesh_properties(updated_cfg, mesh=None)
      self.assertEqual(stripped_cfg, base_cfg)
      self.assertEqual(
          fdl.build(stripped_cfg),
          parallelism.Mesh(field_partitions={'physics': {'lon': 'x'}}),
      )

    with self.subTest('custom_attrs_to_update'):
      base_cfg = fdl.Config(parallelism.Mesh)
      active_mesh = parallelism.Mesh(
          spmd_mesh=spmd_mesh,
          array_partitions={'vertical': ('z', None)},
          field_partitions={'physics': {'lon': 'x'}},
      )
      all_attrs = ('spmd_mesh', 'array_partitions', 'field_partitions')
      updated_cfg = parallelism.update_mesh_properties(
          base_cfg, mesh=active_mesh, attrs_to_update=all_attrs
      )
      self.assertEqual(fdl.build(updated_cfg), active_mesh)
      stripped_cfg = parallelism.update_mesh_properties(
          updated_cfg, mesh=None, attrs_to_update=all_attrs
      )
      self.assertEqual(stripped_cfg, base_cfg)

    with self.subTest('tag_based_mapping_updates'):
      dycore_cfg = fdl.Config(
          parallelism.Mesh, field_partitions={'dycore': {'level': 'z'}}
      )
      physics_cfg = fdl.Config(
          parallelism.Mesh, field_partitions={'physics': {'lon': 'x'}}
      )
      tagged_cfg = fdl.Config(
          dict, dycore_mesh=dycore_cfg, physics_mesh=physics_cfg
      )
      fdl.add_tag(tagged_cfg, 'dycore_mesh', _DycoreMeshTag)
      fdl.add_tag(tagged_cfg, 'physics_mesh', _PhysicsMeshTag)
      updated_tagged = parallelism.update_mesh_properties(
          tagged_cfg,
          mesh={
              _DycoreMeshTag: parallelism.Mesh(spmd_mesh=spmd_mesh),
              _PhysicsMeshTag: None,
          },
      )
      built_tagged = fdl.build(updated_tagged)
      self.assertEqual(built_tagged['dycore_mesh'].spmd_mesh, spmd_mesh)
      self.assertEqual(
          built_tagged['dycore_mesh'].field_partitions,
          {'dycore': {'level': 'z'}},
      )
      self.assertIsNone(built_tagged['physics_mesh'].spmd_mesh)

      # Different attributes for different tags compose via repeated calls.
      custom_mesh = parallelism.Mesh(
          spmd_mesh=spmd_mesh, field_partitions={'custom': {'lat': 'y'}}
      )
      composed = parallelism.update_mesh_properties(
          updated_tagged,
          mesh={_PhysicsMeshTag: custom_mesh},
          attrs_to_update=('spmd_mesh', 'field_partitions'),
      )
      built_composed = fdl.build(composed)
      self.assertEqual(built_composed['dycore_mesh'].spmd_mesh, spmd_mesh)
      self.assertEqual(
          built_composed['dycore_mesh'].field_partitions,
          {'dycore': {'level': 'z'}},
      )
      self.assertEqual(built_composed['physics_mesh'], custom_mesh)

      with self.assertRaisesRegex(ValueError, 'Unsupported Mesh attributes'):
        parallelism.update_mesh_properties(
            tagged_cfg, attrs_to_update=('bad_attr',)
        )

  def test_shard_like(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), axis_names=['x', 'y']
    )
    sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec('x', 'y')
    )
    like = {
        'array': jax.device_put(jnp.zeros((4, 8)), sharding),
        'field': cx.field(
            jax.device_put(jnp.zeros((2, 4)), sharding), 'a', 'b'
        ),
        'numpy': np.zeros(3),
        'scalar': 0.0,
    }
    value = {
        'array': np.ones((4, 8)),
        'field': cx.field(np.ones((2, 4)), 'a', 'b'),
        'numpy': np.ones(3),
        'scalar': 1.0,
    }

    result = parallelism.shard_like(value, like)

    self.assertEqual(result['array'].sharding, sharding)
    np.testing.assert_array_equal(result['array'], np.ones((4, 8)))
    self.assertEqual(result['field'].data.sharding, sharding)
    self.assertEqual(result['field'].dims, ('a', 'b'))
    # Leaves without a `NamedSharding` reference are returned unchanged.
    self.assertIsInstance(result['numpy'], np.ndarray)
    self.assertEqual(result['scalar'], 1.0)

    with self.assertRaises(ValueError):
      parallelism.shard_like({'a': 1.0}, {'b': 1.0})

  def test_shard_module_states_basic_and_unshard(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), axis_names=['x', 'y']
    )
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={'field_schema': {'lon': 'x', 'lat': 'y'}},
    )

    with nnx.use_eager_sharding(False):
      module = ModuleWithShardingMock(
          mesh,
          include_array_param=True,
          array_param_pspec=(None, 'y'),
          array_param_tag='frozen',
          include_nested_field_param=True,
          nested_field_param_schema='field_schema',
          include_single_device_param=True,
          include_rng_key=True,
      )

    nnx.update(module, parallelism.shard_module_states(module))
    replicated_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec()
    )
    pspec_y_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec(None, 'y')
    )
    xy_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec('x', 'y')
    )
    self.assertEqual(module.weights.get_value().sharding, pspec_y_sharding)
    self.assertEqual(module.weights.tag, 'frozen')
    self.assertEqual(
        module.committed_single_device.get_value().sharding, replicated_sharding
    )
    self.assertEqual(module.rng_key.get_value().sharding, replicated_sharding)
    nested_field = module.nested_field_param.get_value()['f']
    self.assertIsInstance(nested_field, cx.Field)
    self.assertEqual(nested_field.dims, ('lon', 'lat'))
    self.assertEqual(nested_field.data.sharding, xy_sharding)

    with self.subTest('unshard_with_mesh_none'):
      unsharded_state = parallelism.shard_module_states(module, mesh=None)
      single_device_sharding = jax.sharding.SingleDeviceSharding(
          jax.local_devices()[0]
      )
      self.assertEqual(
          unsharded_state['weights'].get_value().sharding,
          single_device_sharding,
      )
      self.assertEqual(
          unsharded_state['nested_field_param'].get_value()['f'].data.sharding,
          single_device_sharding,
      )
      self.assertEqual(unsharded_state['weights'].tag, 'frozen')

    with self.subTest('inside_jit'):
      with nnx.use_eager_sharding(False):
        unsharded_module = ModuleWithShardingMock(
            mesh, include_array_param=True, array_param_pspec=(None, 'y')
        )
      nnx.update(
          unsharded_module,
          parallelism.shard_module_states(unsharded_module, mesh=None),
      )

      @nnx.jit
      def shard_in_jit(m):
        nnx.update(m, parallelism.shard_module_states(m))
        return m

      jitted_module = shard_in_jit(unsharded_module)
      self.assertEqual(
          jitted_module.weights.get_value().sharding, pspec_y_sharding
      )

  def test_shard_module_states_sim_state_vs_params_and_conflicts(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), axis_names=['x', 'y']
    )
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={
            'physics': {'lon': 'x', 'lat': 'y'},
            'lon_only': {'lon': 'x'},
        },
    )

    replicated = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec()
    )
    physics_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec('x', 'y')
    )
    lon_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec('x', None)
    )

    with self.subTest('sim_state_sharded_while_field_params_stay_replicated'):
      module = ModuleWithShardingMock(
          mesh,
          include_prognostic=True,
          include_field_param=True,
          include_array_param=True,
      )
      nnx.update(
          module,
          parallelism.shard_module_states(
              module,
              sim_state_partition_schema='physics',
              check_no_conflict=True,
          ),
      )
      self.assertEqual(
          module.temperature.get_value().data.sharding, physics_sharding
      )
      self.assertEqual(module.orography.get_value().data.sharding, replicated)
      self.assertEqual(module.weights.get_value().sharding, replicated)

    with self.subTest('explicit_param_directives'):
      module = ModuleWithShardingMock(
          mesh,
          include_prognostic=True,
          include_field_param=True,
          include_array_param=True,
      )
      nnx.update(
          module,
          parallelism.shard_module_states(
              module,
              sim_state_partition_schema='physics',
              param_partition_schema='lon_only',
              param_partition_spec=jax.sharding.PartitionSpec(None, 'y'),
          ),
      )
      self.assertEqual(
          module.temperature.get_value().data.sharding, physics_sharding
      )
      self.assertEqual(module.orography.get_value().data.sharding, lon_sharding)
      self.assertEqual(
          module.weights.get_value().sharding,
          jax.sharding.NamedSharding(
              spmd_mesh, jax.sharding.PartitionSpec(None, 'y')
          ),
      )

    with self.subTest('missing_schema_raises'):
      module = ModuleWithShardingMock(
          parallelism.Mesh(spmd_mesh=spmd_mesh), include_prognostic=True
      )
      with self.assertRaisesRegex(ValueError, 'not found in field_partitions'):
        parallelism.shard_module_states(
            module, sim_state_partition_schema='physics'
        )

    with self.subTest('check_no_conflict_catches_metadata_mismatch'):
      with nnx.use_eager_sharding(False):
        annotated = ModuleWithShardingMock(
            mesh,
            include_field_param=True,
            field_param_schema='lon_only',
            include_array_param=True,
            array_param_pspec=(None, 'y'),
        )
      # None directives defer to metadata without conflict.
      state = parallelism.shard_module_states(annotated, check_no_conflict=True)
      self.assertEqual(
          state['orography'].get_value().data.sharding, lon_sharding
      )
      self.assertEqual(
          state['weights'].get_value().sharding,
          jax.sharding.NamedSharding(
              spmd_mesh, jax.sharding.PartitionSpec(None, 'y')
          ),
      )

      # Explicit directives that disagree with metadata raise when
      # check_no_conflict=True.
      with self.assertRaisesRegex(
          ValueError, 'conflicts with requested schema'
      ):
        parallelism.shard_module_states(
            annotated,
            param_partition_schema='physics',
            check_no_conflict=True,
        )

      with self.assertRaisesRegex(
          ValueError, 'conflicts with param_partition_spec'
      ):
        parallelism.shard_module_states(
            annotated,
            param_partition_spec=jax.sharding.PartitionSpec(),
            check_no_conflict=True,
        )

      # With check_no_conflict=False (default), explicit directives override
      # variable metadata.
      overridden = parallelism.shard_module_states(
          annotated,
          param_partition_schema='physics',
          param_partition_spec=jax.sharding.PartitionSpec(),
      )
      self.assertEqual(
          overridden['orography'].get_value().data.sharding, physics_sharding
      )
      self.assertEqual(overridden['weights'].get_value().sharding, replicated)

  def test_shard_module_states_nearest_enclosing_mesh(self):
    spmd_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), axis_names=['x', 'y']
    )
    mesh_a = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={'schema': {'lon': 'x'}},
    )
    mesh_b = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={'schema': {'lon': 'y'}},
    )

    coupled = CoupledModuleMock(mesh_a, mesh_b)

    # Variables with no Mesh on any enclosing module raise unless a
    # fallback_mesh is provided.
    with self.assertRaisesRegex(
        ValueError, 'No module enclosing the variable at path .*linear_without'
    ):
      parallelism.shard_module_states(coupled)

    # With fallback_mesh provided, components with their own Mesh (and their
    # nested submodules or containers without a Mesh) keep their enclosing
    # component's Mesh while top-level submodules without a Mesh use
    # fallback_mesh.
    nnx.update(
        coupled,
        parallelism.shard_module_states(coupled, fallback_mesh=mesh_a),
    )
    sharding_a = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec('x', None)
    )
    sharding_b = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec('y', None)
    )
    self.assertEqual(
        coupled.comp_a.orography.get_value().data.sharding,
        sharding_a,
    )
    self.assertEqual(
        coupled.comp_b.orography.get_value().data.sharding,
        sharding_b,
    )
    self.assertEqual(
        coupled.comp_b.nested_without_mesh.orography.get_value().data.sharding,
        sharding_b,
    )
    self.assertEqual(
        coupled.comp_b.container['in_dict'].get_value().data.sharding,
        sharding_b,
    )
    self.assertEqual(
        coupled.linear_without_mesh.kernel.get_value().sharding,
        jax.sharding.NamedSharding(spmd_mesh, jax.sharding.PartitionSpec()),
    )

    # When root module defines a Mesh, submodules without their own Mesh inherit
    # the root Mesh without requiring fallback_mesh, while comp_b and its
    # children still use comp_b's closer Mesh.
    coupled_with_root_mesh = CoupledModuleMock(mesh_a, mesh_b, root_mesh=mesh_a)
    nnx.update(
        coupled_with_root_mesh,
        parallelism.shard_module_states(
            coupled_with_root_mesh, fallback_mesh=None
        ),
    )
    self.assertEqual(
        coupled_with_root_mesh.comp_b.nested_without_mesh.orography.get_value().data.sharding,
        sharding_b,
    )
    self.assertEqual(
        coupled_with_root_mesh.linear_without_mesh.kernel.get_value().sharding,
        jax.sharding.NamedSharding(spmd_mesh, jax.sharding.PartitionSpec()),
    )

    duplicate_mesh_module = ModuleWithShardingMock(
        mesh_instances={'m1': mesh_a, 'm2': mesh_b},
        include_array_param=True,
    )
    with self.assertRaisesRegex(
        ValueError, 'multiple distinct Mesh attributes'
    ):
      parallelism.shard_module_states(duplicate_mesh_module)


if __name__ == '__main__':
  chex.set_n_cpu_devices(8)
  config.update('jax_traceback_filtering', 'off')
  absltest.main()
