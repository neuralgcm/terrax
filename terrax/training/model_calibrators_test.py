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

from unittest import mock

from absl.testing import absltest
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from terrax.core import checkpointing as core_checkpointing
from terrax.core import coordinates
from terrax.core import parallelism
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units
from terrax.training import checkpointing as training_checkpointing
from terrax.training import model_calibrators
import xarray


class MockModule(nnx.Module, pytree=False):

  def __init__(self):
    self.updated = False
    self.kwargs = {}

  def update_from_xarray(self, dataset, **kwargs):
    self.updated = True
    self.dataset = dataset
    self.kwargs = kwargs


class Model(nnx.Module, pytree=False):

  def __init__(self):
    self.mesh = parallelism.default_mesh()
    self.submodule_a = MockModule()
    self.sub_dict = {'b': MockModule()}
    self.submodule_c = MockModule()


@nnx.dataclass
class MockModelParams(nnx.Module):
  l1: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(1.0))
  l2: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(2.0))
  mesh: parallelism.Mesh = nnx.static(
      default_factory=parallelism.default_mesh, kw_only=True
  )


@nnx.dataclass
class MockModelSubsetParams(nnx.Module):
  l1: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(1.0))
  mesh: parallelism.Mesh = nnx.static(
      default_factory=parallelism.default_mesh, kw_only=True
  )


@nnx.dataclass
class MockLargeModel(nnx.Module):
  l1: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(1.0))
  l2: nnx.Param = nnx.data(init=False)
  l3: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(3.0))
  mesh: parallelism.Mesh = nnx.static(
      default_factory=parallelism.default_mesh, kw_only=True
  )

  def __post_init__(self):
    self.l2 = self.l1  # l2 is an alias of l1


class ParentModel(nnx.Module):
  """Mock module for testing component update."""

  def __init__(self, val1=1.0, val2=2.0):
    self.mesh = parallelism.default_mesh()
    self.my_component = MockModelParams(l1=nnx.Param(val1), l2=nnx.Param(val2))


@nnx.dataclass
class MockSmallModel(nnx.Module):
  l2: nnx.Param = nnx.data(default_factory=lambda: nnx.Param(0.0))
  mesh: parallelism.Mesh = nnx.static(
      default_factory=parallelism.default_mesh, kw_only=True
  )


class ModelCalibratorsTest(absltest.TestCase):

  def test_load_component_tracks_shared_alias_in_incoming_params(self):
    model = MockSmallModel(l2=nnx.Param(0.0))
    loaded_model = MockLargeModel(l1=nnx.Param(42.0), l3=nnx.Param(99.0))

    with mock.patch.object(
        model_calibrators.checkpointing, 'load_model_checkpoint'
    ) as mock_load:
      mock_load.return_value = loaded_model
      # This ensures calibrator receives the loaded model that also defines l2
      # as an alias of l1.

      calibrator = model_calibrators.LoadModelComponentParams(
          component_key=None,
          ckpt_path_or_dir='/dummy/path',
          load_subset=True,
      )
      calibrator(model, None, None)
      self.assertEqual(model.l2.get_value(), 42.0)

  def test_load_component_tracks_shared_alias_in_source_params(self):
    model = MockLargeModel(l1=nnx.Param(0.0), l3=nnx.Param(0.0))
    loaded_model = MockSmallModel(l2=nnx.Param(42.0))

    with mock.patch.object(
        model_calibrators.checkpointing, 'load_model_checkpoint'
    ) as mock_load:
      mock_load.return_value = loaded_model
      # Incoming parameters provide values for l2, which in model is shared with
      # l1 and hence both should be updated.

      calibrator = model_calibrators.LoadModelComponentParams(
          component_key=None,
          ckpt_path_or_dir='/dummy/path',
          load_subset=True,
      )
      calibrator(model, None, None)

      self.assertEqual(model.l2.get_value(), 42.0)
      self.assertEqual(model.l1.get_value(), 42.0)

  def test_update_submodules_from_xarray(self):
    model = Model()
    dataset_a = xarray.Dataset({'a': 1})
    dataset_b = xarray.Dataset({'b': 2})

    specs = {
        (MockModule, 'submodule_a'): dataset_a,
        (MockModule, 'b'): dataset_b,
    }
    calibrator = model_calibrators.UpdateSubmodulesFromXarray(
        specs, kwargs={'foo': 'bar'}
    )

    calibrator(model, None, None)

    with self.subTest('updated_submodules'):
      self.assertTrue(model.submodule_a.updated)
      self.assertEqual(model.submodule_a.dataset, dataset_a)
      self.assertEqual(model.submodule_a.kwargs, {'foo': 'bar'})

      self.assertTrue(model.sub_dict['b'].updated)
      self.assertEqual(model.sub_dict['b'].dataset, dataset_b)
      self.assertEqual(model.sub_dict['b'].kwargs, {'foo': 'bar'})

    with self.subTest('untouched_submodules'):
      self.assertFalse(model.submodule_c.updated)

  def test_update_duplicates(self):
    model = Model()
    model.submodule_d = model.submodule_a  # Create a duplicate reference.
    dataset_d = xarray.Dataset({'d': 3})
    # Update using the alias 'submodule_d'
    specs = {(MockModule, 'submodule_d'): dataset_d}
    calibrator = model_calibrators.UpdateSubmodulesFromXarray(
        specs, kwargs={'baz': 'qux'}
    )

    calibrator(model, None, None)

    self.assertTrue(model.submodule_a.updated)
    self.assertEqual(model.submodule_a.dataset, dataset_d)
    self.assertEqual(model.submodule_a.kwargs, {'baz': 'qux'})
    # Ensure both point to the same updated object state
    self.assertTrue(model.submodule_d.updated)

  def test_missing_module_raises_error(self):
    model = Model()
    specs = {
        (MockModule, 'non_existent'): xarray.Dataset({'z': 9}),
    }
    calibrator = model_calibrators.UpdateSubmodulesFromXarray(specs)

    with self.assertRaisesRegex(ValueError, 'No module of type .* found'):
      calibrator(model, None, None)

  def test_update_submodules_from_xarray_keeps_sharding(self):
    grid = coordinates.LonLatGrid.T21()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',)),
        jax.sharding.PartitionSpec(),
    )
    mask = cx.field(jax.device_put(jnp.zeros(grid.shape), sharding), grid)

    class ModelWithMask(nnx.Module):

      def __init__(self):
        self.mesh = parallelism.default_mesh()
        self.mask = transforms.PrescribedFields({'land_sea_mask': mask})

    model = ModelWithMask()
    dataset = (
        cx.field(np.ones(grid.shape), grid)
        .to_xarray()
        .assign_attrs(units='dimensionless')
        .to_dataset(name='land_sea_mask')
    )
    model_calibrators.UpdateSubmodulesFromXarray(
        {(transforms.PrescribedFields, 'mask'): dataset},
        kwargs={'sim_units': units.DEFAULT_UNITS},
    )(model, None, None)

    updated = model.mask.prescribed_fields['land_sea_mask'].get_value()
    np.testing.assert_array_equal(updated.data, np.ones(grid.shape))
    self.assertEqual(updated.data.sharding, sharding)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockModelParams(l1=nnx.Param(1.0), l2=nnx.Param(2.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        is_training_checkpoint=False,
    )
    calibrator(model, None, None)

    self.assertEqual(model.l1.get_value(), 3.0)
    self.assertEqual(model.l2.get_value(), 4.0)
    mock_load.assert_called_once_with(path='/dummy/path', mesh=model.mesh)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_with_key(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = ParentModel(1.0, 2.0)

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key='my_component',
        ckpt_path_or_dir='/dummy/path',
        is_training_checkpoint=False,
    )
    calibrator(model, None, None)

    self.assertEqual(model.my_component.l1.get_value(), 3.0)
    self.assertEqual(model.my_component.l2.get_value(), 4.0)
    mock_load.assert_called_once_with(path='/dummy/path', mesh=model.mesh)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockModelSubsetParams(l1=nnx.Param(1.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=True,
    )
    calibrator(model, None, None)

    self.assertEqual(model.l1.get_value(), 3.0)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset_raises_when_false(
      self, mock_load
  ):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockModelSubsetParams(l1=nnx.Param(1.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=False,
    )

    with self.assertRaisesRegex(
        ValueError, 'Parameter structures do not match'
    ):
      calibrator(model, None, None)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset_raises_on_no_common_params(
      self, mock_load
  ):

    class Unrelated(nnx.Module):

      def __init__(self):
        self.mesh = parallelism.default_mesh()
        self.other = nnx.Param(0.0)

    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=True,
    )

    with self.assertRaisesRegex(ValueError, 'No parameters of types'):
      calibrator(Unrelated(), None, None)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset_shared_target_is_deterministic(
      self, mock_load
  ):
    # `l1` and `l2` alias one variable in the target model but are distinct in
    # the checkpoint; the first-visited target path (`l1`) provides the value.
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = MockLargeModel(l1=nnx.Param(0.0), l3=nnx.Param(0.0))

    model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=True,
    )(model, None, None)

    self.assertEqual(model.l1.get_value(), 3.0)
    self.assertEqual(model.l2.get_value(), 3.0)
    self.assertEqual(model.l3.get_value(), 0.0)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_keeps_target_sharding(self, mock_load):
    device = np.array(jax.devices()[:1])
    target_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(device, ('target',)), jax.sharding.PartitionSpec()
    )
    loaded_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(device, ('ckpt',)), jax.sharding.PartitionSpec()
    )
    place = lambda v, s: nnx.Param(jax.device_put(jnp.array(v), s))
    mock_load.return_value = MockModelParams(
        l1=place(3.0, loaded_sharding), l2=place(4.0, loaded_sharding)
    )

    for load_subset in (False, True):
      with self.subTest(load_subset=load_subset):
        model = MockModelParams(
            l1=place(1.0, target_sharding), l2=place(2.0, target_sharding)
        )
        model_calibrators.LoadModelComponentParams(
            component_key=None,
            ckpt_path_or_dir='/dummy/path',
            load_subset=load_subset,
        )(model, None, None)

        for param, expected in ((model.l1, 3.0), (model.l2, 4.0)):
          self.assertEqual(float(param.get_value()), expected)
          self.assertEqual(param.get_value().sharding, target_sharding)

  def test_set_variable_tags_and_clear(self):
    class CoupledParent(nnx.Module):

      def __init__(self):
        self.atmosphere = MockModelParams(l1=nnx.Param(1.0), l2=nnx.Param(2.0))
        self.ocean = MockModelParams(l1=nnx.Param(3.0), l2=nnx.Param(4.0))

    model = CoupledParent()
    tag_calibrator = model_calibrators.SetVariableTags(
        component_key='atmosphere',
        tag='non_trainable',
    )
    self.assertTrue(tag_calibrator.run_on_restore)
    tag_calibrator(model, None, None)

    trainable_filter = nnx.All(nnx.Param, nnx.Not(nnx.WithTag('non_trainable')))
    trainable_state = nnx.state(model, trainable_filter)
    frozen_state = nnx.state(model, nnx.WithTag('non_trainable'))

    self.assertIn('ocean', trainable_state)
    self.assertNotIn('atmosphere', trainable_state)
    self.assertIn('atmosphere', frozen_state)
    self.assertNotIn('ocean', frozen_state)

    clear_calibrator = model_calibrators.SetVariableTags(
        component_key='atmosphere',
        tag=None,
    )
    clear_calibrator(model, None, None)
    all_trainable = nnx.state(model, trainable_filter)
    self.assertIn('atmosphere', all_trainable)
    self.assertIn('ocean', all_trainable)

  def test_set_variable_tags_sequence_and_filter(self):
    class CoupledParent(nnx.Module):

      def __init__(self):
        self.atmosphere = MockModelParams(l1=nnx.Param(1.0), l2=nnx.Param(2.0))
        self.ocean = MockModelParams(l1=nnx.Param(3.0), l2=nnx.Param(4.0))

    model = CoupledParent()
    calibrator = model_calibrators.SetVariableTags(
        component_key=('atmosphere', 'ocean'),
        tag='frozen_l1',
        variable_filter=nnx.All(nnx.Param, nnx.PathContains('l1')),
    )
    calibrator(model, None, None)

    frozen_state = nnx.state(model, nnx.WithTag('frozen_l1'))
    self.assertIn('l1', frozen_state['atmosphere'])
    self.assertNotIn('l2', frozen_state['atmosphere'])
    self.assertIn('l1', frozen_state['ocean'])
    self.assertNotIn('l2', frozen_state['ocean'])

  def test_set_variable_tags_raises_on_no_matching_variables(self):
    model = ParentModel(1.0, 2.0)
    calibrator = model_calibrators.SetVariableTags(
        component_key='my_component',
        tag='non_trainable',
        variable_filter=nnx.WithTag('non_existent'),
    )
    with self.assertRaisesRegex(ValueError, 'No variables matching'):
      calibrator(model, None, None)

  def test_set_variable_tags_matches_path_filter_via_secondary_alias(self):

    class SharedTower(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(1.0)

    class ModelWithSharedTower(nnx.Module):

      def __init__(self):
        tower = SharedTower()
        # `nnx.iter_graph` visits `a_first` before `z_frozen` and yields the
        # shared tower's variables only under the `a_first` path.
        self.a_first = tower
        self.z_frozen = tower
        self.other = SharedTower()

    model = ModelWithSharedTower()
    model_calibrators.SetVariableTags(
        component_key=None,
        tag='frozen',
        variable_filter=nnx.All(nnx.Param, nnx.PathContains('z_frozen')),
    )(model, None, None)

    self.assertEqual(model.z_frozen.w.tag, 'frozen')
    self.assertEqual(model.a_first.w.tag, 'frozen')  # Same variable.
    self.assertFalse(model.other.w.has_metadata('tag'))

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_preserves_existing_tags(self, mock_load):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param(3.0), l2=nnx.Param(4.0)
    )
    model = ParentModel(1.0, 2.0)

    model_calibrators.SetVariableTags(
        component_key='my_component', tag='non_trainable'
    )(model, None, None)
    model_calibrators.LoadModelComponentParams(
        component_key='my_component',
        ckpt_path_or_dir='/dummy/path',
        is_training_checkpoint=False,
    )(model, None, None)

    self.assertEqual(model.my_component.l1.get_value(), 3.0)
    self.assertEqual(model.my_component.l1.tag, 'non_trainable')

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_component_key_supports_mapping_and_nested_paths(self, mock_load):
    class ModelWithOperators(nnx.Module):

      def __init__(self):
        self.mesh = parallelism.default_mesh()
        self.operators = nnx.data({
            'q07_era5:surface': MockModelParams(
                l1=nnx.Param(1.0), l2=nnx.Param(2.0)
            ),
            'other:surface': MockModelParams(
                l1=nnx.Param(5.0), l2=nnx.Param(6.0)
            ),
        })

    mock_load.return_value = MockModelParams(
        l1=nnx.Param(10.0), l2=nnx.Param(20.0)
    )
    model = ModelWithOperators()

    model_calibrators.SetVariableTags(
        component_key='operators.q07_era5:surface',
        tag='trainable',
    )(model, None, None)
    model_calibrators.LoadModelComponentParams(
        component_key='operators.q07_era5:surface',
        ckpt_path_or_dir='/dummy/path',
    )(model, None, None)

    op = model.operators['q07_era5:surface']
    self.assertEqual(op.l1.get_value(), 10.0)
    self.assertEqual(op.l2.get_value(), 20.0)
    self.assertEqual(op.l1.tag, 'trainable')
    self.assertFalse(model.operators['other:surface'].l1.has_metadata('tag'))

    with self.assertRaisesRegex(KeyError, 'missing:surface'):
      model_calibrators.SetVariableTags(
          component_key='operators.missing:surface', tag='trainable'
      )(model, None, None)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_raises_on_inner_value_mismatch(
      self, mock_load
  ):
    mock_load.return_value = MockModelParams(
        l1=nnx.Param({'a': 3.0}), l2=nnx.Param(4.0)
    )
    model = MockModelParams(l1=nnx.Param({'b': 1.0}), l2=nnx.Param(2.0))

    calibrator = model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
    )
    with self.assertRaisesRegex(
        ValueError, 'Parameter structures do not match'
    ):
      calibrator(model, None, None)

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_multiple_param_types(self, mock_load):

    class ParamAndStat(nnx.Module):

      def __init__(self, param: float, stat: float):
        self.mesh = parallelism.default_mesh()
        self.param = nnx.Param(param)
        self.stat = nnx.BatchStat(stat)

    mock_load.return_value = ParamAndStat(3.0, 4.0)
    model = ParamAndStat(1.0, 2.0)
    model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        param_types=(nnx.Param, nnx.BatchStat),
    )(model, None, None)

    self.assertEqual(model.param.get_value(), 3.0)
    self.assertEqual(model.stat.get_value(), 4.0)

  def test_set_variable_tags_untags_shared_subcomponents_via_top_level_key(
      self,
  ):
    class DownscalingOperator(nnx.Module):

      def __init__(self, coarse_op: nnx.Module):
        self.tower = MockModelParams(l1=nnx.Param(1.0), l2=nnx.Param(2.0))
        self.ref_coarse = coarse_op

    class ModelWithSharedOperators(nnx.Module):

      def __init__(self):
        coarse_op = MockModelParams(l1=nnx.Param(5.0), l2=nnx.Param(6.0))
        downscale_op = DownscalingOperator(coarse_op=coarse_op)
        self.operators = nnx.data({
            'tl63_era5:surface': coarse_op,
            'q07_era5:surface': downscale_op,
        })

    model = ModelWithSharedOperators()

    # Tag the downscaling operator (which also touches the shared coarse_op),
    # then clear tags on the coarse operator via its top-level key.
    model_calibrators.SetVariableTags(
        component_key='operators.q07_era5:surface',
        tag='trainable',
    )(model, None, None)
    model_calibrators.SetVariableTags(
        component_key='operators.tl63_era5:surface',
        tag=None,
    )(model, None, None)

    q07_op = model.operators['q07_era5:surface']
    tl63_op = model.operators['tl63_era5:surface']
    self.assertEqual(q07_op.tower.l1.tag, 'trainable')
    self.assertEqual(q07_op.tower.l2.tag, 'trainable')
    self.assertFalse(tl63_op.l1.has_metadata('tag'))
    self.assertFalse(q07_op.ref_coarse.l1.has_metadata('tag'))

  def test_load_model_component_params_sharded_onto_model_mesh(self):
    spmd_mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('batch',))
    mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh, field_partitions={'horizontal': {}}
    )
    ckpt_dir = self.create_tempdir().full_path
    saved_model = MockModelParams(
        l1=nnx.Param(jnp.array(3.0)), l2=nnx.Param(jnp.array(4.0))
    )
    saved_state = core_checkpointing.split_model_state_for_saving(saved_model)
    with training_checkpointing.training_manager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(),
        model_config_str='{}',
    ) as manager:
      manager.save(
          100,
          args=ocp.args.Composite(
              metadata=ocp.args.JsonSave({'step': 100}),
              ema_params=ocp.args.StandardSave(saved_state.params),
          ),
      )

    untrained = MockModelParams(
        l1=nnx.Param(jnp.array(0.0)), l2=nnx.Param(jnp.array(0.0))
    )
    nnx.update(
        untrained, parallelism.shard_module_states(untrained, mesh)
    )
    with mock.patch.object(
        training_checkpointing,
        'untrained_model_from_training_dir',
        return_value=untrained,
    ) as mock_untrained:
      model = MockModelParams(
          l1=nnx.Param(jnp.array(1.0)), l2=nnx.Param(jnp.array(2.0))
      )
      model.mesh = mesh
      model_calibrators.LoadModelComponentParams(
          component_key=None,
          ckpt_path_or_dir=ckpt_dir,
          ckpt_step=100,
          is_training_checkpoint=True,
      )(model, None, None)
      mock_untrained.assert_called_once_with(
          ckpt_dir, mesh=mesh, attrs_to_update=('spmd_mesh',), eager=False
      )

    expected_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec()
    )
    self.assertEqual(float(model.l1.get_value()), 3.0)
    self.assertEqual(float(model.l2.get_value()), 4.0)
    self.assertEqual(model.l1.get_value().sharding, expected_sharding)
    self.assertEqual(model.l2.get_value().sharding, expected_sharding)

    with mock.patch.object(
        model_calibrators.checkpointing,
        'load_model_checkpoint',
        return_value=MockModelParams(l1=nnx.Param(7.0), l2=nnx.Param(8.0)),
    ) as mock_model_load_from_model_mesh:
      model_with_mesh = MockModelParams(l1=nnx.Param(1.0), l2=nnx.Param(2.0))
      model_with_mesh.mesh = mesh
      model_calibrators.LoadModelComponentParams(
          component_key=None,
          ckpt_path_or_dir='/dummy/model_ckpt',
          is_training_checkpoint=False,
      )(model_with_mesh, None, None)
      mock_model_load_from_model_mesh.assert_called_once_with(
          path='/dummy/model_ckpt',
          mesh=mesh,
      )

  @mock.patch.object(model_calibrators.checkpointing, 'load_model_checkpoint')
  def test_load_model_component_params_subset_with_shifted_shared_submodules(
      self, mock_load
  ):
    class SharedFeature(nnx.Module):

      def __init__(self, val: float):
        self.weight = nnx.Param(val)

    class CoarseOperator(nnx.Module):

      def __init__(self, feature: SharedFeature, val: float):
        self.feature = feature
        self.bias = nnx.Param(val)

    class DownscalingOperator(nnx.Module):

      def __init__(self, coarse_op: CoarseOperator, val: float):
        self.ref_coarse = coarse_op
        self.tower = nnx.Param(val)

    class ModelWithOptionalDownscaling(nnx.Module):

      def __init__(
          self,
          include_downscaling: bool,
          feat_val: float,
          coarse_val: float,
          tower_val: float = 0.0,
      ):
        self.mesh = parallelism.default_mesh()
        feat = SharedFeature(feat_val)
        coarse = CoarseOperator(feat, coarse_val)
        ops = {'tl63:surface': coarse}
        if include_downscaling:
          # Alphabetically precedes 'tl63:surface', shifting first-visited
          # traversal paths of `coarse` and `feat` in `nnx.iter_graph`.
          ops['equiangular:surface'] = DownscalingOperator(coarse, tower_val)
        self.operators = nnx.data(dict(sorted(ops.items())))

    base_model = ModelWithOptionalDownscaling(
        include_downscaling=False, feat_val=10.0, coarse_val=20.0
    )
    mock_load.return_value = base_model

    target_model = ModelWithOptionalDownscaling(
        include_downscaling=True, feat_val=1.0, coarse_val=2.0, tower_val=5.0
    )
    model_calibrators.LoadModelComponentParams(
        component_key=None,
        ckpt_path_or_dir='/dummy/path',
        load_subset=True,
    )(target_model, None, None)

    self.assertEqual(
        target_model.operators['tl63:surface'].bias.get_value(), 20.0
    )
    self.assertEqual(
        target_model.operators['tl63:surface'].feature.weight.get_value(), 10.0
    )
    self.assertEqual(
        target_model.operators['equiangular:surface'].tower.get_value(), 5.0
    )

  def test_collect_normalization_stats_resets_diagnostic_state(self):
    x_axis = cx.SizedAxis('x', 2)

    class ToyModelWithDiagnostics(nnx.Module):

      def __init__(self):
        self.norm = transforms.StreamNorm({'u': x_axis})
        self.state_u = typing.Prognostic(cx.field(jnp.zeros(2), x_axis))
        self.diag_clock = typing.Diagnostic(jnp.array(0.0))
        self.diag_accum = typing.Diagnostic(cx.field(jnp.zeros(2), x_axis))

      def reset_diagnostic_state(self):
        self.diag_clock.set_value(jnp.array(0.0))
        self.diag_accum.set_value(cx.field(jnp.zeros(2), x_axis))

      def initialize_random_processes(self, rng):
        del rng

      def update_dynamic_inputs(self, dynamic_data):
        del dynamic_data

      def assimilate(self, inputs):
        self.state_u.set_value(inputs['data']['u'].isel(timedelta=-1))

      def advance(self):
        self.diag_clock.set_value(self.diag_clock.get_value() + 1.0)
        self.diag_accum.set_value(
            self.diag_accum.get_value() + self.state_u.get_value()
        )

      def observe(self, queries):
        del queries
        return {'data': self.norm({'u': self.diag_accum.get_value()})}

    model = ToyModelWithDiagnostics()
    td_axis = coordinates.TimeDelta(np.array([np.timedelta64(0, 'h')]))
    batch = (
        {'data': {'u': cx.field(jnp.array([[2.0, 4.0]]), td_axis, x_axis)}},
        {},
    )
    loader = mock.MagicMock()
    loader.build_eval_inputs.return_value = [batch, batch]
    stage = mock.MagicMock()
    stage.inputs_spec = {}
    stage.dynamic_inputs_spec = {}
    stage.time_sample_offset = np.timedelta64(6, 'h')
    stage.queries_spec = {'data': {'u': x_axis}}
    schedule = mock.MagicMock(stages=[stage])

    calibrator = model_calibrators.CollectNormalizationStats(
        train_stage_idx=0,
        rng_seed=0,
        time_start='2005-01-01',
        time_end='2005-01-02',
        batch_count=2,
    )
    calibrator(model, loader, schedule)

    self.assertFalse(model.norm.update_stats)
    np.testing.assert_allclose(
        model.norm.stream_norm.means.get_value()['u'].data,
        np.array([2.0, 4.0]),
    )
    self.assertEqual(float(model.diag_clock.get_value()), 0.0)
    np.testing.assert_allclose(
        model.diag_accum.get_value().data, np.zeros(2)
    )


if __name__ == '__main__':
  absltest.main()
