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

"""Tests for checkpointing routines."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
import coordax as cx
from fiddle.experimental import auto_config
from flax import nnx
import jax
import numpy as np
from terrax.core import api
from terrax.core import checkpointing
from terrax.core import coordinates
from terrax.core import parallelism
from terrax.core import random_processes


class MockModel(api.Model):
  """Mock Model for testing."""

  def __init__(
      self,
      grid: cx.Coordinate,
      rngs: nnx.Rngs,
      mesh: parallelism.Mesh = parallelism.default_mesh(),
      host_numpy_in_init: bool = False,
  ):
    super().__init__(mesh=mesh)
    if host_numpy_in_init:
      # Simulates host-side NumPy conversion on a JAX array derived from rngs,
      # which raises TracerArrayConversionError when traced under nnx.jit.
      _ = np.asarray(jax.random.uniform(rngs(), (4,)))
    self.process = random_processes.UniformUncorrelated.construct(
        coord=grid,
        minval=0.0,
        maxval=1.0,
        rngs=rngs,
    )
    self.linear = nnx.Linear(in_features=1, out_features=1, rngs=rngs)

  def advance(self, *args, **kwargs):
    ...

  def assimilate(self, *args, **kwargs):
    ...

  def observe(self, *args, **kwargs):
    ...

  @property
  def timestep(self):
    ...


@auto_config.auto_config
def build_model(host_numpy_in_init: bool = False):
  rngs = nnx.Rngs(0)
  grid = coordinates.LonLatGrid.T21()
  mesh = parallelism.Mesh()
  model = MockModel(
      grid, rngs, mesh=mesh, host_numpy_in_init=host_numpy_in_init
  )
  return model


class CheckpointingTest(parameterized.TestCase):

  def test_save_and_load_roundtrip(self):
    path = os.path.join(self.create_tempdir(), 'checkpoint')
    model_cfg = build_model.as_buildable()
    model = api.Model.from_fiddle_config(model_cfg)
    checkpointing.save_checkpoint(model, path)
    restored_model = checkpointing.load_model_checkpoint(path)
    restored_model_params = nnx.state(restored_model)
    expected_model_params = nnx.state(model)
    chex.assert_trees_all_equal(restored_model_params, expected_model_params)
    self.assertEqual(model_cfg, restored_model.fiddle_config)

  def test_eager_build_and_load_roundtrip(self):
    path = os.path.join(self.create_tempdir(), 'eager_checkpoint')
    model_cfg = build_model.as_buildable(host_numpy_in_init=True)
    with self.assertRaises(jax.errors.TracerArrayConversionError):
      api.Model.from_fiddle_config(model_cfg, eager=False)

    model = api.Model.from_fiddle_config(model_cfg, eager=True)
    checkpointing.save_checkpoint(model, path)
    restored_model = checkpointing.load_model_checkpoint(path, eager=True)
    chex.assert_trees_all_equal(nnx.state(restored_model), nnx.state(model))

  def test_save_and_load_with_spmd_mesh(self):
    path = os.path.join(self.create_tempdir(), 'checkpoint')
    model_cfg = build_model.as_buildable()
    model_cfg.mesh.field_partitions = {'horizontal': {'longitude': 'batch'}}
    spmd_mesh = jax.sharding.Mesh(jax.devices()[:1], ('batch',))
    train_mesh = parallelism.Mesh(
        spmd_mesh=spmd_mesh,
        field_partitions={'horizontal': {'longitude': 'batch'}},
    )
    sharded_model = api.Model.from_fiddle_config(
        model_cfg, mesh=parallelism.Mesh(spmd_mesh=spmd_mesh)
    )
    self.assertEqual(sharded_model.mesh, train_mesh)
    self.assertEqual(sharded_model.fiddle_config, model_cfg)

    checkpointing.save_checkpoint(sharded_model, path)

    # Loading with mesh=None restores an unsharded single-device model while
    # preserving partition schemas defined in the saved config.
    unsharded_restored = checkpointing.load_model_checkpoint(path, mesh=None)
    self.assertEqual(
        unsharded_restored.mesh,
        parallelism.Mesh(
            field_partitions={'horizontal': {'longitude': 'batch'}}
        ),
    )
    self.assertEqual(unsharded_restored.fiddle_config, model_cfg)
    single_device_sharding = jax.sharding.SingleDeviceSharding(
        jax.local_devices()[0]
    )
    for leaf in jax.tree.leaves(
        checkpointing.split_model_state_for_saving(unsharded_restored)
    ):
      self.assertEqual(leaf.sharding, single_device_sharding)

    # Loading with an explicit parallelism.Mesh shards parameters onto that
    # mesh.
    target_mesh = parallelism.Mesh(spmd_mesh=spmd_mesh)
    restored_model = checkpointing.load_model_checkpoint(path, mesh=target_mesh)
    self.assertEqual(restored_model.mesh, train_mesh)
    expected_sharding = jax.sharding.NamedSharding(
        spmd_mesh, jax.sharding.PartitionSpec()
    )
    for leaf in jax.tree.leaves(
        checkpointing.split_model_state_for_saving(restored_model)
    ):
      self.assertEqual(leaf.sharding, expected_sharding)

  def test_without_paths(self):
    serialized = {
        'root': {'type': 'ref', 'key': 'model_1'},
        'objects': {
            'model_1': {
                'type': {'type': 'pyref', 'module': 'm', 'name': 'Model'},
                'items': [
                    ['x', {'type': 'leaf', 'value': 1, 'paths': ['<root>.x']}]
                ],
                'paths': ['<root>'],
            }
        },
    }
    expected = {
        'root': {'type': 'ref', 'key': 'model_1'},
        'objects': {
            'model_1': {
                'type': {'type': 'pyref', 'module': 'm', 'name': 'Model'},
                'items': [['x', {'type': 'leaf', 'value': 1}]],
            }
        },
    }
    self.assertEqual(checkpointing.without_paths(serialized), expected)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
