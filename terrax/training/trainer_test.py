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

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import optax
import pandas as pd
from terrax.core import api
from terrax.core import coordinates
from terrax.core import data_specs
from terrax.core import observation_operators
from terrax.core import transforms
from terrax.core import typing
from terrax.core import xarray_utils
from terrax.metrics import aggregation
from terrax.metrics import deterministic_metrics
from terrax.metrics import evaluators
from terrax.metrics import probabilistic_losses
from terrax.metrics import scaling
from terrax.toy_model_examples import lorenz96
from terrax.training import data_loading
from terrax.training import model_calibrators
from terrax.training import train_utils
from terrax.training import trainer


class DampingParameterization(nnx.Module):

  def __init__(self, rate: float = 0.5):
    self.rate = nnx.Param(jnp.array(rate))

  def __call__(self, prognostics: dict[str, cx.Field], dt: float) -> cx.Field:
    return -self.rate.get_value() * prognostics['x'] * dt


class TestLorenz96(lorenz96.Lorenz96):

  pass


class TestLorenz96WithTwoScales(lorenz96.Lorenz96WithTwoScales):

  def __post_init__(self):
    super().__post_init__()
    # Add a dummy param to ensure params is not empty
    self.dummy_param = nnx.Param(jnp.array(0.0))


class InMemorySaver(trainer.OnlineMetricsSaver):
  """Online metrics saver that stores metrics in memory for testing."""

  def __init__(self):
    self.metrics = []

  def save(self, step, metrics):
    self.metrics.append((step, metrics))


def _construct_spec(supported_spec, timedelta):
  """Construct concrete data specs from inputs spec."""

  # Assumes all supported_spec are CoordSpec with "any" timedelta.
  def _set_timedelta(c):
    assert isinstance(c, data_specs.CoordSpec)
    c = c.coord
    return cx.coords.compose(*[
        timedelta if isinstance(ax, coordinates.TimeDelta) else ax
        for ax in c.axes
    ])

  return jax.tree.map(
      _set_timedelta,
      supported_spec,
      is_leaf=lambda x: isinstance(x, data_specs.CoordSpec),
  )


def _make_coupled_model() -> TestLorenz96:
  """Returns a Lorenz96 model with a `damping` parameterization component."""
  damping = DampingParameterization(rate=0.5)
  model = TestLorenz96(
      k_axis=cx.LabeledAxis('k', np.arange(8)),
      dt=0.01,
      forcing=nnx.Param(jnp.array(8.0)),
      parameterizations=(damping,),
  )
  model.damping = damping
  return model


class TrainerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def create_model_and_data(
      self, multiscale: bool = False, add_fixed_point_obs_operator: bool = False
  ):
    k = cx.LabeledAxis('k', np.arange(8))
    t0 = jdt.Datetime.from_isoformat('2000-01-01')
    dummy_dt = coordinates.TimeDelta(np.timedelta64(0, 's') * np.arange(1))
    x_init = jnp.zeros(k.sizes['k'])
    if multiscale:
      j = cx.LabeledAxis('j', np.arange(4))
      kj = cx.coords.compose(k, j)
      model = TestLorenz96WithTwoScales(k, j)
      y_init = jnp.zeros((k.sizes['k'], j.sizes['j']))
      inputs = {
          'slow': {
              'x': cx.field(x_init[None, ...], dummy_dt, k),
              'time': cx.field(t0[None, ...], dummy_dt),
          },
          'fast': {'y': cx.field(y_init[None, ...], dummy_dt, kj)},
      }
    else:
      kj = None  # make pytype happy.
      model = TestLorenz96(k_axis=k, dt=0.01, forcing=nnx.Param(10.0))
      x_init = jnp.zeros(k.sizes['k'])
      inputs = {
          'slow': {
              'x': cx.field(x_init[None, ...], dummy_dt, k),
              'time': cx.field(t0[None, ...], dummy_dt),
          }
      }

    if add_fixed_point_obs_operator:
      sel_point = transforms.Sel({'k': 4})
      point_obs = observation_operators.TransformObservationOperator(sel_point)
      model.operators['fixed_point'] = point_obs

    inference_model = api.InferenceModel.from_model_api(model)
    state = inference_model.assimilate(inputs)

    trajectories = {}
    dt_slow = model.timestep * 4
    n_steps = 400
    _, trajectory = api.unroll_from_advance(
        inference_model,
        initial_state=state,
        timedelta=dt_slow,
        steps=n_steps,
        queries={'slow': {'x': k, 'time': cx.Scalar()}},
        prepend_init=True,
        trim_last=True,
    )
    trajectories.update(trajectory)
    if add_fixed_point_obs_operator:
      dt_fixed_point = dt_slow * 2
      _, trajectory_fixed_point = api.unroll_from_advance(
          inference_model,
          initial_state=state,
          timedelta=dt_fixed_point,
          steps=n_steps // 2,
          queries={'fixed_point': {'x': cx.Scalar(), 'time': cx.Scalar()}},
          prepend_init=True,
          trim_last=True,
      )
      trajectories.update(trajectory_fixed_point)
    if multiscale:
      assert isinstance(kj, cx.Coordinate)
      dt_fast = model.timestep
      _, trajectory_fast = api.unroll_from_advance(
          inference_model,
          initial_state=state,
          timedelta=dt_fast,
          steps=n_steps * 4,
          queries={'fast': {'y': kj, 'time': cx.Scalar()}},
          prepend_init=True,
          trim_last=True,
      )
      trajectories.update(trajectory_fast)

    # Data loaders expect datasets indexed by `time` rather than `timedelta`.
    all_data = xarray_utils.ensure_time_axis(
        xarray_utils.to_xarray(trajectories),
        reference_time=pd.Timestamp('2000-01-01'),
    )
    return model, all_data

  def _create_data_loader(self, all_data):
    spmd_mesh = trainer.create_spmd_mesh(1, 1, 1, 1)
    training_mesh = trainer.create_training_mesh(spmd_mesh, {'physics': {}})
    return data_loading.DataLoader(
        all_data=all_data,
        parallelism_mesh=training_mesh,
        loading_partition_schema='physics',
    )

  def _create_evaluators(self, dims_to_reduce, scale_by=()):
    aggregator = aggregation.Aggregator(
        dims_to_reduce=dims_to_reduce,
        weight_by=[],
        scale_by=scale_by,
    )
    mse = deterministic_metrics.MSE()
    eval_metrics = evaluators.Evaluator(
        metrics={'mse': mse}, aggregators={'mse': aggregator}
    )
    crps = probabilistic_losses.CRPS()
    loss = evaluators.Evaluator(
        metrics={'crps': crps}, aggregators={'crps': aggregator}
    )
    return eval_metrics, loss

  def _create_queries_specs(
      self, model, add_fixed_point_obs_operator: bool = False
  ):
    remove_timedelta = lambda c: cx.coords.compose(
        *[ax for ax in c.axes if not isinstance(ax, coordinates.TimeDelta)]
    )
    queries_specs = {}
    for data_key, k_spec in model.inputs_spec.items():
      queries_specs[data_key] = {
          k: remove_timedelta(v.coord) for k, v in k_spec.items() if k != 'time'
      }
    if add_fixed_point_obs_operator:
      queries_specs['fixed_point'] = {'x': cx.Scalar()}
    return queries_specs

  @parameterized.named_parameters(
      dict(testcase_name='default', include_initial_loss=False),
      dict(testcase_name='with_initial_loss', include_initial_loss=True),
  )
  def test_trainer(self, include_initial_loss: bool):
    model, all_data = self.create_model_and_data(multiscale=False)
    queries_specs = self._create_queries_specs(model)
    data_loader = self._create_data_loader(all_data)
    time_mask_coord = coordinates.TimeDelta([np.timedelta64(0, 's')])
    masked_t0_scaler = scaling.CoordinateMaskScaler(
        mask_coord=time_mask_coord,
        masked_value=0.5,
        unmasked_value=1.0,
    )
    leadtime_scaler = scaling.LeadTimeScaler(base_squared_error_in_hours=24.0)
    eval_metrics, loss = self._create_evaluators(
        dims_to_reduce=('ensemble', 'batch', 'k'),
        scale_by=[masked_t0_scaler, leadtime_scaler],
    )
    eval_metrics = evaluators.FlattenedEvaluator(eval_metrics)
    loss = evaluators.FlattenedEvaluator(loss)
    dt_slow = model.timestep * 4

    optimizer = optax.adam(1e-3)
    opt_config = trainer.OptimizationConfig(optimizer, ema_num_steps=10)

    train_stages = []
    for steps in [2, 5]:
      timedelta = coordinates.TimeDelta(np.arange(steps) * dt_slow)
      inputs_spec = _construct_spec(model.inputs_spec, timedelta)
      dyn_spec = _construct_spec(model.dynamic_inputs_spec, timedelta)
      train_stages.append(
          trainer.TrainStage(
              duration=steps,
              inputs_spec=inputs_spec,
              dynamic_inputs_spec=dyn_spec,
              queries_spec=queries_specs,
              loss=loss,
              time_sample_offset=dt_slow,
              batch_size_per_device=1,
              shuffle_buffer_size=0,
              train_time_slice=None,
              eval_time_slice=None,
          )
      )
    train_schedule = trainer.TrainSchedule(stages=train_stages)

    eval_timedelta = coordinates.TimeDelta(np.arange(10) * dt_slow)
    eval_inputs_spec = _construct_spec(model.inputs_spec, eval_timedelta)
    eval_dyn_spec = _construct_spec(model.dynamic_inputs_spec, eval_timedelta)
    eval_schedule = trainer.EvalSchedule(
        stages=[
            trainer.EvalSchema(
                cadence=2,
                inputs_spec=eval_inputs_spec,
                dynamic_inputs_spec=eval_dyn_spec,
                queries_spec=queries_specs,
                metrics_evaluator=eval_metrics,
                loss_evaluator=loss,
                time_sample_offset=dt_slow,
                batch_size_per_device=1,
                num_batches=1,
                train_time_slice=None,
                eval_time_slice=None,
                name='eval',
            )
        ]
    )

    checkpoint_config = trainer.CheckpointConfig(
        save_interval_steps=2,
        keep_every_n_steps=10,
        model_config_str='{}',
        metadata={},
    )

    auto_restart = trainer.AutoRestartConfig()
    remat = trainer.RematConfig()
    process_obs = transforms.Identity()
    metrics_saver = InMemorySaver()
    rollout_trainer = trainer.RolloutTrainer(
        experiment_dir=self.test_dir,
        model=model,
        data_loader=data_loader,
        process_observations=process_obs,
        calibration_modules=None,
        train_schedule=train_schedule,
        eval_schedule=eval_schedule,
        optimization_config=opt_config,
        initial_checkpoint=None,
        checkpoint_config=checkpoint_config,
        auto_restart_config=auto_restart,
        remat_config=remat,
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=metrics_saver,
        include_initial_loss=include_initial_loss,
    )
    rollout_trainer.run_training()

    self.assertTrue(metrics_saver.metrics)
    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'checkpoints')))

  @parameterized.named_parameters(
      dict(testcase_name='default', include_initial_loss=False),
      dict(testcase_name='with_initial_loss', include_initial_loss=True),
  )
  def test_trainer_nested(self, include_initial_loss: bool):
    model, all_data = self.create_model_and_data(multiscale=True)
    queries_specs = self._create_queries_specs(model)
    data_loader = self._create_data_loader(all_data)
    eval_metrics, loss = self._create_evaluators(
        dims_to_reduce=('ensemble', 'batch', 'k', 'j')
    )
    # Using nested evaluators for nested inputs.
    eval_metrics = evaluators.NestedEvaluators(
        evaluators={'slow': eval_metrics, 'fast': eval_metrics}
    )
    loss = evaluators.NestedEvaluators(evaluators={'slow': loss, 'fast': loss})

    optimizer = optax.adam(1e-3)
    opt_config = trainer.OptimizationConfig(optimizer, ema_num_steps=10)

    # Create train stage with nested data specs
    train_stages = []
    # 2 slow steps = 5 fast steps (0, 1, 2, 3, 4) if dt_slow=4*dt_fast
    slow_steps = 2
    dt_slow = model.timestep * 4
    dt_fast = model.timestep

    fast_steps = (slow_steps - 1) * 4 + 1
    timedelta_slow = coordinates.TimeDelta(np.arange(slow_steps) * dt_slow)
    timedelta_fast = coordinates.TimeDelta(np.arange(fast_steps) * dt_fast)

    # inputs_spec must match model structure
    inputs_spec = {
        'slow': _construct_spec(model.inputs_spec['slow'], timedelta_slow),
        'fast': _construct_spec(model.inputs_spec['fast'], timedelta_fast),
    }
    # Dynamic inputs are empty for this model, so we don't need to set them.
    dyn_spec = {}

    train_stages.append(
        trainer.TrainStage(
            duration=5,
            inputs_spec=inputs_spec,
            dynamic_inputs_spec=dyn_spec,
            queries_spec=queries_specs,
            loss=loss,
            time_sample_offset=dt_slow,
            batch_size_per_device=1,
            shuffle_buffer_size=0,
            train_time_slice=None,
            eval_time_slice=None,
        )
    )
    train_schedule = trainer.TrainSchedule(stages=train_stages)

    eval_slow_steps = 4
    eval_fast_steps = (eval_slow_steps - 1) * 4 + 1
    eval_td_slow = coordinates.TimeDelta(np.arange(eval_slow_steps) * dt_slow)
    eval_td_fast = coordinates.TimeDelta(np.arange(eval_fast_steps) * dt_fast)
    eval_inputs_spec = {
        'slow': _construct_spec(model.inputs_spec['slow'], eval_td_slow),
        'fast': _construct_spec(model.inputs_spec['fast'], eval_td_fast),
    }
    eval_dyn_spec = {}
    eval_schedule = trainer.EvalSchedule(
        stages=[
            trainer.EvalSchema(
                cadence=2,
                inputs_spec=eval_inputs_spec,
                dynamic_inputs_spec=eval_dyn_spec,
                queries_spec=queries_specs,
                metrics_evaluator=eval_metrics,
                loss_evaluator=loss,
                time_sample_offset=dt_slow,
                batch_size_per_device=1,
                num_batches=1,
                train_time_slice=None,
                eval_time_slice=None,
                name='eval',
            )
        ]
    )

    checkpoint_config = trainer.CheckpointConfig(
        save_interval_steps=2,
        keep_every_n_steps=10,
        model_config_str='{}',
        metadata={},
    )

    auto_restart = trainer.AutoRestartConfig()
    remat = trainer.RematConfig()
    process_obs = transforms.Identity()
    metrics_saver = InMemorySaver()
    rollout_trainer = trainer.RolloutTrainer(
        experiment_dir=self.test_dir,
        model=model,
        data_loader=data_loader,
        process_observations=process_obs,
        calibration_modules=None,
        train_schedule=train_schedule,
        eval_schedule=eval_schedule,
        optimization_config=opt_config,
        initial_checkpoint=None,
        checkpoint_config=checkpoint_config,
        auto_restart_config=auto_restart,
        remat_config=remat,
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=metrics_saver,
        include_initial_loss=include_initial_loss,
    )
    rollout_trainer.run_training()

    self.assertTrue(metrics_saver.metrics)
    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'checkpoints')))

  def test_all_features(self):
    model, all_data = self.create_model_and_data(
        multiscale=True, add_fixed_point_obs_operator=True
    )
    queries_specs = self._create_queries_specs(
        model, add_fixed_point_obs_operator=False
    )
    data_loader = self._create_data_loader(all_data)
    eval_metrics, loss = self._create_evaluators(
        dims_to_reduce=('ensemble', 'batch', 'k', 'j')
    )
    # Using nested evaluators for nested inputs.
    fixed_point_eval_metrics, fixed_point_loss = self._create_evaluators(
        ('ensemble', 'batch')
    )
    eval_metrics = evaluators.NestedEvaluators(
        evaluators={
            'slow': eval_metrics,
            'fast': eval_metrics,
            'fixed_point': fixed_point_eval_metrics,
        }
    )
    loss = evaluators.NestedEvaluators(
        evaluators={'slow': loss, 'fast': loss, 'fixed_point': fixed_point_loss}
    )

    optimizer = optax.adam(1e-3)
    opt_config = trainer.OptimizationConfig(optimizer, ema_num_steps=10)

    # Create train stage with nested data specs
    train_stages = []
    # 2 slow steps = 5 fast steps (0, 1, 2, 3, 4) if dt_slow=4*dt_fast
    slow_steps = 2
    dt_slow = model.timestep * 4
    dt_fast = model.timestep

    fast_steps = (slow_steps - 1) * 4 + 1
    timedelta_slow = coordinates.TimeDelta(np.arange(slow_steps) * dt_slow)
    timedelta_fast = coordinates.TimeDelta(np.arange(fast_steps) * dt_fast)

    # inputs_spec must match model structure
    inputs_spec = {
        'slow': _construct_spec(model.inputs_spec['slow'], timedelta_slow),
        'fast': _construct_spec(model.inputs_spec['fast'], timedelta_fast),
    }
    # Dynamic inputs are empty for this model, so we don't need to set them.
    dyn_spec = {}

    train_stages.append(
        trainer.TrainStage(
            duration=5,
            inputs_spec=inputs_spec,
            dynamic_inputs_spec=dyn_spec,
            queries_spec=queries_specs,
            loss=loss,
            time_sample_offset=dt_slow,
            batch_size_per_device=1,
            shuffle_buffer_size=0,
            train_time_slice=None,
            eval_time_slice=None,
        )
    )
    train_schedule = trainer.TrainSchedule(stages=train_stages)

    eval_slow_steps = 4
    eval_fast_steps = (eval_slow_steps - 1) * 4 + 1
    eval_td_slow = coordinates.TimeDelta(np.arange(eval_slow_steps) * dt_slow)
    eval_td_fast = coordinates.TimeDelta(np.arange(eval_fast_steps) * dt_fast)
    eval_inputs_spec = {
        'slow': _construct_spec(model.inputs_spec['slow'], eval_td_slow),
        'fast': _construct_spec(model.inputs_spec['fast'], eval_td_fast),
    }
    eval_dyn_spec = {}
    eval_schedule = trainer.EvalSchedule(
        stages=[
            trainer.EvalSchema(
                cadence=2,
                inputs_spec=eval_inputs_spec,
                dynamic_inputs_spec=eval_dyn_spec,
                queries_spec=queries_specs,
                metrics_evaluator=eval_metrics,
                loss_evaluator=loss,
                time_sample_offset=dt_slow,
                batch_size_per_device=1,
                num_batches=1,
                train_time_slice=None,
                eval_time_slice=None,
                name='eval',
            )
        ]
    )

    checkpoint_config = trainer.CheckpointConfig(
        save_interval_steps=2,
        keep_every_n_steps=10,
        model_config_str='{}',
        metadata={},
    )

    auto_restart = trainer.AutoRestartConfig()
    remat = trainer.RematConfig()
    process_obs = transforms.Identity()
    metrics_saver = InMemorySaver()
    rollout_trainer = trainer.RolloutTrainer(
        experiment_dir=self.test_dir,
        model=model,
        data_loader=data_loader,
        process_observations=process_obs,
        calibration_modules=None,
        train_schedule=train_schedule,
        eval_schedule=eval_schedule,
        optimization_config=opt_config,
        initial_checkpoint=None,
        checkpoint_config=checkpoint_config,
        auto_restart_config=auto_restart,
        remat_config=remat,
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=metrics_saver,
    )
    rollout_trainer.run_training()

    self.assertTrue(metrics_saver.metrics)
    self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'checkpoints')))

  def _create_single_stage_trainer(self, model, all_data, queries_spec, name):
    """Returns a trainer with a single short train stage and no evaluation."""
    _, loss = self._create_evaluators(dims_to_reduce=('ensemble', 'batch', 'k'))
    dt_slow = model.timestep * 4
    timedelta = coordinates.TimeDelta(np.arange(2) * dt_slow)
    stage_kwargs = dict(
        inputs_spec=_construct_spec(model.inputs_spec, timedelta),
        dynamic_inputs_spec=_construct_spec(
            model.dynamic_inputs_spec, timedelta
        ),
        queries_spec=queries_spec,
        time_sample_offset=dt_slow,
        batch_size_per_device=1,
        train_time_slice=None,
        eval_time_slice=None,
    )
    train_stage = trainer.TrainStage(
        duration=1, loss=evaluators.FlattenedEvaluator(loss), **stage_kwargs
    )
    eval_schema = trainer.EvalSchema(
        cadence=1,
        metrics_evaluator=None,
        num_batches=0,
        name='eval',
        **stage_kwargs,
    )
    return trainer.RolloutTrainer(
        experiment_dir=os.path.join(self.test_dir, name),
        model=model,
        data_loader=self._create_data_loader(all_data),
        process_observations=transforms.Identity(),
        calibration_modules=None,
        train_schedule=trainer.TrainSchedule(stages=[train_stage]),
        eval_schedule=trainer.EvalSchedule(stages=[eval_schema]),
        optimization_config=trainer.OptimizationConfig(
            optax.adam(1e-3), ema_num_steps=10
        ),
        initial_checkpoint=None,
        checkpoint_config=trainer.CheckpointConfig(
            save_interval_steps=2,
            keep_every_n_steps=10,
            model_config_str='{}',
            metadata={},
        ),
        auto_restart_config=trainer.AutoRestartConfig(),
        remat_config=trainer.RematConfig(),
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=InMemorySaver(),
    )

  def test_auxiliary_field_in_query_does_not_change_loss(self):
    model, all_data = self.create_model_and_data(multiscale=False)
    base_queries = self._create_queries_specs(model)
    k = cx.LabeledAxis('k', np.arange(8))
    field_in_query = data_specs.FieldInQuerySpec(cx.Scalar())
    optional_time = data_specs.OptionalSpec(field_in_query)
    optional_x_prior = data_specs.OptionalSpec(data_specs.FieldInQuerySpec(k))
    extra_entries = {
        'field_in_query': {'time': field_in_query},
        'auxiliary_field_in_query': {'time': typing.Auxiliary(field_in_query)},
        # 'time' is loaded, so it is taken from data.
        'optional_present': {'time': optional_time},
        # 'x_prior' is not loaded, so it is requested from the model, which
        # ignores it, since it is Auxiliary after rewrapping.
        'optional_missing': {'x_prior': optional_x_prior},
        'auxiliary_coordinate': {'x_request': typing.Auxiliary(k)},
    }
    queries_specs = {'base': base_queries} | {
        name: {'slow': base_queries['slow'] | entries}
        for name, entries in extra_entries.items()
    }
    losses = {}
    for name, queries_spec in queries_specs.items():
      rollout_trainer = self._create_single_stage_trainer(
          model, all_data, queries_spec, name
      )
      _, _, experiment_state = rollout_trainer.initialize_experiment()
      train_iter, train_step_fn, _ = (
          rollout_trainer.build_train_and_eval_iterators(0)
      )
      inputs, dynamic_data = next(train_iter)
      _, loss = train_step_fn(experiment_state, 0, inputs, dynamic_data)
      losses[name] = float(loss)

    for name in extra_entries:
      with self.subTest(name):
        np.testing.assert_allclose(losses[name], losses['base'], rtol=1e-6)

  @parameterized.named_parameters(
      dict(testcase_name='wrap', wrap=True),
      dict(testcase_name='no_wrap', wrap=False),
  )
  def test_standardize_queries_spec(self, wrap: bool):
    k = cx.LabeledAxis('k', np.arange(8))
    td = coordinates.TimeDelta(np.arange(1, 3) * np.timedelta64(1, 'h'))
    field_in_query = data_specs.FieldInQuerySpec(cx.Scalar())
    auxiliary_field_in_query = typing.Auxiliary(data_specs.FieldInQuerySpec(k))
    optional_field_in_query = data_specs.OptionalSpec(
        data_specs.FieldInQuerySpec(k)
    )
    queries_spec = {
        'slow': {
            'x': k,
            'time': field_in_query,
            'location': auxiliary_field_in_query,
            'x_present': optional_field_in_query,
            'x_missing': typing.Auxiliary(optional_field_in_query),
            'x_request': typing.Auxiliary(k),
        }
    }
    k_td = cx.coords.compose(td, k)
    inputs_spec = {
        'slow': {'x': k_td, 'time': td, 'location': k_td, 'x_present': k_td}
    }
    actual = trainer._standardize_queries_spec(
        queries_spec, inputs_spec, wrap_field_in_query_as_auxiliary=wrap
    )
    maybe_auxiliary = typing.Auxiliary if wrap else lambda x: x
    expected = {
        'slow': {
            'x': k,
            'time': maybe_auxiliary(field_in_query),
            'location': auxiliary_field_in_query,
            'x_present': maybe_auxiliary(data_specs.FieldInQuerySpec(k)),
            'x_missing': typing.Auxiliary(k),
            'x_request': typing.Auxiliary(k),
        }
    }
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='field',
          entry=cx.field(np.zeros(8), cx.LabeledAxis('k', np.arange(8))),
          message='Concrete Field',
      ),
      dict(
          testcase_name='missing_target',
          entry=cx.LabeledAxis('k', np.arange(8)),
          message='not in inputs_spec',
      ),
      dict(
          testcase_name='missing_field_in_query',
          entry=typing.Auxiliary(data_specs.FieldInQuerySpec(cx.Scalar())),
          message='not in inputs_spec',
      ),
      dict(
          testcase_name='missing_optional_output',
          entry=data_specs.OptionalSpec(
              data_specs.FieldInQuerySpec(cx.Scalar())
          ),
          message='scored without data',
          wrap=False,
      ),
  )
  def test_standardize_queries_spec_raises(self, entry, message, wrap=True):
    with self.assertRaisesRegex(ValueError, message):
      trainer._standardize_queries_spec(
          {'slow': {'v': entry}},
          {'slow': {}},
          wrap_field_in_query_as_auxiliary=wrap,
      )

  def test_construct_full_queries_spec(self):
    k = cx.LabeledAxis('k', np.arange(8))
    td = coordinates.TimeDelta(np.arange(1, 3) * np.timedelta64(1, 'h'))
    inputs_spec = {'slow': {'x': cx.coords.compose(td, k), 'time': td}}
    auxiliary_time = typing.Auxiliary(data_specs.FieldInQuerySpec(cx.Scalar()))
    queries_spec = {'slow': {'x': k, 'time': auxiliary_time}}
    actual = trainer._construct_full_queries_spec(inputs_spec, queries_spec)
    expected = {
        'slow': {
            'x': cx.coords.compose(td, k),
            'time': typing.Auxiliary(data_specs.FieldInQuerySpec(td)),
        }
    }
    self.assertEqual(actual, expected)

    with self.subTest('inconsistent_auxiliary_entry_raises'):
      j = cx.LabeledAxis('j', np.arange(4))
      bad_spec = {
          'slow': {'x': typing.Auxiliary(data_specs.FieldInQuerySpec(j))}
      }
      with self.assertRaisesRegex(ValueError, 'inconsistent coordinates'):
        trainer._construct_full_queries_spec(inputs_spec, bad_spec)

    with self.subTest('auxiliary_coordinate_uses_timedelta_of_dataset'):
      j = cx.LabeledAxis('j', np.arange(4))
      request = typing.Auxiliary(data_specs.CoordSpec(j))
      request_spec = {'slow': {'x': k, 'request': request}}
      actual = trainer._construct_full_queries_spec(inputs_spec, request_spec)
      self.assertEqual(
          actual['slow']['request'],
          typing.Auxiliary(cx.coords.compose(td, j)),
      )

    with self.subTest('auxiliary_coordinate_without_data_raises'):
      request_only = {'fast': {'request': typing.Auxiliary(k)}}
      with self.assertRaisesRegex(ValueError, 'dataset has none') as cm:
        trainer._construct_full_queries_spec(inputs_spec, request_only)
      self.assertIn('fast/request', ' '.join(cm.exception.__notes__))

  def test_create_initial_evaluators(self):
    k = cx.LabeledAxis('k', np.arange(8))
    td = coordinates.TimeDelta(np.arange(1, 5) * np.timedelta64(1, 'h'))
    k_td = cx.coords.compose(td, k)
    target_spec = ({'data': {'x': k_td}},)

    masked_t0_scaler = scaling.CoordinateMaskScaler(
        mask_coord=coordinates.TimeDelta([np.timedelta64(0, 's')]),
        masked_value=0.0,
        unmasked_value=1.0,
    )
    leadtime_scaler = scaling.LeadTimeScaler(base_squared_error_in_hours=24.0)
    aggregator = aggregation.Aggregator(
        dims_to_reduce=('k',),
        scale_by=[masked_t0_scaler, leadtime_scaler],
    )
    mse = deterministic_metrics.MSE()
    evaluator = evaluators.Evaluator(
        metrics={'mse': mse}, aggregators={'mse': aggregator}
    )

    nested_evaluators = trainer.create_nested_evaluators(
        evaluator, target_spec, np.timedelta64(1, 'h')
    )
    (initial_evaluator,) = trainer.create_initial_evaluators_from_nested(
        nested_evaluators
    )
    agg = initial_evaluator.aggregators['mse']
    cx.testing.assert_fields_equal(
        agg.context['timedelta'], cx.field(np.timedelta64(0, 's'))
    )
    cx.testing.assert_fields_equal(
        agg.context['times'], td.fields['timedelta']
    )

    pred = {'x': cx.field(jnp.ones(8), k)}
    target = {'x': cx.field(jnp.zeros(8), k)}
    # masked_t0_scaler sets scale to 0 at t=0, so agg_state should have 0.
    result = initial_evaluator.evaluate(pred, target)
    mse_val = result['mse'].metric_values(mse)
    cx.testing.assert_fields_allclose(mse_val['x'], cx.field(0.0))

  def _create_coupled_model_trainer(
      self,
      model: TestLorenz96,
      data_loader: data_loading.DataLoader,
      experiment_dir: str,
      *,
      freeze_damping: bool = False,
      duration: int = 2,
      initial_checkpoint: trainer.InitialCheckpoint | None = None,
      trainable_parameters_filter: nnx.filterlib.Filter | None = None,
  ) -> trainer.RolloutTrainer:
    """Returns a trainer for a model created by `_make_coupled_model`."""
    queries_specs = self._create_queries_specs(model)
    eval_metrics, loss = self._create_evaluators(
        dims_to_reduce=('ensemble', 'batch', 'k')
    )
    eval_metrics = evaluators.FlattenedEvaluator(eval_metrics)
    loss = evaluators.FlattenedEvaluator(loss)
    dt_slow = model.timestep * 4
    timedelta = coordinates.TimeDelta(np.arange(2) * dt_slow)
    inputs_spec = _construct_spec(model.inputs_spec, timedelta)
    dyn_spec = _construct_spec(model.dynamic_inputs_spec, timedelta)
    train_stage = trainer.TrainStage(
        duration=duration,
        inputs_spec=inputs_spec,
        dynamic_inputs_spec=dyn_spec,
        queries_spec=queries_specs,
        loss=loss,
        time_sample_offset=dt_slow,
        batch_size_per_device=1,
        shuffle_buffer_size=0,
        train_time_slice=None,
        eval_time_slice=None,
    )
    eval_schedule = trainer.EvalSchedule(
        stages=[
            trainer.EvalSchema(
                cadence=2,
                inputs_spec=inputs_spec,
                dynamic_inputs_spec=dyn_spec,
                queries_spec=queries_specs,
                metrics_evaluator=eval_metrics,
                loss_evaluator=loss,
                time_sample_offset=dt_slow,
                batch_size_per_device=1,
                num_batches=1,
                train_time_slice=None,
                eval_time_slice=None,
                name='eval',
            )
        ]
    )
    calibrators = None
    if freeze_damping:
      calibrators = [
          model_calibrators.SetVariableTags(
              component_key='damping', tag='non_trainable'
          )
      ]
    if trainable_parameters_filter is None:
      # Pass a negated tag filter without wrapping in nnx.All(nnx.Param, ...)
      # to verify RolloutTrainer still restricts trainable parameters to
      # nnx.Param. Without tagged variables, all params are trainable.
      trainable_parameters_filter = nnx.Not(nnx.WithTag('non_trainable'))
    return trainer.RolloutTrainer(
        experiment_dir=experiment_dir,
        model=model,
        data_loader=data_loader,
        process_observations=transforms.Identity(),
        calibration_modules=calibrators,
        train_schedule=trainer.TrainSchedule(stages=[train_stage]),
        eval_schedule=eval_schedule,
        optimization_config=trainer.OptimizationConfig(
            optax.adam(1e-1), ema_num_steps=10
        ),
        initial_checkpoint=initial_checkpoint,
        checkpoint_config=trainer.CheckpointConfig(
            save_interval_steps=2,
            keep_every_n_steps=10,
            model_config_str='{}',
        ),
        auto_restart_config=trainer.AutoRestartConfig(),
        remat_config=trainer.RematConfig(),
        ensemble_axis=cx.SizedAxis('ensemble', 2),
        online_metrics_saver=InMemorySaver(),
        trainable_parameters_filter=trainable_parameters_filter,
    )

  def test_trainer_with_trainable_parameters_filter(self):
    _, all_data = self.create_model_and_data(multiscale=False)
    data_loader = self._create_data_loader(all_data)
    model = _make_coupled_model()
    rollout_trainer = self._create_coupled_model_trainer(
        model, data_loader, self.test_dir, freeze_damping=True
    )
    rollout_trainer.run_training()

    forcing_after_stage1 = float(model.forcing.get_value())
    self.assertNotAlmostEqual(forcing_after_stage1, 8.0)
    self.assertAlmostEqual(float(model.damping.rate.get_value()), 0.5)

    # Verify saved checkpoint stores distinct ema_params and params, and
    # includes full model parameters (both trainable and non-trainable).
    restore_args = rollout_trainer.get_checkpoint_state(restore=True)
    ckpt = rollout_trainer.checkpoint_manager.restore(2, args=restore_args)
    saved_forcing = float(ckpt.params['forcing'].get_value())
    saved_ema_forcing = float(ckpt.ema_params['forcing'].get_value())
    self.assertAlmostEqual(saved_forcing, forcing_after_stage1)
    self.assertNotAlmostEqual(saved_ema_forcing, saved_forcing)
    self.assertAlmostEqual(
        float(ckpt.params['damping']['rate'].get_value()), 0.5
    )
    self.assertAlmostEqual(
        float(ckpt.ema_params['damping']['rate'].get_value()), 0.5
    )

    # Verify resuming from checkpoint preserves tags and only updates trainable
    # parameters.
    resumed_model = _make_coupled_model()
    resumed_trainer = self._create_coupled_model_trainer(
        resumed_model,
        data_loader,
        self.test_dir,
        freeze_damping=True,
        duration=4,
    )
    resumed_trainer.run_training()

    forcing_after_resume = float(resumed_model.forcing.get_value())
    self.assertNotAlmostEqual(forcing_after_resume, forcing_after_stage1)
    self.assertAlmostEqual(float(resumed_model.damping.rate.get_value()), 0.5)

  def test_trainer_raises_if_no_parameters_are_trainable(self):
    _, all_data = self.create_model_and_data(multiscale=False)
    rollout_trainer = self._create_coupled_model_trainer(
        _make_coupled_model(),
        self._create_data_loader(all_data),
        self.test_dir,
        trainable_parameters_filter=nnx.WithTag('missing_tag'),
    )
    with self.assertRaisesRegex(ValueError, 'No trainable parameters matched'):
      rollout_trainer.run_training()

  def test_new_trainer_restores_checkpoint_with_frozen_parameters(self):
    _, all_data = self.create_model_and_data(multiscale=False)
    data_loader = self._create_data_loader(all_data)
    self._create_coupled_model_trainer(
        _make_coupled_model(), data_loader, self.test_dir, freeze_damping=True
    ).run_training()

    # Tags are set when the trainer is constructed, hence checkpoints with
    # frozen parameters can be restored without initializing the experiment.
    model = _make_coupled_model()
    new_trainer = self._create_coupled_model_trainer(
        model, data_loader, self.test_dir, freeze_damping=True
    )
    self.assertEqual(model.damping.rate.tag, 'non_trainable')
    restore_args = new_trainer.get_checkpoint_state(restore=True)
    ckpt = new_trainer.checkpoint_manager.restore(2, args=restore_args)
    self.assertAlmostEqual(
        float(ckpt.params['damping']['rate'].get_value()), 0.5
    )

  def test_warm_start_with_newly_frozen_parameters(self):
    _, all_data = self.create_model_and_data(multiscale=False)
    data_loader = self._create_data_loader(all_data)
    source_dir = self.create_tempdir().full_path
    source_trainer = self._create_coupled_model_trainer(
        _make_coupled_model(), data_loader, source_dir
    )
    source_trainer.run_training()
    source_ema_params = source_trainer.checkpoint_manager.restore(
        2, args=source_trainer.get_checkpoint_state(restore=True)
    ).ema_params
    source_damping = float(source_ema_params['damping']['rate'].get_value())
    source_forcing = float(source_ema_params['forcing'].get_value())
    self.assertNotAlmostEqual(source_damping, 0.5)  # damping was trained.
    checkpoint_dir = os.path.join(source_dir, 'checkpoints')

    with self.subTest('restoring_optimizer_state_raises'):
      warm_start_trainer = self._create_coupled_model_trainer(
          _make_coupled_model(),
          data_loader,
          self.create_tempdir().full_path,
          freeze_damping=True,
          initial_checkpoint=trainer.InitialCheckpoint(
              checkpoint_dir=checkpoint_dir,
              checkpoint_step=2,
              reset_optimizer_state=False,
          ),
      )
      with self.assertRaisesRegex(ValueError, 'reset_optimizer_state=True'):
        warm_start_trainer.run_training()

    with self.subTest('resetting_optimizer_state_keeps_frozen_parameters'):
      model = _make_coupled_model()
      self._create_coupled_model_trainer(
          model,
          data_loader,
          self.create_tempdir().full_path,
          freeze_damping=True,
          initial_checkpoint=trainer.InitialCheckpoint(
              checkpoint_dir=checkpoint_dir,
              checkpoint_step=2,
              reset_optimizer_state=True,
          ),
      ).run_training()
      self.assertAlmostEqual(
          float(model.damping.rate.get_value()), source_damping
      )
      self.assertNotAlmostEqual(
          float(model.forcing.get_value()), source_forcing
      )

  def test_ensure_replicated(self):
    mesh = jax.sharding.Mesh(jax.devices()[:1], ('batch',))
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )

    with self.subTest('replicates_unsharded_leaves'):
      state = {'w': nnx.Param(jnp.ones((2, 2)))}
      replicated = train_utils.ensure_replicated(state, mesh=mesh)
      self.assertEqual(
          replicated['w'].get_value().sharding, replicated_sharding
      )

    with self.subTest('raises_on_sharded_array'):
      sharded_arr = jax.device_put(
          jnp.ones((2, 2)),
          jax.sharding.NamedSharding(
              mesh, jax.sharding.PartitionSpec('batch', None)
          ),
      )
      with self.assertRaisesRegex(
          ValueError, 'Expected unsharded or replicated sharding'
      ):
        train_utils.ensure_replicated({'w': sharded_arr}, mesh=mesh)

    with self.subTest('raises_on_mismatched_named_sharding'):
      other_mesh = jax.sharding.Mesh(jax.devices()[:1], ('other_axis',))
      other_sharding = jax.sharding.NamedSharding(
          other_mesh, jax.sharding.PartitionSpec()
      )
      mismatched_arr = jax.device_put(jnp.ones((2, 2)), other_sharding)
      with self.assertRaisesRegex(
          ValueError, 'Expected unsharded or replicated sharding'
      ):
        train_utils.ensure_replicated({'w': mismatched_arr}, mesh=mesh)


if __name__ == '__main__':
  absltest.main()
