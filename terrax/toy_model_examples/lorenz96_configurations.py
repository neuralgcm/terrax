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

"""Fiddle configurations of the Lorenz-96 models used in the tutorials.

Model builders are `auto_config` functions: calling `parameterized_model(...)`
returns a model, while `parameterized_model.as_buildable(...)` returns a
`fdl.Config` that can be serialized, modified and built with
`api.Model.from_fiddle_config`. All builders share the system parameters
(`k_size`, `forcing`, `dt`), so variants of the system are one argument away.

The module also keeps a small registry of model checkpoints used across the
tutorials, see `load_checkpoint`.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
import functools
import os

import coordax as cx
import fiddle as fdl
from fiddle.experimental import auto_config
from flax import nnx
from terrax.core import api
from terrax.core import checkpointing
from terrax.core import feature_transforms
from terrax.core import learned_transforms
from terrax.core import observation_operators
from terrax.core import standard_layers
from terrax.core import towers
from terrax.core import transforms
from terrax.toy_model_examples import lorenz96

# Two-scale system of Lorenz (1996) with `K` slow and `K x J` fast variables.
# The forcing follows Wilks (2005), which puts the system in a strongly chaotic
# regime where errors saturate within about 10 days.
K_SIZE = 36
J_SIZE = 10
FORCING = 20.0
C, B, H = 10.0, 10.0, 1.0
# Time steps in model time units (1 MTU is interpreted as 5 days):
DT_FAST = 0.0025  # 18 min, integrator step of the two-scale truth.
DT = 0.05  # 6 hours, step of the single-scale models (20 truth steps).

# Noise structure options for stochastic model components, see
# `lorenz96.CorrelatedNoise`. `None` stands for a deterministic component.
NOISE_PRESETS: dict[str, dict[str, float]] = {
    'white': dict(correlation_steps=0.0, correlation_length=0.0),
    'temporal': dict(correlation_steps=4.0, correlation_length=0.0),
    'spatiotemporal': dict(correlation_steps=4.0, correlation_length=2.0),
}

# Noise fields of `pointwise_y_observation_operator`: arguments of
# `lorenz96.CorrelatedNoise` for noise that is correlated along `j`. The waves
# of `y` are about 5 and 2.5 points long (harmonics 2 and 4 of a ring of
# `J_SIZE` points). Memory and drift of the wave-like fields are set from the
# lag-1 autocorrelation of these harmonics for `y` sampled every `DT` in the
# default two-scale system, see the tutorial on correlation-aware scores; they
# cannot be trained with a loss that scores one lead time at a time. Spatial
# scales are deliberately only in the ballpark, training is meant to find them.
NOISE_FIELD_PRESETS: dict[str, dict[str, float]] = {
    'smooth': dict(correlation_length=1.5),
    'long_waves': dict(
        correlation_length=1.5, wavelength=7.0, correlation_steps=1.6, drift=1.8
    ),
    'short_waves': dict(
        correlation_length=1.5,
        wavelength=3.0,
        correlation_steps=0.8,
        drift=-0.4,
    ),
}

_auto_config = functools.partial(
    auto_config.auto_config,
    experimental_allow_control_flow=True,
    experimental_allow_dataclass_attribute_access=True,
)


@_auto_config
def two_scale_truth(
    k_size: int = K_SIZE,
    j_size: int = J_SIZE,
    forcing: float = FORCING,
    c: float = C,
    b: float = B,
    h: float = H,
    dt: float = DT_FAST,
) -> lorenz96.Lorenz96WithTwoScales:
  """Two-scale Lorenz-96 system used as the ground truth."""
  return lorenz96.Lorenz96WithTwoScales(
      k_axis=lorenz96.ring_axis(k_size),
      j_axis=lorenz96.ring_axis(j_size, 'j'),
      forcing=forcing,
      c=c,
      b=b,
      h=h,
      dt=dt,
  )


@_auto_config
def one_level_model(
    k_size: int = K_SIZE,
    forcing: float = FORCING,
    dt: float = DT,
) -> lorenz96.Lorenz96:
  """Single-scale Lorenz-96 model without any parameterization."""
  return lorenz96.Lorenz96(
      k_axis=lorenz96.ring_axis(k_size), forcing=forcing, dt=dt
  )


@_auto_config
def _mlp_transform(
    k_axis: cx.Coordinate,
    targets: dict[str, cx.Coordinate],
    inputs_transform: transforms.Transform,
    *,
    hidden_size: int,
    hidden_layers: int,
    out_transform: transforms.Transform | None,
    rngs: nnx.Rngs,
) -> learned_transforms.ForwardTowerTransform:
  """MLP applied at every point to features of `x` from `inputs_transform`."""
  mlp = functools.partial(
      standard_layers.MlpUniform,
      hidden_size=hidden_size,
      n_hidden_layers=hidden_layers,
  )
  tower = functools.partial(
      towers.ForwardTower.build_using_factories,
      inputs_in_dims=('c',),
      out_dims=('c',),
      neural_net_factory=mlp,
  )
  return learned_transforms.ForwardTowerTransform.build_using_factories(
      input_shapes={'x': cx.shape_struct_field(k_axis)},
      target_split_axes=targets,
      tower_factory=tower,
      concat_dims=('feature',),
      inputs_transform=inputs_transform,
      out_transform=out_transform,
      rngs=rngs,
  )


@_auto_config
def local_mlp_transform(
    k_axis: cx.Coordinate,
    targets: dict[str, cx.Coordinate],
    *,
    hidden_size: int = 64,
    hidden_layers: int = 2,
    stencil: tuple[int, ...] = (-2, -1, 0, 1, 2),
    noise: str | None = 'white',
    noise_channels: int = 1,
    out_transform: transforms.Transform | None = None,
    seed: int = 0,
) -> learned_transforms.ForwardTowerTransform:
  """Pointwise MLP acting on a stencil of `x` and optional noise channels.

  Args:
    k_axis: Periodic axis of the slow variables.
    targets: Output names mapped to the axis each output is split into, e.g.
      `{'x_correction': cx.Scalar()}` for a scalar output per grid point.
    hidden_size: Width of the MLP hidden layers.
    hidden_layers: Number of MLP hidden layers.
    stencil: Offsets of the `x` values seen at each grid point.
    noise: Name of a `NOISE_PRESETS` entry or `None` for no noise inputs.
    noise_channels: Number of independent noise inputs per grid point.
    out_transform: Optional transform applied to the MLP outputs.
    seed: Seed for parameter initialization and noise.

  Returns:
    A transform mapping prognostics `{'x': ...}` to the fields in `targets`.
  """
  rngs = nnx.Rngs(seed)
  feature_axis = cx.DummyAxis('feature', 1)
  features: dict[str, transforms.Transform] = {
      'stencil': transforms.Sequential([
          lorenz96.StencilFeatures(k_axis, stencil),
          transforms.ExpandDims(feature_axis, loc=k_axis),
      ]),
  }
  if noise is not None:
    noise_coord = cx.coords.compose(
        cx.DummyAxis('feature', noise_channels), k_axis
    )
    preset = NOISE_PRESETS[noise]
    process = lorenz96.CorrelatedNoise(
        noise_coord,
        correlation_steps=preset['correlation_steps'],
        correlation_length=preset['correlation_length'],
        rngs=rngs,
    )
    features['noise'] = feature_transforms.RandomnessFeatures(
        process, coord=noise_coord
    )
  return _mlp_transform(
      k_axis,
      targets,
      transforms.Merge(features),
      hidden_size=hidden_size,
      hidden_layers=hidden_layers,
      out_transform=out_transform,
      rngs=rngs,
  )


def order_as(field: cx.Field, axes: tuple[str, ...]) -> cx.Field:
  """Returns `field` with named `axes` last, in the given order."""
  return field.order_as(..., *axes)


@_auto_config
def y_observation_operator(
    k_axis: cx.Coordinate,
    j_size: int = J_SIZE,
    *,
    hidden_size: int = 64,
    noise: str | None = 'white',
    noise_channels: int = 4,
    output_scale: float = 0.1,
    seed: int = 1,
) -> observation_operators.TransformObservationOperator:
  """Observation operator predicting fast variables `y` from the slow `x`.

  Args:
    k_axis: Periodic axis of the slow variables.
    j_size: Number of fast variables per slow variable.
    hidden_size: Width of the MLP hidden layers.
    noise: Name of a `NOISE_PRESETS` entry or `None` for a deterministic
      operator.
    noise_channels: Number of independent noise inputs per grid point.
    output_scale: Factor applied to the MLP outputs, matching the typical
      magnitude of `y`.
    seed: Seed for parameter initialization and noise.

  Returns:
    Observation operator answering queries for `y` on `(k, j)` and `time`.
  """
  j_axis = lorenz96.ring_axis(j_size, 'j')
  # The MLP returns `y` on `(j, k)`; order and scale it to match the data.
  out_transform = transforms.Sequential([
      transforms.ApplyFnToKeys(
          functools.partial(order_as, axes=('k', 'j')), ['y']
      ),
      transforms.ScaleFields(cx.field(output_scale)),
  ])
  y_transform = local_mlp_transform(
      k_axis,
      {'y': j_axis},
      hidden_size=hidden_size,
      noise=noise,
      noise_channels=noise_channels,
      out_transform=out_transform,
      seed=seed,
  )
  with_time = transforms.Merge(
      {'y': y_transform, 'time': transforms.SelectKeys('time')}
  )
  return observation_operators.TransformObservationOperator(with_time)


@_auto_config
def pointwise_y_observation_operator(
    k_axis: cx.Coordinate,
    j_size: int = J_SIZE,
    *,
    hidden_size: int = 64,
    noise_fields: Sequence[str] = ('smooth', 'long_waves', 'short_waves'),
    learnable_noise: bool = True,
    output_scale: float = 0.1,
    seed: int = 1,
) -> observation_operators.TransformObservationOperator:
  """Operator predicting `y` from `x` and noise fields correlated along `j`.

  Unlike `y_observation_operator`, whose MLP returns all `y` of a slow variable
  at once, here an MLP is applied at every point of the `(k, j)` grid. It sees
  the `x` stencil around `k` and the value of every noise field at this point.
  The structure of `y` along `j` hence comes from the noise fields, whose
  scales can be trained.

  Args:
    k_axis: Periodic axis of the slow variables.
    j_size: Number of fast variables per slow variable.
    hidden_size: Width of the MLP hidden layers.
    noise_fields: Names of `NOISE_FIELD_PRESETS` entries; the MLP sees one
      `lorenz96.CorrelatedNoise` field for each of them.
    learnable_noise: Whether the correlation lengths and wavelengths of the
      noise fields are trained, starting from their preset values. Memory and
      drift of the noise are not trained: a loss evaluated separately at each
      lead time does not depend on them.
    output_scale: Factor applied to the MLP outputs, matching the typical
      magnitude of `y`.
    seed: Seed for parameter initialization and noise.

  Returns:
    Observation operator answering queries for `y` on `(k, j)` and `time`.
  """
  if isinstance(noise_fields, str):
    raise TypeError(
        'noise_fields must be a sequence of `NOISE_FIELD_PRESETS` names, got'
        f' the string {noise_fields!r}.'
    )
  if not noise_fields:
    raise ValueError('noise_fields must name at least one noise field.')
  trainable: tuple[lorenz96.NoiseParameter, ...] = ()
  if learnable_noise:
    trainable = ('correlation_length', 'wavelength')
  j_axis = lorenz96.ring_axis(j_size, 'j')
  rngs = nnx.Rngs(seed)
  feature_axis = cx.DummyAxis('feature', 1)
  noise_coord = cx.coords.compose(feature_axis, k_axis, j_axis)
  features: dict[str, transforms.Transform] = {
      'stencil': transforms.Sequential([
          lorenz96.StencilFeatures(k_axis),
          transforms.ExpandDims(feature_axis, loc=k_axis),
      ]),
  }
  for i, name in enumerate(noise_fields):
    preset = NOISE_FIELD_PRESETS[name]
    process = lorenz96.CorrelatedNoise(
        noise_coord,
        dim='j',
        learnable=tuple(p for p in trainable if p in preset),
        rngs=rngs,
        **preset,
    )
    features[f'noise_{i}'] = feature_transforms.RandomnessFeatures(
        process, coord=noise_coord, feature_name=name
    )
  # Stencil features do not depend on `j` and are broadcast to the noise fields.
  inputs_transform = transforms.Sequential(
      [transforms.Merge(features), transforms.Broadcast()]
  )
  y_transform = _mlp_transform(
      k_axis,
      {'y': cx.Scalar()},
      inputs_transform,
      hidden_size=hidden_size,
      hidden_layers=2,
      out_transform=transforms.ScaleFields(cx.field(output_scale)),
      rngs=rngs,
  )
  with_time = transforms.Merge(
      {'y': y_transform, 'time': transforms.SelectKeys('time')}
  )
  return observation_operators.TransformObservationOperator(with_time)


@_auto_config
def parameterized_model(
    k_size: int = K_SIZE,
    forcing: float = FORCING,
    dt: float = DT,
    *,
    hidden_size: int = 64,
    noise: str | None = 'white',
    y_operator: bool = False,
    y_noise: str | None = 'white',
    seed: int = 0,
) -> lorenz96.Lorenz96:
  """Single-scale Lorenz-96 model with a learned (stochastic) closure.

  Args:
    k_size: Number of slow variables.
    forcing: Forcing `F` of the Lorenz-96 equations.
    dt: Model time step in model time units.
    hidden_size: Width of the closure MLP.
    noise: Name of a `NOISE_PRESETS` entry for the closure or `None` for a
      deterministic closure.
    y_operator: Whether to include an observation operator for `y`.
    y_noise: Noise preset of the `y` observation operator, if included.
    seed: Seed for parameter initialization and noise.

  Returns:
    A `lorenz96.Lorenz96` model.
  """
  k_axis = lorenz96.ring_axis(k_size)
  closure = local_mlp_transform(
      k_axis,
      {'x_correction': cx.Scalar()},
      hidden_size=hidden_size,
      noise=noise,
      seed=seed,
  )
  operators = {}
  if y_operator:
    operators['fast'] = y_observation_operator(
        k_axis, noise=y_noise, seed=seed + 1
    )
  return lorenz96.Lorenz96(
      k_axis=k_axis,
      forcing=forcing,
      dt=dt,
      parameterizations=[lorenz96.TransformParameterization(closure)],
      operators=operators,
  )


@_auto_config
def learned_model(
    k_size: int = K_SIZE,
    dt: float = DT,
    *,
    hidden_size: int = 64,
    noise: str | None = 'white',
    y_operator: bool = False,
    y_noise: str | None = 'white',
    seed: int = 0,
) -> lorenz96.LearnedLorenz96:
  """Data-driven Lorenz-96 model that learns the full update of `x`.

  Args:
    k_size: Number of slow variables.
    dt: Model time step in model time units.
    hidden_size: Width of the update MLP.
    noise: Name of a `NOISE_PRESETS` entry or `None` for a deterministic update.
    y_operator: Whether to include an observation operator for `y`.
    y_noise: Noise preset of the `y` observation operator, if included.
    seed: Seed for parameter initialization and noise.

  Returns:
    A `lorenz96.LearnedLorenz96` model.
  """
  k_axis = lorenz96.ring_axis(k_size)
  update = local_mlp_transform(
      k_axis,
      {'x_delta': cx.Scalar()},
      hidden_size=hidden_size,
      noise=noise,
      seed=seed,
  )
  operators = {}
  if y_operator:
    operators['fast'] = y_observation_operator(
        k_axis, noise=y_noise, seed=seed + 1
    )
  return lorenz96.LearnedLorenz96(
      k_axis=k_axis, transform=update, dt=dt, operators=operators
  )


def with_y_operator(
    config: fdl.Config[lorenz96.SingleScaleBase],
    *,
    operator_fn: auto_config.AutoConfig = y_observation_operator,
    **kwargs,
) -> fdl.Config[lorenz96.SingleScaleBase]:
  """Returns a copy of a single-scale model `config` with a `y` operator.

  The operator is built for the `k_axis` of `config`, so the resulting model
  is identical to `config` except for the added `operators['fast']`. This is
  used to extend a trained model (e.g. from a checkpoint) with a new head.

  Args:
    config: Fiddle config of a `lorenz96.SingleScaleBase` model.
    operator_fn: Builder of the operator, `y_observation_operator` or
      `pointwise_y_observation_operator`.
    **kwargs: Arguments of `operator_fn` other than `k_axis`.

  Returns:
    A new config with the `y` observation operator under `operators['fast']`.
  """
  config = copy.deepcopy(config)
  operator = operator_fn.as_buildable(config.k_axis, **kwargs)
  # `operators` may be unset in `config`, in which case the model default
  # (an empty dict) applies and cannot be read through Fiddle.
  operators = dict(config.__arguments__.get('operators', {}))
  config.operators = operators | {'fast': operator}
  return config


# Checkpoints saved by the tutorials to a local directory.
LOCAL_CHECKPOINT_DIR = '/tmp/terrax_lorenz96'
LOCAL_CHECKPOINTS: dict[str, str] = {
    'parameterized_crps': os.path.join(
        LOCAL_CHECKPOINT_DIR, 'parameterized_crps'
    ),
}
# TODO(dkochkov): Populate with checkpoints hosted on a public GCS bucket.
PRETRAINED_CHECKPOINTS: dict[str, str] = {}


def checkpoint_names() -> list[str]:
  """Returns names of all registered checkpoints."""
  return sorted(PRETRAINED_CHECKPOINTS | LOCAL_CHECKPOINTS)


def checkpoint_path(name: str) -> str:
  """Returns the path of a registered checkpoint."""
  paths = PRETRAINED_CHECKPOINTS | LOCAL_CHECKPOINTS
  if name not in paths:
    raise KeyError(f'Unknown checkpoint {name!r}, choose from {sorted(paths)}')
  return paths[name]


def load_checkpoint(name: str) -> api.Model:
  """Loads a registered model checkpoint by name."""
  return checkpointing.load_model_checkpoint(checkpoint_path(name))
