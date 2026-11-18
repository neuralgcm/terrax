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

"""Implementation of the Lorenz96 model components."""

from collections.abc import Sequence
import functools
from typing import Callable, Literal, Protocol

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from terrax.core import api
from terrax.core import data_specs
from terrax.core import module_utils
from terrax.core import observation_operators
from terrax.core import random_processes
from terrax.core import time_integrators
from terrax.core import transforms
from terrax.core import typing

IntegratorCls = Callable[[time_integrators.ExplicitODE, float], nnx.Module]


def ring_axis(size: int, name: str = 'k') -> cx.LabeledAxis:
  """Returns a periodic grid axis with points labeled `0, ..., size - 1`."""
  return cx.LabeledAxis(name, np.arange(size))


def lorenz96_x_term(x: jax.Array, f: float) -> jax.Array:
  """Computes uncoupled term for `X` variables in L96 system."""
  x_p1 = jnp.roll(x, -1)
  x_m1 = jnp.roll(x, 1)
  x_m2 = jnp.roll(x, 2)
  return -x_m1 * (x_m2 - x_p1) - x + f


def lorenz96_y_term(y: jax.Array, c: float, b: float) -> jax.Array:
  """Computes uncoupled term for `Y` variables in L96 system."""
  y_p1 = jnp.roll(y, -1)
  y_p2 = jnp.roll(y, -2)
  y_m1 = jnp.roll(y, 1)
  return -(c * b) * y_p1 * (y_p2 - y_m1) - c * y


def advance_random_processes(module: nnx.Module) -> None:
  """Advances all random processes found on `module` by one step."""
  for random_process in module_utils.retrieve_subclass_modules(
      module, random_processes.RandomProcessModule
  ):
    random_process.advance()


@nnx.dataclass
class StencilFeatures(transforms.PytreeTransformABC):
  """Returns periodic shifts of `inputs[key]` along `coord` as features.

  Feature `key+s` holds the value at grid point `k + s`, e.g. for `key='x'` and
  `shifts=(-1, 0, 1)` the outputs are `'x-1': X[k-1]`, `'x': X[k]` and
  `'x+1': X[k+1]`. This is the stencil that a local parameterization sees,
  analogous to the receptive field of a small convolution on a periodic grid.

  Attributes:
    coord: Periodic coordinate along which `inputs[key]` is shifted.
    shifts: Offsets of the stencil relative to the current grid point.
    key: Key of the input field to build the stencil from.
  """

  coord: cx.Coordinate
  shifts: tuple[int, ...] = (-2, -1, 0, 1, 2)
  key: str = 'x'

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    x = inputs[self.key].untag(self.coord)
    features = {}
    for shift in self.shifts:
      roll_fn = cx.cpmap(functools.partial(jnp.roll, shift=-shift))
      name = self.key if shift == 0 else f'{self.key}{shift:+d}'
      features[name] = roll_fn(x).tag(self.coord)
    return features


def _ring_distance(size: int) -> np.ndarray:
  """Returns distances from point 0 to all points on a ring of `size` points."""
  offsets = np.arange(size)
  return np.minimum(offsets, size - offsets)


def _periodic_convolve(
    field: cx.Field, dim: str, kernel: jax.Array
) -> cx.Field:
  """Convolves `field` along periodic `dim` with `kernel`.

  Args:
    field: Field with a named dimension `dim`.
    dim: Periodic dimension to convolve along.
    kernel: Kernel weights indexed by the offset along `dim`.

  Returns:
    Convolved field with the same coordinates as `field`.
  """
  size = field.named_shape[dim]
  axis = field.axes.get(dim, dim)
  kernel_hat = jnp.fft.rfft(kernel)
  convolve = lambda z: jnp.fft.irfft(jnp.fft.rfft(z) * kernel_hat, n=size)
  return cx.cpmap(convolve)(field.untag(axis)).tag(axis)


def _drifting_ar1_step(
    z: jax.Array,
    innovation: jax.Array,
    phi: typing.Numeric,
    drift: typing.Numeric,
) -> jax.Array:
  """Returns `phi * shift(z, drift) + sqrt(1 - phi**2) * innovation` on a ring.

  The shift by `drift` grid points is a phase rotation of the Fourier modes of
  `z`. The Nyquist mode of a real field cannot be rotated and is damped
  instead; its innovation is scaled up accordingly, so that white `z` with unit
  variance keeps this distribution for any `drift`.

  Args:
    z: Values on a ring, white with unit variance.
    innovation: White noise with unit variance, of the same shape as `z`.
    phi: Correlation between the shifted `z` and the returned values.
    drift: Shift along the ring in (possibly fractional) grid points.

  Returns:
    Next values on the ring.
  """
  (size,) = z.shape
  wavenumbers = np.arange(size // 2 + 1)
  is_nyquist = 2 * wavenumbers == size  # no such mode for odd sizes.
  nyquist_gain = jnp.cos(jnp.pi * drift)
  rotation = jnp.exp(-2j * jnp.pi * wavenumbers * drift / size)
  rotation = jnp.where(is_nyquist, nyquist_gain, rotation)
  squared_gain = jnp.where(is_nyquist, nyquist_gain**2, 1.0)  # |rotation|**2.
  innovation_scale = jnp.sqrt(1 - phi**2 * squared_gain)
  z_hat = jnp.fft.rfft(z, norm='ortho')
  innovation_hat = jnp.fft.rfft(innovation, norm='ortho')
  next_z_hat = phi * rotation * z_hat + innovation_scale * innovation_hat
  return jnp.fft.irfft(next_z_hat, n=size, norm='ortho')


# Names of the `CorrelatedNoise` parameters that can be trained.
NoiseParameter = Literal[
    'correlation_steps', 'correlation_length', 'wavelength', 'drift'
]

# softplus(_SOFTPLUS_INVERSE_1) == 1.0, so that learnable parameters start at
# their initial values.
_SOFTPLUS_INVERSE_1 = 0.5413248546129181


class CorrelatedNoise(random_processes.RandomProcessModule):
  """Gaussian noise with optional correlations in time and along a ring.

  The process keeps a latent field `z` on `coord`, white along `dim`, that
  follows an AR(1) recursion with one-step correlation
  `phi = exp(-1 / correlation_steps)`:

    z_{t+1} = phi * shift(z_t) + sqrt(1 - phi**2) * eps_t,    eps_t ~ N(0, 1),

  where `shift` moves the field by `drift` grid points along `dim`, so that
  structures of the noise travel around the ring. `state_values` returns `z`
  convolved along the periodic `dim` with the kernel

    w(d) ~ exp(-(d / correlation_length)**2 / 2) * cos(2 * pi * d / wavelength)

  normalized to unit variance of the result. The Gaussian envelope sets the
  range of correlations and the optional cosine carrier makes the noise
  wave-like, with most of its variance near `wavelength`. With default
  arguments the process is white in time and space, equivalent to
  `random_processes.NormalUncorrelated`.

  Parameters listed in `learnable` become `nnx.Param`s. With the same
  reparameterization as in `random_processes.GaussianRandomField`, a positive
  parameter is `initial_value * softplus(raw + c)`, where `softplus(c) == 1` and
  `raw` is the `nnx.Param`, initialized at zero: training starts from the
  initial value, additive updates of `raw` act as relative adjustments and the
  parameter stays positive. `drift` is `initial_value + raw`.

  Attributes:
    coord: Coordinate on which the process is defined.
    dim: Periodic dimension of `coord` along which the noise is correlated.
    raw_parameters: Raw values of the `learnable` parameters.
  """

  def __init__(
      self,
      coord: cx.Coordinate,
      correlation_steps: float = 0.0,
      correlation_length: float = 0.0,
      dim: str = 'k',
      *,
      wavelength: float | None = None,
      drift: float = 0.0,
      learnable: Sequence[NoiseParameter] = (),
      rngs: nnx.Rngs,
  ):
    """Constructs the process.

    Args:
      coord: Coordinate on which the process is defined.
      correlation_steps: Decorrelation time in steps; `0` means no memory.
      correlation_length: Width of the Gaussian envelope of the kernel in grid
        points; `0` means no spatial correlation.
      dim: Periodic dimension of `coord` along which the noise is correlated.
      wavelength: Wavelength of the cosine carrier of the kernel in grid points;
        `None` means no carrier. Requires `correlation_length > 0`.
      drift: Number of grid points by which the noise travels along `dim` in one
        step. Requires `correlation_steps > 0`.
      learnable: Names of the arguments above that are trained. Parameters that
        are switched off (`0` or `None`) cannot be trained, except for `drift`.
      rngs: Random number generators for the initial state of the process.
    """
    learnable = tuple(learnable)
    if dim not in coord.dims:
      raise ValueError(f'{dim=} is not a dimension of {coord=}.')
    if correlation_steps < 0 or correlation_length < 0:
      raise ValueError(
          'correlation_steps and correlation_length must be non-negative, got'
          f' {correlation_steps=} and {correlation_length=}.'
      )
    if wavelength is not None and wavelength <= 0:
      raise ValueError(f'{wavelength=} must be positive or None.')
    if wavelength is not None and correlation_length == 0:
      raise ValueError(f'{wavelength=} requires a positive correlation_length.')
    if (drift != 0 or 'drift' in learnable) and correlation_steps == 0:
      raise ValueError(f'{drift=} requires positive correlation_steps.')
    initial_values = {
        'correlation_steps': correlation_steps,
        'correlation_length': correlation_length,
        'wavelength': wavelength,
        'drift': drift,
    }
    for name in learnable:
      if name not in initial_values:
        raise ValueError(f'Unknown parameter {name!r} in {learnable=}.')
      if name != 'drift' and not initial_values[name]:
        raise ValueError(f'{name!r} is switched off and cannot be learnable.')
    distribution = random_processes.NormalDistribution()
    sampler = random_processes.DistributionSampler(distribution, coord)
    super().__init__(samplers={'normal': sampler})
    self.coord = coord
    self.dim = dim
    self._correlation_steps = correlation_steps
    self._correlation_length = correlation_length
    self._wavelength = wavelength
    self._drift = drift
    raw_parameters: dict[str, nnx.Param] = {
        name: nnx.Param(jnp.zeros(())) for name in learnable
    }
    self.raw_parameters = nnx.Dict(raw_parameters)
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k, 2)))(rngs.params())
    self.state_rng = random_processes.Randomness(rng)
    self.rng_step = random_processes.Randomness(cx.field(jnp.uint32(0)))
    self.core = random_processes.Randomness(self.sampler.draw(k))

  @property
  def sampler(self) -> random_processes.Sampler:
    return self.samplers['normal']

  def _scaled(self, name: str, initial_value: float) -> typing.Numeric:
    """Returns `initial_value` times the learned scale of `name`, if any."""
    if name not in self.raw_parameters:
      return initial_value
    raw = self.raw_parameters[name][...]
    return initial_value * jax.nn.softplus(raw + _SOFTPLUS_INVERSE_1)

  @property
  def correlation_steps(self) -> typing.Numeric:
    """Decorrelation time in steps."""
    return self._scaled('correlation_steps', self._correlation_steps)

  @property
  def correlation_length(self) -> typing.Numeric:
    """Width of the Gaussian envelope of the kernel in grid points."""
    return self._scaled('correlation_length', self._correlation_length)

  @property
  def wavelength(self) -> typing.Numeric | None:
    """Wavelength of the carrier of the kernel in grid points, if any."""
    if self._wavelength is None:
      return None
    return self._scaled('wavelength', self._wavelength)

  @property
  def drift(self) -> typing.Numeric:
    """Number of grid points by which the noise travels in one step."""
    if 'drift' not in self.raw_parameters:
      return self._drift
    return self._drift + self.raw_parameters['drift'][...]

  @property
  def phi(self) -> typing.Numeric:
    """One-step autocorrelation of the latent AR(1) process."""
    if self._correlation_steps == 0:
      return 0.0
    return jnp.exp(-1.0 / self.correlation_steps)

  def kernel(self) -> jax.Array:
    """Returns kernel weights indexed by the offset along `dim`."""
    distance = _ring_distance(self.coord.sizes[self.dim])
    if self._correlation_length == 0:
      return jnp.asarray(distance == 0, jnp.float32)
    weights = jnp.exp(-0.5 * (distance / self.correlation_length) ** 2)
    wavelength = self.wavelength
    if wavelength is not None:
      weights = weights * jnp.cos(2 * jnp.pi * distance / wavelength)
    return weights / jnp.sqrt(jnp.sum(weights**2))

  def unconditional_sample(self, rng: cx.Field) -> None:
    k, rng = cx.cmap(lambda k: tuple(jax.random.split(k)))(rng)
    self.state_rng.set_value(rng)
    self.rng_step.set_value(
        cx.field(jnp.zeros(k.shape, jnp.uint32), k.coordinate)
    )
    self.core.set_value(self.sampler.draw(k))

  def advance(self) -> None:
    key = cx.cmap(jax.random.fold_in)(
        self.state_rng.get_value(), self.rng_step.get_value()
    )
    innovation = self.sampler.draw(key)
    core = self.core.get_value()
    phi = self.phi
    if self._drift != 0 or 'drift' in self.raw_parameters:
      step_fn = functools.partial(_drifting_ar1_step, phi=phi, drift=self.drift)
      axis = core.axes.get(self.dim, self.dim)
      z = cx.cpmap(step_fn)(core.untag(axis), innovation.untag(axis)).tag(axis)
    else:
      z = core * phi + innovation * jnp.sqrt(1 - phi**2)  # layout of `core`.
    self.core.set_value(z)
    self.rng_step.set_value(self.rng_step.get_value() + 1)

  def state_values(self, coord: cx.Coordinate | None = None) -> cx.Field:
    if coord is not None and coord != self.coord:
      raise ValueError(
          f'CorrelatedNoise is defined on {self.coord}, but got {coord=}.'
      )
    values = self.core.get_value()
    if self._correlation_length > 0:
      values = _periodic_convolve(values, self.dim, self.kernel())
    return values


@nnx.dataclass
class Lorenz96Base(api.Model):
  """Base class for Lorenz-96 models providing common methods."""

  dt: float = nnx.static(default=0.002, kw_only=True)
  time_integrator_cls: IntegratorCls = nnx.static(
      default=time_integrators.RungeKutta4, kw_only=True
  )
  operators: dict[str, typing.ObservationOperator] = nnx.data(
      default_factory=dict, kw_only=True
  )

  def __post_init__(self):
    # Adds input-like obs-operators if not set explicitly for demo purposes.
    if 'slow' not in self.operators and hasattr(self, 'x'):
      sel_x_and_t = transforms.SelectKeys(['x', 'time'])
      obs_op = observation_operators.TransformObservationOperator(sel_x_and_t)
      self.operators['slow'] = obs_op
    if 'fast' not in self.operators and hasattr(self, 'y'):
      sel_y_and_t = transforms.SelectKeys(['y', 'time'])
      obs_op = observation_operators.TransformObservationOperator(sel_y_and_t)
      self.operators['fast'] = obs_op

  @property
  def prognostics(self):
    prognostic_fields = {}
    if hasattr(self, 'x'):
      prognostic_fields['x'] = self.x.get_value()
    if hasattr(self, 'y'):
      prognostic_fields['y'] = self.y.get_value()
    prognostic_fields['time'] = self.time.get_value()  # pyrefly: ignore[missing-attribute]
    return prognostic_fields

  @module_utils.ensure_unchanged_state_structure
  def observe(
      self,
      queries: typing.Queries,
  ) -> typing.Observation:
    result = {}
    for k, q in queries.items():
      if k in self.operators:
        result[k] = self.operators[k].observe(self.prognostics, q)
      else:
        raise ValueError(f'No observation operator for key "{k}"')
    return result

  @property
  def timestep(self) -> np.timedelta64:
    # Lorenz96 non dimensional time roughly corresponds to 120h.
    return np.timedelta64(int(self.dt * 120 * 3600), 's')

  @property
  def inputs_spec(self):
    specs = {}
    if hasattr(self, 'x'):
      cs = {'x': self.x.get_value().coordinate, 'time': cx.Scalar()}
      specs['slow'] = {
          k: data_specs.CoordSpec.with_any_timedelta(v) for k, v in cs.items()
      }
    if hasattr(self, 'y'):
      cs = {'y': self.y.get_value().coordinate}
      if not hasattr(self, 'x'):  # fast-only
        cs['time'] = cx.Scalar()
      specs['fast'] = {
          k: data_specs.CoordSpec.with_any_timedelta(v) for k, v in cs.items()
      }
    return specs


@nnx.dataclass
class Lorenz96WithTwoScales(Lorenz96Base):
  """Two timescale Lorenz-96 model."""

  k_axis: cx.Coordinate
  j_axis: cx.Coordinate
  forcing: float | nnx.Param = nnx.data(default=10.0)
  c: float | nnx.Param = nnx.data(default=10.0)
  b: float | nnx.Param = nnx.data(default=10.0)
  h: float | nnx.Param = nnx.data(default=1.0)
  # Prognostic fields initialized in __post_init__
  x: typing.Prognostic = nnx.data(init=False)
  y: typing.Prognostic = nnx.data(init=False)
  time: typing.Prognostic = nnx.data(init=False)

  def __post_init__(self):
    ks, js = self.k_axis, self.j_axis
    kjs = cx.coords.compose(ks, js)
    self.x = typing.Prognostic(cx.field(jnp.zeros(ks.shape), ks))
    self.y = typing.Prognostic(cx.field(jnp.zeros(kjs.shape), kjs))
    self.time = typing.Prognostic(cx.field(jdt.Datetime(jdt.Timedelta())))
    super().__post_init__()

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    slice_last_time = lambda f: f.isel(timedelta=-1)
    self.x.set_value(slice_last_time(inputs['slow']['x']))
    self.y.set_value(slice_last_time(inputs['fast']['y']))
    self.time.set_value(slice_last_time(inputs['slow']['time']))

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    k, j = self.k_axis, self.j_axis
    f, c, b, h = self.forcing, self.c, self.b, self.h
    x_dot = cx.cpmap(functools.partial(lorenz96_x_term, f=f))
    y_dot = cx.cpmap(functools.partial(lorenz96_y_term, b=b, c=c))
    sum_fn = cx.cmap(jnp.sum)
    # Equation for coupled X and Y variables.
    # fmt: off
    all_terms_fn = lambda s: {
        'x': x_dot(s['x'].untag(k)).tag(k) - (h*c/b) * sum_fn(s['y'].untag(j)),
        'y': y_dot(s['y'].untag(j)).tag(j) + (h*c/b) * s['x'],
    }
    # fmt: on
    full_ode = time_integrators.ExplicitODE.from_functions(all_terms_fn)
    x, y = self.x.get_value(), self.y.get_value()
    next_xy = self.time_integrator_cls(full_ode, self.dt)({'x': x, 'y': y})  # pyrefly: ignore[bad-argument-type]
    self.x.set_value(next_xy['x'])
    self.y.set_value(next_xy['y'])
    # treated separately to avoid round off.
    self.time.set_value(self.time.get_value() + self.timestep)
    advance_random_processes(self)


class L96Parameterization(Protocol):
  """Protocol class defining the interface for Lorenz96 parameterizations."""

  def __call__(
      self,
      prognostics: dict[str, cx.Field],
      dt: float,
  ) -> cx.Field:
    """Returns cumulative correction over `dt` for a `x` or `y` variable."""


@nnx.dataclass
class StochasticLinearParameterization(nnx.Module):
  """A simple stochastic linear parameterization.

  This implements update of the form:
    update = -(b0 + b1 * X) * dt + noise_amplitude * N(0, 1) * sqrt(dt)

  The random process is advanced by the model at the end of each step.

  # Default values for `b0, b1` are taken from M2Lines book:
  # https://m2lines.github.io/L96_demo/notebooks/gcm-analogue.html
  """

  random_process: random_processes.RandomProcessModule
  b0: float | nnx.Param = nnx.data(default=0.75218026)
  b1: float | nnx.Param = nnx.data(default=0.85439536)
  noise_amplitude: float | nnx.Param = nnx.data(default=1.0)

  def __call__(
      self,
      prognostics: dict[str, cx.Field],
      dt: float,
  ) -> cx.Field:
    x = prognostics['x']
    k = x.axes['k']
    poly_fn = cx.cpmap(lambda x_k: -(self.b0 + self.b1 * x_k))
    deterministic_term = poly_fn(x.untag(k)).tag(k)
    randomness = self.random_process.state_values(k)
    stochastic_term = self.noise_amplitude * randomness
    return deterministic_term * dt + stochastic_term * jnp.sqrt(dt)


@nnx.dataclass
class TransformParameterization(nnx.Module):
  """Parameterization given by a transform of the prognostics, scaled by `dt`.

  Returns `transform(prognostics)[key] * dt`, which fits the
  `L96Parameterization` protocol. Used to plug learned transforms such as
  `learned_transforms.ForwardTowerTransform` into `Lorenz96`.

  Attributes:
    transform: Transform mapping prognostics to a dict of output fields.
    key: Key of the transform output holding the tendency correction.
  """

  transform: typing.Transform = nnx.data()
  key: str = nnx.static(default='x_correction', kw_only=True)

  def __call__(
      self,
      prognostics: dict[str, cx.Field],
      dt: float,
  ) -> cx.Field:
    return self.transform(prognostics)[self.key] * dt


@nnx.dataclass
class SingleScaleBase(Lorenz96Base):
  """Base class for models of the slow `X` variables only."""

  k_axis: cx.Coordinate
  # Prognostic fields initialized in __post_init__
  x: typing.Prognostic = nnx.data(init=False)
  time: typing.Prognostic = nnx.data(init=False)

  def __post_init__(self):
    self.x = typing.Prognostic(
        cx.field(jnp.zeros(self.k_axis.shape), self.k_axis)
    )
    self.time = typing.Prognostic(cx.field(jdt.Datetime(jdt.Timedelta())))
    super().__post_init__()

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    slice_last_time = lambda f: f.isel(timedelta=-1)
    self.x.set_value(slice_last_time(inputs['slow']['x']))
    self.time.set_value(slice_last_time(inputs['slow']['time']))


@nnx.dataclass
class Lorenz96(SingleScaleBase):
  """Single timescale Lorenz-96 model with a modular parameterizations."""

  parameterizations: Sequence[L96Parameterization] = nnx.data(default=())
  forcing: float | nnx.Param = nnx.data(default=10.0)

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    k, f = self.k_axis, self.forcing
    x, time = self.x.get_value(), self.time.get_value()
    x_dot_fn = cx.cpmap(functools.partial(lorenz96_x_term, f=f))
    x_ode = time_integrators.ExplicitODE.from_functions(
        lambda s: x_dot_fn(s.untag(k)).tag(k)
    )
    next_x = self.time_integrator_cls(x_ode, self.dt)(x)  # pyrefly: ignore[bad-argument-type]
    for parameterization in self.parameterizations:
      next_x += parameterization({'x': x, 'time': time}, self.dt)
    self.x.set_value(next_x)
    self.time.set_value(self.time.get_value() + self.timestep)
    advance_random_processes(self)


@nnx.dataclass
class LearnedLorenz96(SingleScaleBase):
  """Data-driven model of the slow variables with a learned update.

  Advances the state as `x += transform({'x': x, 'time': time})[key]`, i.e.
  the dynamics are learned from data without using the Lorenz-96 equations.

  Attributes:
    transform: Transform mapping prognostics to a dict with the update field.
    key: Key of the transform output holding the update of `x` over one step.
  """

  transform: typing.Transform = nnx.data()
  key: str = nnx.static(default='x_delta', kw_only=True)

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    x, time = self.x.get_value(), self.time.get_value()
    delta = self.transform({'x': x, 'time': time})[self.key]
    self.x.set_value(x + delta)
    self.time.set_value(time + self.timestep)
    advance_random_processes(self)


@nnx.dataclass
class Lorenz96FastMode(Lorenz96Base):
  """Fast timescale Lorenz-96 model."""

  k_axis: cx.Coordinate
  j_axis: cx.Coordinate
  parameterizations: Sequence[L96Parameterization] = nnx.data(default=())
  c: float | nnx.Param = 10.0
  b: float | nnx.Param = 10.0
  # Prognostic fields initialized in __post_init__
  y: typing.Prognostic = nnx.data(init=False)
  time: typing.Prognostic = nnx.data(init=False)

  def __post_init__(self):
    kjs = cx.coords.compose(self.k_axis, self.j_axis)
    self.y = typing.Prognostic(cx.field(jnp.zeros(kjs.shape), kjs))
    self.time = typing.Prognostic(cx.field(jdt.Datetime(jdt.Timedelta())))
    super().__post_init__()

  @module_utils.ensure_unchanged_state_structure
  def assimilate(self, inputs: dict[str, dict[str, cx.Field]]) -> None:
    slice_last_time = lambda f: f.isel(timedelta=-1)
    self.y.set_value(slice_last_time(inputs['fast']['y']))
    self.time.set_value(slice_last_time(inputs['fast']['time']))

  @module_utils.ensure_unchanged_state_structure
  def advance(self) -> None:
    c, b = self.c, self.b
    y, time = self.y.get_value(), self.time.get_value()
    y_dot_fn = cx.cpmap(functools.partial(lorenz96_y_term, b=b, c=c))
    kjs = cx.coords.compose(self.k_axis, self.j_axis)
    y_ode = time_integrators.ExplicitODE.from_functions(
        lambda s: y_dot_fn(s.untag(kjs)).tag(kjs)
    )
    next_y = self.time_integrator_cls(y_ode, self.dt)(y)  # pyrefly: ignore[bad-argument-type]
    for parameterization in self.parameterizations:
      next_y += parameterization({'y': y, 'time': time}, self.dt)
    self.y.set_value(next_y)
    self.time.set_value(self.time.get_value() + self.timestep)
    advance_random_processes(self)
