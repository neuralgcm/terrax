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

"""Modules that define differential equations."""

from typing import Sequence

import coordax as cx
from flax import nnx
from terrax.core import pytree_utils
from terrax.core import time_integrators
from terrax.core import typing

ImplicitExplicitODE = time_integrators.ImplicitExplicitODE
SemiLagrangianImplicitExplicitODE = (
    time_integrators.SemiLagrangianImplicitExplicitODE
)
ExplicitODE = time_integrators.ExplicitODE
ShapeFloatStruct = typing.ShapeFloatStruct


class SimTimeEquation(time_integrators.ExplicitODE):
  """Equation module describing evolution of `sim_time` variable."""

  def explicit_terms(self, x: typing.Pytree) -> typing.Pytree:
    x_dict, from_dict_fn = pytree_utils.as_dict(x)
    if 'sim_time' not in x_dict:
      raise ValueError(f'sim_time not found in {x_dict.keys()}')
    terms = {k: None if k != 'sim_time' else 1.0 for k in x_dict.keys()}
    return from_dict_fn(terms)


@nnx.dataclass
class ExplicitTransformEquation(time_integrators.ExplicitODE):
  """Explicit equation whose terms are parameterized by a transform."""

  explicit_terms_transform: typing.Transform = nnx.data()

  def explicit_terms(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    tendencies = self.explicit_terms_transform(inputs)
    tendencies = pytree_utils.replace_with_matching_or_default(
        inputs, tendencies, None
    )
    return tendencies


def _sum_non_nones(*args: cx.Field | None) -> cx.Field:
  terms = [x for x in args if x is not None]
  return sum(terms) if terms else cx.field(0.0)  # pyrefly: ignore[bad-return]


@nnx.dataclass
class ComposedODE(ImplicitExplicitODE):
  """Composed equation with exactly one ImplicitExplicitODE instance."""

  equations: Sequence[ExplicitODE | ImplicitExplicitODE] = nnx.data()

  def __post_init__(self):
    imex_equations = [
        x for x in self.equations if isinstance(x, ImplicitExplicitODE)
    ]
    if len(imex_equations) != 1:
      raise ValueError(
          'ComposedODE only supports exactly 1 ImplicitExplicitODE, '
          f'got {imex_equations=}'
      )
    (implicit_explicit_eq,) = imex_equations
    self.implicit_explicit_equation = implicit_explicit_eq

  def explicit_terms(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    explicit_tendencies = [fn.explicit_terms(x) for fn in self.equations]
    # pylint: disable=undefined-variable
    return {
        k: _sum_non_nones(*(et.get(k, None) for et in explicit_tendencies))
        for k in x.keys()
    }
    # pylint: enable=undefined-variable

  def implicit_terms(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.implicit_explicit_equation.implicit_terms(x)

  def implicit_inverse(
      self, x: dict[str, cx.Field], step_size: float
  ) -> dict[str, cx.Field]:
    return self.implicit_explicit_equation.implicit_inverse(x, step_size)

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


@nnx.dataclass
class ComposedSemiLagrangianODE(SemiLagrangianImplicitExplicitODE):
  """Composed equation with one `SemiLagrangianImplicitExplicitODE` instance.

  Trajectories, semi-Lagrangian transport, and implicit solves are delegated
  to the single `SemiLagrangianImplicitExplicitODE` instance, while the
  explicit terms of other `ExplicitODE` instances are combined with its
  `nonadvective_terms` as additional non-advective forcing.
  """

  equations: Sequence[ExplicitODE | SemiLagrangianImplicitExplicitODE] = (
      nnx.data()
  )

  def __post_init__(self):
    sl_equations = [
        x
        for x in self.equations
        if isinstance(x, SemiLagrangianImplicitExplicitODE)
    ]
    if len(sl_equations) != 1:
      raise ValueError(
          'ComposedSemiLagrangianODE only supports exactly 1 '
          f'SemiLagrangianImplicitExplicitODE, got {sl_equations=}'
      )
    (sl_eq,) = sl_equations
    self.semi_lagrangian_equation = sl_eq
    self.explicit_equations = tuple(
        x
        for x in self.equations
        if not isinstance(x, SemiLagrangianImplicitExplicitODE)
    )

  def nonadvective_terms(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    tendencies = [self.semi_lagrangian_equation.nonadvective_terms(x)] + [
        fn.explicit_terms(x) for fn in self.explicit_equations
    ]
    # pylint: disable=undefined-variable
    return {
        k: _sum_non_nones(*(t.get(k, None) for t in tendencies))
        for k in x.keys()
    }
    # pylint: enable=undefined-variable

  def implicit_terms(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self.semi_lagrangian_equation.implicit_terms(x)

  def implicit_inverse(
      self, x: dict[str, cx.Field], step_size: float
  ) -> dict[str, cx.Field]:
    return self.semi_lagrangian_equation.implicit_inverse(x, step_size)

  def nodal_velocities(self, x: dict[str, cx.Field]) -> typing.Pytree:
    return self.semi_lagrangian_equation.nodal_velocities(x)

  def departure_points(
      self,
      velocities: typing.Pytree,
      dt: float,
      initial_guess: typing.Pytree | None = None,
  ) -> typing.Pytree:
    return self.semi_lagrangian_equation.departure_points(
        velocities, dt, initial_guess=initial_guess
    )

  def semi_lagrangian_transport(
      self, x: dict[str, cx.Field], departure: typing.Pytree
  ) -> dict[str, cx.Field]:
    return self.semi_lagrangian_equation.semi_lagrangian_transport(x, departure)

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


@nnx.dataclass
class ComposedExplicitODE(ExplicitODE):
  """Composed explicit equation."""

  equations: Sequence[ExplicitODE] = nnx.data()

  def explicit_terms(self, x: dict[str, cx.Field]) -> dict[str, cx.Field]:
    explicit_tendencies = [fn.explicit_terms(x) for fn in self.equations]
    # pylint: disable=undefined-variable
    return {
        k: _sum_non_nones(*(et.get(k, None) for et in explicit_tendencies))
        for k in x.keys()
    }
    # pylint: enable=undefined-variable


def compose_equations(
    equations: Sequence[
        ExplicitODE | ImplicitExplicitODE | SemiLagrangianImplicitExplicitODE
    ],
) -> ComposedExplicitODE | ComposedODE | ComposedSemiLagrangianODE:
  """Composes a sequence of equations into an appropriate composed ODE module."""
  if any(isinstance(eq, SemiLagrangianImplicitExplicitODE) for eq in equations):
    if any(isinstance(eq, ImplicitExplicitODE) for eq in equations):
      raise ValueError(
          'compose_equations does not support mixing'
          ' SemiLagrangianImplicitExplicitODE and ImplicitExplicitODE.'
      )
    return ComposedSemiLagrangianODE(equations)  # pyrefly: ignore[bad-argument-type]
  if any(isinstance(eq, ImplicitExplicitODE) for eq in equations):
    return ComposedODE(equations)  # pyrefly: ignore[bad-argument-type]
  return ComposedExplicitODE(equations)  # pyrefly: ignore[bad-argument-type]
