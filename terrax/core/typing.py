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

"""Types used by terrax API."""

from __future__ import annotations

import dataclasses
import datetime
import functools
from typing import Any, Callable, Generic, Protocol, TypeVar

import coordax as cx
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from terrax.core import scales

units = scales.units
#
# Generic types.
#
Array = np.ndarray | jax.Array
Dtype = jax.typing.DTypeLike | Any
Numeric = float | int | Array
PRNGKeyArray = jax.Array
ShapeDtypeStruct = jax.ShapeDtypeStruct
Timestep = np.timedelta64 | float
TimedeltaLike = str | np.timedelta64 | pd.Timestamp | datetime.timedelta
ShapeFloatStruct = functools.partial(ShapeDtypeStruct, dtype=jnp.float32)
Quantity = scales.Quantity

T = TypeVar('T')


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Auxiliary(Generic[T]):
  """Wrapper indicating a supporting field/coord excluded from final output.

  Use this to mark query entries that are needed for internal computation
  (e.g., conditioning inputs for observation operators) but should not appear
  in the returned observations. This enables a clean three-stage pipeline:

  1. **Unwrap:** Use ``unwrap_auxiliary`` to extract the underlying spec.
  2. **Compute:** Perform transforms using the unwrapped value.
  3. **Filter:** Exclude keys whose original query entry was ``Auxiliary``.
  """

  spec: T

  def tree_flatten(self):
    return (self.spec,), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(children[0])


def unwrap_auxiliary(spec: T | Auxiliary[T]) -> tuple[T, bool]:
  """Returns underlying spec and a bool indicating if spec is Auxiliary."""
  is_auxiliary = isinstance(spec, Auxiliary)
  inner = spec.spec if is_auxiliary else spec  # pytype: disable=attribute-error
  return inner, is_auxiliary


#
# Main structured API types.
#
Fields = dict[str, cx.Field]
InputFields = dict[str, Fields]
Observation = dict[str, dict[str, cx.Field]]
Query = dict[
    str,
    cx.Coordinate | cx.Field | Auxiliary[cx.Coordinate] | Auxiliary[cx.Field],
]
Queries = dict[str, Query]


#
# Generic unstructured input/output types.
#
Pytree = Any
PyTreeState = TypeVar('PyTreeState')


#
# Simulation state types, structs and function signatures.
#


class DynamicInput(nnx.Variable):
  """A class for variables that hold conditioning data for the simulation."""


class SimulationVariable(nnx.Variable):
  """A base class for variables that capture dynamic simulation state."""


class Prognostic(SimulationVariable):
  """Variables that represent the prognostic state of the simulation."""


class Diagnostic(SimulationVariable):
  """Variables that represent the diagnostic state of the simulation."""


class Randomness(SimulationVariable):
  """Variables that represent the random processes state of the simulation."""


class Coupling(SimulationVariable):
  """Variables that represent the coupling state of the simulation."""


State = TypeVar('State')


@dataclasses.dataclass(frozen=True)
class SimulationState(Generic[State]):
  """Simulation state decomposed into prognostic, diagnostic and randomness.

  Attributes:
    prognostics: Prognostic state of the simulation.
    diagnostics: Optional diagnostic state of the simulation.
    randomness: Optional state of random processes in the simulation.
    extras: Optional additional simulation state variables.
  """

  prognostics: State
  diagnostics: State
  randomness: State
  extras: dict[str, State] = dataclasses.field(default_factory=dict)


jax.tree_util.register_dataclass(
    SimulationState,
    data_fields=('prognostics', 'diagnostics', 'randomness', 'extras'),
    meta_fields=(),
)


StepFn = Callable[[PyTreeState], PyTreeState]


#
# Auxiliary types for intermediate computations.
#
@dataclasses.dataclass(eq=True, order=True, frozen=True)
class KeyWithCosLatFactor:
  """Class describing a key by `name` and an integer `factor_order`."""

  name: str
  factor_order: int


#
# Protocol definitions.
#


class Transform(Protocol):
  """Protocol for pytree transforms."""

  def __call__(self, inputs: dict[str, cx.Field]) -> dict[str, cx.Field]:
    ...

  def output_shapes(
      self, input_shapes: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    ...


class FieldTransform(Protocol):
  """Protocol for elementwise transforms supporting single item delivery."""

  def __call__(self, inputs: cx.Field) -> cx.Field:
    ...


class ObservationOperator(Protocol):
  """Protocol for observation operators."""

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: Query,
  ) -> dict[str, cx.Field]:
    """Returns observations for `query`."""
    ...


TransformFactory = Callable[..., Transform]
