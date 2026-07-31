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

"""Defines observation operator API and sample operators for NeuralGCM."""

import abc
import dataclasses
from typing import Sequence

import coordax as cx
from flax import nnx
from terrax.core import typing


class ObservationOperatorABC(nnx.Module, abc.ABC):
  """Base class for observation operators."""

  @abc.abstractmethod
  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    """Returns observations for `query`."""
    ...

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(pytree=False, **kwargs)


@dataclasses.dataclass
class DataObservationOperator(ObservationOperatorABC):
  """Operator that returns pre-computed fields for matching queries.

  Resolves each query entry against the pre-computed ``fields`` dictionary:

  - ``Coordinate`` query → selects/slices from the matching field.
  - ``Field`` query → passes the query value through as-is.
  - ``Auxiliary(...)`` query → skipped (excluded from output).

  Attributes:
    fields: A dictionary of ``coordax.Field``s to match against.
  """

  fields: dict[str, cx.Field]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    """Returns observations for ``query`` matched against ``self.fields``."""
    del inputs  # unused.
    observations = {}
    valid_keys = list(self.fields.keys())
    for k, raw_entry in query.items():
      entry, is_aux = typing.unwrap_auxiliary(raw_entry)  # pyrefly: ignore[bad-argument-type]
      if is_aux:
        continue
      if cx.is_field(entry):
        observations[k] = entry
      elif cx.is_coord(entry):
        if k not in valid_keys:
          raise ValueError(f'query contains {k=} not in {valid_keys}')
        field = self.fields[k]
        if field.coordinate == entry:
          result = field
        else:
          try:
            result = field.sel({field.coordinate: entry})
          except KeyError as e:
            raise ValueError(
                f'query coordinate for {k!r} is not a valid slice of field:\n'
                f'{entry}\nvs\n{field.coordinate}'
            ) from e
        observations[k] = result
      else:
        raise ValueError(
            f'Unsupported query entry type for {k!r}: {type(entry)}'
        )
    return observations


@dataclasses.dataclass
class TransformObservationOperator(ObservationOperatorABC):
  """Operator that transforms inputs and resolves queries from the output.

  Pipeline:

  1. Collect ``Field`` entries from the query as direct values.
  2. For ``requested_fields_from_query`` keys missing from query fields,
     run ``fallback_transform`` to produce them.
  3. Run ``transform`` on ``inputs`` augmented with the resolved fields.
  4. Build an effective query where fallback-produced fields replace their
     original coordinate entries (preserving ``Auxiliary`` status), and keys
     absent from the original query are added as ``Auxiliary``.
  5. Delegate all output resolution to ``DataObservationOperator``.

  Attributes:
    transform: Transform to apply to inputs.
    requested_fields_from_query: Keys that the transform expects as inputs (not
      outputs). Values come from ``Field`` entries in the query, or from
      ``fallback_transform`` when the query has a ``Coordinate`` or omits the
      key entirely.
    fallback_transform: Optional transform that produces missing requested
      fields from ``inputs`` and available query fields.
  """

  transform: typing.Transform
  requested_fields_from_query: tuple[str, ...] = ()
  fallback_transform: typing.Transform | None = None

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    unwrapped_query = {}
    auxiliary_keys = set()
    for k, v in query.items():
      inner, is_aux = typing.unwrap_auxiliary(v)  # pyrefly: ignore[bad-argument-type]
      unwrapped_query[k] = inner
      if is_aux:
        auxiliary_keys.add(k)
    q_fields = {k: v for k, v in unwrapped_query.items() if cx.is_field(v)}

    missing = [k for k in self.requested_fields_from_query if k not in q_fields]
    fallback_fields = {}
    if missing:
      if self.fallback_transform is None:
        raise ValueError(
            f'Missing fields: {missing}, and {self.fallback_transform=}'
        )
      fallback_outputs = self.fallback_transform(inputs | q_fields)
      missing_in_fallback = [k for k in missing if k not in fallback_outputs]
      if missing_in_fallback:
        raise ValueError(
            'fallback_transform did not resolve all missing requested fields. '
            f'Missing: {missing_in_fallback}. '
            f'All available: {list(fallback_outputs.keys())}'
        )
      fallback_fields = {k: fallback_outputs[k] for k in missing}

    all_fields = q_fields | fallback_fields
    observations = self.transform(inputs | all_fields)

    # Build effective query for DataObservationOperator.
    #    - Fallback fields for keys present in query: replace the coordinate
    #      with the fallback field, preserving Auxiliary wrapping.
    #    - Fallback fields for keys absent from query: add as Auxiliary.
    effective_query = dict(query)
    for k, field in fallback_fields.items():
      if k in query:
        if k in auxiliary_keys:
          effective_query[k] = typing.Auxiliary(field)
        else:  # passed as a coordinate in query, fill with fallback field.
          effective_query[k] = field
      else:
        effective_query[k] = typing.Auxiliary(field)

    return DataObservationOperator(observations).observe({}, effective_query)


@dataclasses.dataclass
class ObservationOperatorWithRenaming(ObservationOperatorABC):
  """Operator wrapper that converts between different naming conventions.

  Attributes:
    operator: Observation operator that performs computation.
    renaming_dict: A dictionary mapping new names to those used by `operator`.
  """

  operator: typing.ObservationOperator
  renaming_dict: dict[str, str]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    """Returns observations for `query` matched against `self.fields`."""
    renamed_query = {self.renaming_dict.get(k, k): v for k, v in query.items()}
    observation = self.operator.observe(inputs, renamed_query)
    inverse_renaming_dict = {v: k for k, v in self.renaming_dict.items()}
    return {inverse_renaming_dict.get(k, k): v for k, v in observation.items()}


@dataclasses.dataclass
class MultiObservationOperator(ObservationOperatorABC):
  """Operator that dispatches queries to multiple operators.

  Attributes:
    keys_to_operator: A dictionary mapping query keys to observation operators.
  """

  keys_to_operator: dict[tuple[str, ...], typing.ObservationOperator]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    outputs = {}
    supported_keys = set(sum(self.keys_to_operator.keys(), start=()))
    query_keys = set(query.keys())
    if not query_keys.issubset(supported_keys):
      raise ValueError(
          f'query keys {query_keys} are not a subset of supported keys'
          f' {supported_keys}'
      )
    for key_tuple, obs_op in self.keys_to_operator.items():
      sub_query = {}
      for key in key_tuple:
        if key in query:
          sub_query[key] = query[key]
      outputs |= obs_op.observe(inputs, sub_query)
    return outputs


@nnx.dataclass
class DispatchByCoordinateObservationOperator(ObservationOperatorABC):
  """Operator that dispatches queries to operators based on coordinates.

  Attributes:
    coord_to_operator: A dict mapping coordinate to dispatch operators.
  """

  coord_to_operator: dict[cx.Coordinate, typing.ObservationOperator]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: typing.Query,
  ) -> dict[str, cx.Field]:
    outputs = {}
    # Group query entries by matching coordinate in coord_to_operator.
    coord_to_sub_query = {coord: {} for coord in self.coord_to_operator}

    for k, raw_entry in query.items():
      entry, _ = typing.unwrap_auxiliary(raw_entry)  # pyrefly: ignore[bad-argument-type]
      if cx.is_field(entry):
        entry_coord = entry.coordinate
      elif cx.is_coord(entry):
        entry_coord = entry
      else:
        raise ValueError(
            f'Unsupported query entry type for {k!r}: {type(entry)}'
        )

      entry_axes = set(entry_coord.axes)
      matched_coords = [
          coord
          for coord in self.coord_to_operator
          if set(coord.axes).issubset(entry_axes)
      ]

      if not matched_coords:
        supported_coords = list(self.coord_to_operator.keys())
        raise ValueError(
            f'query entry {k!r} with coordinate {entry_coord} does not match '
            f'any supported coordinates: {supported_coords}'
        )
      if len(matched_coords) > 1:
        raise ValueError(
            f'query entry {k!r} with coordinate {entry_coord} matches multiple '
            f'supported coordinates: {matched_coords}'
        )

      [matched_coord] = matched_coords
      coord_to_sub_query[matched_coord][k] = raw_entry

    for coord, obs_op in self.coord_to_operator.items():
      sub_query = coord_to_sub_query[coord]
      if sub_query:
        outputs |= obs_op.observe(inputs, sub_query)

    return outputs

  @classmethod
  def construct(
      cls,
      coords: Sequence[cx.Coordinate] | cx.Coordinate,
      operators: (
          Sequence[typing.ObservationOperator] | typing.ObservationOperator
      ),
  ):
    """Custom constructor based on grids and operators sequences."""
    if isinstance(coords, cx.Coordinate):
      coords = [coords]
    if not isinstance(operators, Sequence):
      operators = [operators] * len(coords)
    coord_to_operator = {
        grid: op for grid, op in zip(coords, operators, strict=True)
    }
    return cls(coord_to_operator=coord_to_operator)
