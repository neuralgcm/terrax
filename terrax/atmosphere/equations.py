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

"""Modules parameterizing PDEs describing atmospheric processes."""

import functools
from typing import Callable, Literal, Sequence

import coordax as cx
from dinosaur import coordinate_systems
from dinosaur import held_suarez
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations
from dinosaur import sigma_coordinates
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates
from terrax.core import field_utils
from terrax.core import orographies
from terrax.core import spherical_harmonics
from terrax.core import time_integrators
from terrax.core import transforms
from terrax.core import typing
from terrax.core import units

# Averaged temperature per pressure level from arco-ERA5 data over the period of
# ('1990-01-01', '1998-01-01') subsampled every 6*35 hours and weighted by area.
REF_PRESSURE = [
    100,
    500,
    3000,
    7000,
    10000,
    12500,
    15000,
    17500,
    20000,
    25000,
    45000,
    65000,
    80000,
    90000,
    100000,
]
REF_TEMPERATURE = [
    260.7,
    240.36,
    217.34,
    207.15,
    204.92,
    207.68,
    211.41,
    215.04,
    218.43,
    225.5,
    253.27,
    270.29,
    278.98,
    282.98,
    288.24
]


def get_reference_temperature(
    model_levels: (
        coordinates.SigmaLevels
        | coordinates.HybridLevels
        | coordinates.PressureLevels
    ),
) -> cx.Field:
  """Returns reference temperature for the given model levels."""
  p_surf_ref = 101325.0
  if isinstance(model_levels, coordinates.SigmaLevels):
    get_ticks_fn = lambda c: c.fields['sigma'] * p_surf_ref
  elif isinstance(model_levels, coordinates.HybridLevels):
    get_ticks_fn = functools.partial(
        coordinates.HybridLevels.pressure_centers, surface_pressure=p_surf_ref  # pyrefly: ignore[bad-argument-type]
    )
  elif isinstance(model_levels, coordinates.PressureLevels):
    get_ticks_fn = lambda c: c.fields['pressure'] * 100  # Convert to Pa.
  else:
    raise ValueError(
        f'levels should be Sigma, Pressure or Hybrid, got {type(model_levels)=}'
    )
  return field_utils.reconstruct_1d_field_from_ref_values(
      model_levels,
      REF_PRESSURE,
      REF_TEMPERATURE,
      interpolation_space='linear',
      get_tick_fn=get_ticks_fn,
  )


def get_temperature_linearization_transform(
    ref_temperatures: cx.Field,
    abs_temperature_key: str = 'temperature',
    del_temperature_key: str = 'temperature_variation',
) -> typing.Transform:
  """Constructs transform for linearizing temperature around `ref_temperature`."""

  def linearize_fn(abs_temp: cx.Field) -> cx.Field:
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in abs_temp.dims for d in ylm_dims):
      ylm_grid = cx.coords.extract(
          abs_temp.coordinate, coordinates.SphericalHarmonicGrid
      )
      del_temp = ylm_grid.add_constant(abs_temp, -ref_temperatures)
    else:
      del_temp = abs_temp - ref_temperatures
    return del_temp

  return transforms.Sequential([
      transforms.ApplyFnToKeys(
          fn=linearize_fn,
          keys=[abs_temperature_key],
          include_remaining=True,
      ),
      transforms.RenameKeys(rename_dict={abs_temperature_key: del_temperature_key}),
  ])


def get_temperature_delinearization_transform(
    ref_temperatures: cx.Field,
    abs_temperature_key: str = 'temperature',
    del_temperature_key: str = 'temperature_variation',
) -> typing.Transform:
  """Constructs transform for reversing temperature linearization."""

  def delinearize_fn(del_temp: cx.Field) -> cx.Field:
    """Applies delinearization to `del_temp` field."""
    ylm_dims = ('longitude_wavenumber', 'total_wavenumber')
    if all(d in del_temp.dims for d in ylm_dims):
      ylm_grid = cx.coords.extract(
          del_temp.coordinate, coordinates.SphericalHarmonicGrid
      )
      abs_temp = ylm_grid.add_constant(del_temp, ref_temperatures)
    else:
      abs_temp = del_temp + ref_temperatures
    return abs_temp

  return transforms.Sequential([
      transforms.ApplyFnToKeys(
          fn=delinearize_fn,
          keys=[del_temperature_key],
          include_remaining=True,
      ),
      transforms.RenameKeys(rename_dict={del_temperature_key: abs_temperature_key}),
  ])


class _PrimitiveEquationsBase:
  """Shared state conversion and implicit solve methods for primitive equations."""

  equation_cls: Callable[..., object]

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float] | cx.Field,
      tracer_names: Sequence[str],
      orography_module: orographies.ModalOrography,
      vertical_advection: Callable[..., typing.Array] | None = None,
      include_vertical_advection: bool = True,
      humidity_key: str = 'specific_humidity',
      cloud_keys: tuple[str, ...] = (
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
  ):
    if cx.is_field(reference_temperatures):
      if not cx.contains_dims(reference_temperatures, levels):
        raise ValueError(
            f'reference_temperatures provided on levels different from {levels}'
        )
      assert isinstance(reference_temperatures, cx.Field)  # make pytype happy.
      t_ref_tuple = tuple(float(t) for t in reference_temperatures.data)  # pyrefly: ignore[bad-argument-type]
    else:
      t_ref_tuple = tuple(reference_temperatures)
      reference_temperatures = cx.field(np.array(t_ref_tuple), levels)

    self.ylm_map = ylm_map
    self.levels = levels
    self.orography_module = orography_module
    self.sim_units = sim_units
    self.orography = orography_module
    self.t_ref_tuple = t_ref_tuple
    self.tracer_names = tracer_names
    self.nodal_tracers: tuple[str, ...] = ()
    self.include_vertical_advection = include_vertical_advection
    self.linearize_transform = get_temperature_linearization_transform(
        ref_temperatures=reference_temperatures
    )
    self.delinearize_transform = get_temperature_delinearization_transform(
        ref_temperatures=reference_temperatures
    )
    self.linear_to_absolute_rename = transforms.RenameKeys(
        rename_dict={'temperature_variation': 'temperature'}
    )
    if isinstance(levels, coordinates.SigmaLevels):
      self.dinosaur_coords = coordinate_systems.CoordinateSystem(
          horizontal=self.ylm_map.dinosaur_grid,
          vertical=self.levels.sigma_levels,  # pyrefly: ignore[missing-attribute]
          spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
      )
      if vertical_advection is None:
        vertical_advection = sigma_coordinates.centered_vertical_advection
      self.vertical_advection = vertical_advection
      self.unit_kwargs = {}
    elif isinstance(levels, coordinates.HybridLevels):
      self.dinosaur_coords = coordinate_systems.CoordinateSystem(
          horizontal=self.ylm_map.dinosaur_grid,
          vertical=self.levels.hybrid_levels,  # pyrefly: ignore[missing-attribute]
          spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
      )
      if vertical_advection is None:
        vertical_advection = hybrid_coordinates.centered_vertical_advection
      self.vertical_advection = vertical_advection
      self.unit_kwargs = {
          'hpa_quantity': typing.units.hPa,
          'reference_surface_pressure': 101325.0 * typing.units.pascal,
      }
    else:
      raise ValueError(f'Unsupported vertical coordinate system: {levels}')
    if humidity_key in tracer_names:
      self.humidity_key = humidity_key
    else:
      self.humidity_key = None
    present_cloud_keys = tuple(k for k in cloud_keys if k in tracer_names)
    if present_cloud_keys:
      self.cloud_keys = present_cloud_keys
    else:
      self.cloud_keys = None

  def _equation_kwargs(self) -> dict[str, object]:
    return {
        'coords': self.dinosaur_coords,
        'physics_specs': self.sim_units,
        'reference_temperature': np.asarray(self.t_ref_tuple),
        'orography': self.orography_module.modal_orography.data,
        'vertical_advection': self.vertical_advection,
        'include_vertical_advection': self.include_vertical_advection,
        'humidity_key': self.humidity_key,
        'cloud_keys': self.cloud_keys,
        **self.unit_kwargs,
    }

  @property
  def primitive_equation(self):
    return self.equation_cls(**self._equation_kwargs())

  @property
  def T_ref(self) -> typing.Array:  # pylint: disable=invalid-name
    return self.primitive_equation.T_ref

  def _to_primitive_equations_state(
      self, inputs: dict[str, cx.Field]
  ) -> primitive_equations.State:
    """Converts a dict of fields to a primitive equations state."""
    inputs = self.linearize_transform(inputs)
    tracers_dict = {k: inputs[k].data for k in self.tracer_names}
    log_surface_pressure = inputs['log_surface_pressure'].data[np.newaxis]
    return primitive_equations.State(
        divergence=inputs['divergence'].data,  # pyrefly: ignore[bad-argument-type]
        vorticity=inputs['vorticity'].data,  # pyrefly: ignore[bad-argument-type]
        temperature_variation=inputs['temperature_variation'].data,  # pyrefly: ignore[bad-argument-type]
        tracers=tracers_dict,  # pyrefly: ignore[bad-argument-type]
        log_surface_pressure=log_surface_pressure,  # pyrefly: ignore[bad-argument-type]
    )

  def _from_primitive_equations_state(
      self, state: primitive_equations.State, is_tendency: bool = True
  ) -> dict[str, cx.Field]:
    """Converts a primitive equations state to a dict of fields."""
    sigma_levels, ylm_grid = self.levels, self.ylm_map.modal_grid
    nodal_grid = self.ylm_map.nodal_grid
    tracers = {
        k: cx.field(
            state.tracers[k],  # pyrefly: ignore[bad-argument-type]
            sigma_levels,
            nodal_grid if k in self.nodal_tracers else ylm_grid,
        )
        for k in self.tracer_names
    }
    volume_field_names = ['divergence', 'vorticity', 'temperature_variation']
    volume_fields = {
        k: cx.field(getattr(state, k), sigma_levels, ylm_grid)
        for k in volume_field_names
    }
    if is_tendency:
      volume_fields = self.linear_to_absolute_rename(volume_fields)
    else:
      volume_fields = self.delinearize_transform(volume_fields)
    lsp = cx.field(jnp.squeeze(state.log_surface_pressure, axis=0), ylm_grid)  # pyrefly: ignore[bad-argument-type]
    return volume_fields | tracers | {'log_surface_pressure': lsp}

  def implicit_terms(self, state: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self._from_primitive_equations_state(
        self.primitive_equation.implicit_terms(
            self._to_primitive_equations_state(state)
        )
    )

  def implicit_inverse(
      self, state: dict[str, cx.Field], step_size: float
  ) -> dict[str, cx.Field]:
    return self._from_primitive_equations_state(
        self.primitive_equation.implicit_inverse(
            self._to_primitive_equations_state(state), step_size
        ),
        is_tendency=False,
    )


class PrimitiveEquations(
    _PrimitiveEquationsBase, time_integrators.ImplicitExplicitODE
):
  """Equation module for Eulerian primitive equations.

  This module wraps methods of an appropriate primitive equations class from
  `dinosaur` and converts between dict[str, cx.Field] and dinosaur convention
  representations. The type of primitive equation solver is selected by the
  type of the vertical coordinate system. Supported vertical coordinates include
  SigmaLevels and HybridLevels for which spectral solvers are available. Other
  arguments control the additional features of the primitive equations solver,
  such as vertical advection and account for moisture species.

  Attributes:
    ylm_map: Spherical harmonics mapping for the horizontal grid.
    levels: Vertical levels coordinate.
    sim_units: Physical constants and units for nondimensionalization.
    reference_temperatures: Reference temperatures used for linearization.
    tracer_names: A sequence of names of tracers to be evolved by dynamics.
    orography_module: Orography module that provides modal orography data.
    vertical_advection: A optional custom function that implements vertical
      advection scheme. If None, a default centered difference scheme will be
      used based on the type of `levels`.
    include_vertical_advection: Whether to include vertical advection terms.
    humidity_key: Key in tracers names that corresponds to specific humidity. If
      the key is not present in `tracer_names`, uses dry primitive equations.
    cloud_keys: Keys in tracers names that corresponds to cloud species. Uses
      only keys that are present in `tracer_names`. If at least one of the cloud
      species is present, humidity key must be present in `tracer_names`.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float] | cx.Field,
      tracer_names: Sequence[str],
      orography_module: orographies.ModalOrography,
      vertical_advection: Callable[..., typing.Array] | None = None,
      include_vertical_advection: bool = True,
      humidity_key: str = 'specific_humidity',
      cloud_keys: tuple[str, ...] = (
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
  ):
    super().__init__(
        ylm_map=ylm_map,
        levels=levels,
        sim_units=sim_units,
        reference_temperatures=reference_temperatures,
        tracer_names=tracer_names,
        orography_module=orography_module,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
        humidity_key=humidity_key,
        cloud_keys=cloud_keys,
    )
    if isinstance(levels, coordinates.SigmaLevels):
      self.equation_cls = primitive_equations.PrimitiveEquationsSigma
    elif isinstance(levels, coordinates.HybridLevels):
      self.equation_cls = primitive_equations.PrimitiveEquationsHybrid

  def explicit_terms(self, state: dict[str, cx.Field]) -> dict[str, cx.Field]:
    return self._from_primitive_equations_state(
        self.primitive_equation.explicit_terms(
            self._to_primitive_equations_state(state)
        )
    )


class SemiLagrangianPrimitiveEquations(
    _PrimitiveEquationsBase, time_integrators.SemiLagrangianImplicitExplicitODE
):
  """Equation module for semi-Lagrangian primitive equations.

  Wraps `dinosaur.primitive_equations.SemiLagrangianPrimitiveEquations` (for
  `SigmaLevels`) or `SemiLagrangianPrimitiveEquationsHybrid` (for
  `HybridLevels`), converting between `dict[str, cx.Field]` and `dinosaur`
  state representations. Must be integrated with a semi-Lagrangian time
  integrator such as `time_integrators.SemiLagrangianCrankNicolsonRK2` (possibly
  composed with explicit forcing via `equations.ComposedSemiLagrangianODE`).

  Attributes:
    ylm_map: Spherical harmonics mapping for the horizontal grid.
    levels: Vertical levels coordinate (`SigmaLevels` or `HybridLevels`).
    sim_units: Physical constants and units for nondimensionalization.
    reference_temperatures: Reference temperatures used for linearization.
    tracer_names: A sequence of names of tracers to be evolved by dynamics.
    orography_module: Orography module that provides modal orography data.
    vertical_advection: Retained for call-compatibility with
      `PrimitiveEquations`; vertical advection in the semi-Lagrangian solver is
      handled by transport along 3-D trajectories.
    include_vertical_advection: Retained for call-compatibility with
      `PrimitiveEquations`.
    humidity_key: Key in `tracer_names` corresponding to specific humidity. If
      not present in `tracer_names`, uses dry primitive equations.
    cloud_keys: Keys in `tracer_names` corresponding to cloud species.
    coriolis_mode: `'planetary_momentum'` transports planetary momentum
      (suitable for long time steps); `'explicit'` keeps the Coriolis force as
      an explicit tendency.
    interpolation_order: Horizontal interpolation order for transported fields,
      `'cubic'` or `'linear'`.
    vertical_interpolation_order: Vertical interpolation order for transported
      fields, `'linear'` or `'cubic'`.
    monotone_tracers: Whether to transport tracers with quasi-monotone limiter.
    monotone_dynamics: Whether to transport dynamical fields with quasi-monotone
      limiter.
    nodal_tracers: Names of tracers that are carried in nodal representation
      instead of modal. Such tracers are expected on the nodal grid of `ylm_map`
      and are returned on that grid (and must be excluded from modal filters).
    departure_iterations: Number of fixed-point iterations in departure-point
      solves.
    terrain_smoothed_log_sp: Whether to transport terrain-smoothed log surface
      pressure following Ritchie & Tanguay (1996).
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float] | cx.Field,
      tracer_names: Sequence[str],
      orography_module: orographies.ModalOrography,
      vertical_advection: Callable[..., typing.Array] | None = None,
      include_vertical_advection: bool = True,
      humidity_key: str = 'specific_humidity',
      cloud_keys: tuple[str, ...] = (
          'specific_cloud_ice_water_content',
          'specific_cloud_liquid_water_content',
      ),
      *,
      coriolis_mode: Literal[
          'planetary_momentum', 'explicit'
      ] = 'planetary_momentum',
      interpolation_order: Literal['cubic', 'linear'] = 'cubic',
      vertical_interpolation_order: Literal['linear', 'cubic'] = 'linear',
      monotone_tracers: bool = False,
      monotone_dynamics: bool = False,
      nodal_tracers: Sequence[str] = (),
      departure_iterations: int = 1,
      terrain_smoothed_log_sp: bool = True,
  ):
    super().__init__(
        ylm_map=ylm_map,
        levels=levels,
        sim_units=sim_units,
        reference_temperatures=reference_temperatures,
        tracer_names=tracer_names,
        orography_module=orography_module,
        vertical_advection=vertical_advection,
        include_vertical_advection=include_vertical_advection,
        humidity_key=humidity_key,
        cloud_keys=cloud_keys,
    )
    if isinstance(levels, coordinates.SigmaLevels):
      self.equation_cls = primitive_equations.SemiLagrangianPrimitiveEquations
    elif isinstance(levels, coordinates.HybridLevels):
      self.equation_cls = (
          primitive_equations.SemiLagrangianPrimitiveEquationsHybrid
      )
    nodal_tracers_tuple = tuple(nodal_tracers)
    unknown_nodal_tracers = set(nodal_tracers_tuple) - set(tracer_names)
    if unknown_nodal_tracers:
      raise ValueError(
          f'nodal_tracers {sorted(unknown_nodal_tracers)} are not present in'
          f' {tracer_names=}'
      )
    self.coriolis_mode = coriolis_mode
    self.interpolation_order = interpolation_order
    self.vertical_interpolation_order = vertical_interpolation_order
    self.monotone_tracers = monotone_tracers
    self.monotone_dynamics = monotone_dynamics
    self.nodal_tracers = nodal_tracers_tuple
    self.departure_iterations = departure_iterations
    self.terrain_smoothed_log_sp = terrain_smoothed_log_sp

  def _equation_kwargs(self) -> dict[str, object]:
    return super()._equation_kwargs() | {
        'coriolis_mode': self.coriolis_mode,
        'interpolation_order': self.interpolation_order,
        'vertical_interpolation_order': self.vertical_interpolation_order,
        'monotone_tracers': self.monotone_tracers,
        'monotone_dynamics': self.monotone_dynamics,
        'nodal_tracers': self.nodal_tracers,
        'departure_iterations': self.departure_iterations,
        'terrain_smoothed_log_sp': self.terrain_smoothed_log_sp,
    }

  def nonadvective_terms(
      self, state: dict[str, cx.Field]
  ) -> dict[str, cx.Field]:
    """Returns non-advective explicit tendencies."""
    return self._from_primitive_equations_state(
        self.primitive_equation.nonadvective_terms(
            self._to_primitive_equations_state(state)
        )
    )

  def nodal_velocities(
      self, state: dict[str, cx.Field]
  ) -> primitive_equations.NodalVelocities:
    """Returns nodal velocities that define semi-Lagrangian trajectories."""
    return self.primitive_equation.nodal_velocities(
        self._to_primitive_equations_state(state)
    )

  def departure_points(
      self,
      velocities: primitive_equations.NodalVelocities,
      dt: float,
      initial_guess: primitive_equations.PrimitiveDeparturePoints | None = None,
  ) -> primitive_equations.PrimitiveDeparturePoints:
    """Returns departure points of trajectories arriving at grid points."""
    return self.primitive_equation.departure_points(
        velocities, dt, initial_guess=initial_guess
    )

  def semi_lagrangian_transport(
      self,
      state: dict[str, cx.Field],
      departure: primitive_equations.PrimitiveDeparturePoints,
  ) -> dict[str, cx.Field]:
    """Remaps state-like `state` from `departure` to arrival points."""
    return self._from_primitive_equations_state(
        self.primitive_equation.semi_lagrangian_transport(
            self._to_primitive_equations_state(state), departure
        ),
        is_tendency=False,
    )


class HeldSuarezForcing(time_integrators.ExplicitODE):
  """Equation module for Held-Suarez forcing.

  This module implements Held-Suarez forcing terms, which are often used for
  benchmarking atmospheric models. It includes Rayleigh friction to relax
  horizontal velocities to zero, and Newtonian cooling to relax temperature
  to an equilibrium profile.

  Attributes:
    ylm_map: Spherical harmonics mapping for the horizontal grid.
    levels: Vertical levels coordinate.
    sim_units: Physical constants and units for nondimensionalization.
    reference_temperatures: Reference temperature used for linearization. When
      used with PrimitiveEquations class, this should be the same as the one
      used to initialize the primitive equations class.
    p0: Reference surface pressure used in Held-Suarez forcing.
    sigma_b: Sigma level below which Rayleigh friction is applied.
    kf: Time scale for Rayleigh friction.
    ka: Time scale for Newtonian cooling in the troposphere.
    ks: Time scale for Newtonian cooling in the stratosphere.
    min_t: Minimum equilibrium temperature for Newtonian cooling.
    max_t: Maximum equilibrium temperature for Newtonian cooling.
    d_ty: Temperature diff for equilibrium profile in meridional direction.
    d_thz: Temperature diff for equilibrium profile in vertical direction.
  """

  def __init__(
      self,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      levels: coordinates.SigmaLevels | coordinates.HybridLevels,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float] | cx.Field,
      p0: typing.Quantity = 1e5 * typing.units.pascal,
      sigma_b: float = 0.7,
      kf: typing.Quantity = 1 / (1 * typing.units.day),
      ka: typing.Quantity = 1 / (40 * typing.units.day),
      ks: typing.Quantity = 1 / (4 * typing.units.day),
      min_t: typing.Quantity = 200 * typing.units.kelvin,
      max_t: typing.Quantity = 315 * typing.units.kelvin,
      d_ty: typing.Quantity = 60 * typing.units.kelvin,
      d_thz: typing.Quantity = 10 * typing.units.kelvin,
  ):
    if cx.is_field(reference_temperatures):
      if not cx.contains_dims(reference_temperatures, levels):
        raise ValueError(
            f'reference_temperatures provided on levels different from {levels}'
        )
      assert isinstance(reference_temperatures, cx.Field)  # make pytype happy.
      t_ref_tuple = tuple(float(t) for t in reference_temperatures.data)  # pyrefly: ignore[bad-argument-type]
    else:
      t_ref_tuple = tuple(reference_temperatures)
      reference_temperatures = cx.field(np.array(t_ref_tuple), levels)

    self.ylm_map = ylm_map
    self.levels = levels
    self.sim_units = sim_units
    self.t_ref_tuple = t_ref_tuple
    self.p0 = p0
    self.sigma_b = sigma_b
    self.kf = kf
    self.ka = ka
    self.ks = ks
    self.min_t = min_t
    self.max_t = max_t
    self.d_ty = d_ty
    self.d_thz = d_thz
    if isinstance(levels, coordinates.SigmaLevels):
      self.forcing_cls = held_suarez.HeldSuarezForcingSigma
      self.units_kwargs = {}
    elif isinstance(levels, coordinates.HybridLevels):
      self.forcing_cls = held_suarez.HeldSuarezForcingHybrid
      self.units_kwargs = {'hpa_quantity': typing.units.hPa}
    else:
      raise ValueError(f'Unsupported vertical coordinate system: {levels}')
    self.linearize_transform = get_temperature_linearization_transform(
        ref_temperatures=reference_temperatures
    )
    self.linear_to_absolute_rename = transforms.RenameKeys(
        rename_dict={'temperature_variation': 'temperature'}
    )

  @property
  def forcing(self):
    if isinstance(self.levels, coordinates.SigmaLevels):
      vertical_coords = self.levels.sigma_levels
    else:
      vertical_coords = self.levels.hybrid_levels
    dinosaur_coords = coordinate_systems.CoordinateSystem(
        horizontal=self.ylm_map.dinosaur_grid,
        vertical=vertical_coords,
        spmd_mesh=self.ylm_map.dinosaur_spmd_mesh,
    )
    return self.forcing_cls(
        coords=dinosaur_coords,
        physics_specs=self.sim_units,  # pyrefly: ignore[bad-argument-type]
        reference_temperature=np.asarray(self.t_ref_tuple),
        p0=self.p0,
        sigma_b=self.sigma_b,  # pyrefly: ignore[bad-argument-type]
        kf=self.kf,
        ka=self.ka,
        ks=self.ks,
        minT=self.min_t,
        maxT=self.max_t,
        dTy=self.d_ty,
        dThz=self.d_thz,
        **self.units_kwargs,
    )

  def _to_primitive_equations_state(
      self, inputs: dict[str, cx.Field]
  ) -> primitive_equations.State:
    """Converts a dict of fields to a primitive equations state."""
    inputs = self.linearize_transform(inputs)  # temperature -> variation.
    log_surface_pressure = inputs['log_surface_pressure'].data[np.newaxis]
    return primitive_equations.State(
        divergence=inputs['divergence'].data,  # pyrefly: ignore[bad-argument-type]
        vorticity=inputs['vorticity'].data,  # pyrefly: ignore[bad-argument-type]
        temperature_variation=inputs['temperature_variation'].data,  # pyrefly: ignore[bad-argument-type]
        log_surface_pressure=log_surface_pressure,  # pyrefly: ignore[bad-argument-type]
    )

  def _from_primitive_equations_state(
      self, state: primitive_equations.State
  ) -> dict[str, cx.Field]:
    levels, ylm_grid = self.levels, self.ylm_map.modal_grid
    volume_field_names = ['divergence', 'vorticity', 'temperature_variation']
    volume_fields = {
        k: cx.field(getattr(state, k), levels, ylm_grid)
        for k in volume_field_names
    }
    volume_fields = self.linear_to_absolute_rename(volume_fields)
    lsp = cx.field(jnp.squeeze(state.log_surface_pressure, axis=0), ylm_grid)  # pyrefly: ignore[bad-argument-type]
    return volume_fields | {'log_surface_pressure': lsp}

  def explicit_terms(
      self, state: primitive_equations.StateWithTime
  ) -> primitive_equations.StateWithTime:
    return self._from_primitive_equations_state(  # pyrefly: ignore[bad-return]
        self.forcing.explicit_terms(self._to_primitive_equations_state(state))  # pyrefly: ignore[bad-argument-type]
    )
