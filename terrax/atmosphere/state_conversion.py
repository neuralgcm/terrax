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

"""Helpers for converting between different atmospheric states."""

import functools
from typing import Literal

import coordax as cx
from dinosaur import hybrid_coordinates
from dinosaur import primitive_equations as dinosaur_primitive_equations
import jax.numpy as jnp
from terrax.atmosphere import interpolators
from terrax.core import coordinates
from terrax.core import orographies
from terrax.core import spherical_harmonics
from terrax.core import typing
from terrax.core import units


# Standard atmosphere lapse rate in K/m, used for below-ground extrapolation.
STANDARD_LAPSE_RATE = 0.0065


def get_geopotential(
    inputs: dict[str, cx.Field],
    orography: orographies.Orography,
    sim_units: units.SimUnits,
    surface_pressure: cx.Field | None = None,
):
  """Computes geopotential from temperature and moisture species."""
  temperature = inputs['temperature']
  specific_humidity = inputs['specific_humidity']
  clouds = None
  if (
      'specific_cloud_ice_water_content' in inputs
      and 'specific_cloud_liquid_water_content' in inputs
  ):
    clouds = (
        inputs['specific_cloud_ice_water_content'].data  # pyrefly: ignore[unsupported-operation]
        + inputs['specific_cloud_liquid_water_content'].data
    )
  levels = cx.coords.extract(
      temperature.coordinate,
      (coordinates.SigmaLevels, coordinates.HybridLevels),
  )
  if isinstance(levels, coordinates.SigmaLevels):
    # TODO(dkochkov): Simplify and generalize this function in dinosaur, also
    # consider exposing this function elsewhere in the codebase.
    dino_get_geopotential = functools.partial(
        dinosaur_primitive_equations.get_geopotential_with_moisture,
        nodal_orography=orography.nodal_orography.data,  # pyrefly: ignore[bad-argument-type]
        coordinates=levels.sigma_levels,
        gravity_acceleration=sim_units.gravity_acceleration,
        ideal_gas_constant=sim_units.ideal_gas_constant,
        water_vapor_gas_constant=sim_units.water_vapor_gas_constant,
    )
    geopotential = dino_get_geopotential(
        temperature=temperature.data,  # pyrefly: ignore[bad-argument-type]
        specific_humidity=specific_humidity.data,  # pyrefly: ignore[bad-argument-type]
        clouds=clouds,
    )
  elif isinstance(levels, coordinates.HybridLevels):
    a, b = levels.a_boundaries, levels.b_boundaries
    a_nondim = sim_units.nondimensionalize(a * typing.units.hPa)
    nondim_hybrid_levels = hybrid_coordinates.HybridCoordinates(a_nondim, b)  # pyrefly: ignore[bad-argument-type]
    dino_get_geopotential = functools.partial(
        dinosaur_primitive_equations.get_geopotential_on_hybrid,
        nodal_orography=orography.nodal_orography.data,  # pyrefly: ignore[bad-argument-type]
        coordinates=nondim_hybrid_levels,
        gravity_acceleration=sim_units.gravity_acceleration,
        ideal_gas_constant=sim_units.ideal_gas_constant,
        water_vapor_gas_constant=sim_units.water_vapor_gas_constant,
    )
    if surface_pressure is None:
      raise ValueError(
          'Missing `surface_pressure` in inputs, needed for geopotential on'
          ' hybrid levels.'
      )
    geopotential = dino_get_geopotential(
        temperature=temperature.data,  # pyrefly: ignore[bad-argument-type]
        specific_humidity=specific_humidity.data,  # pyrefly: ignore[bad-argument-type]
        clouds=clouds,
        surface_pressure=jnp.expand_dims(surface_pressure.data, axis=0),  # pyrefly: ignore[bad-argument-type]
    )
  else:
    raise ValueError(
        'Expected exactly one sigma or hybrid level in'
        f' {temperature.coordinate}, got {levels}'
    )

  coord = temperature.coordinate
  geopotential = cx.field(geopotential, coord)
  return geopotential


def uvtz_to_primitive_equations(
    inputs: dict[str, cx.Field],
    levels: coordinates.SigmaLevels | coordinates.HybridLevels,
    orography: orographies.Orography,
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Converts velocity/temperature/geopotential to primitive equations state."""
  if 'geopotential' not in inputs and 'surface_pressure' not in inputs:
    raise ValueError(
        'Missing `geopotential` and `surface_pressure` in source data keys'
        f' {inputs.keys()}, at least one is needed to obtain surface pressure.'
    )

  inputs = inputs.copy()  # avoid mutating inputs.
  geopotential = inputs.pop('geopotential')
  input_levels = {
      geopotential.axes.get(k) for k in ['sigma', 'hybrid', 'pressure']
  }
  input_levels.discard(None)
  if len(input_levels) != 1:
    raise ValueError('expected only one type of level type, got {levels}.')
  [input_levels] = input_levels
  if isinstance(
      input_levels, (coordinates.SigmaLevels, coordinates.HybridLevels)
  ):
    surface_pressure = inputs.pop('surface_pressure')
  else:
    geopotential_at_surface = orography.nodal_orography * sim_units.g
    surface_pressure = interpolators.get_surface_pressure(
        geopotential, geopotential_at_surface, sim_units
    )
  regrid = interpolators.LinearOnPressure(levels, sim_units=sim_units)
  on_levels = regrid(inputs | {'surface_pressure': surface_pressure})
  on_levels['log_surface_pressure'] = cx.cpmap(jnp.log)(surface_pressure)
  return on_levels


def _lapse_rate_below_lowest_level(
    outputs: dict[str, cx.Field],
    temperature: cx.Field,
    geopotential: cx.Field,
    surface_pressure: cx.Field,
    target_levels: (
        coordinates.PressureLevels
        | coordinates.SigmaLevels
        | coordinates.HybridLevels
    ),
    sim_units: units.SimUnits,
) -> dict[str, cx.Field]:
  """Replaces T, z below the lowest source level with lapse-rate profiles.

  Unlimited linear extrapolation in pressure amplifies the temperature
  difference between the two lowest source levels by a factor
  `(p_target - p_lowest) / (p_lowest - p_second_lowest)`, which is very large
  over high terrain when the lowest levels are thin (e.g. for hybrid levels).
  Instead, similar to ECMWF, we extrapolate below the lowest source level
  assuming a standard atmosphere lapse rate starting from the lowest level:

    T(p) = T_low * (p / p_low) ** (R * gamma / g)
    z(p) = z_low + (g / gamma) * T_low * (1 - (p / p_low) ** (R * gamma / g))

  Args:
    outputs: fields interpolated to `target_levels` (must contain `temperature`
      and `geopotential`).
    temperature: temperature on source sigma/hybrid levels.
    geopotential: geopotential on source sigma/hybrid levels.
    surface_pressure: nondimensional surface pressure.
    target_levels: levels to which `outputs` were interpolated.
    sim_units: simulation units.

  Returns:
    `outputs` with temperature and geopotential replaced below the lowest level.
  """
  source_levels = cx.coords.extract(
      temperature.coordinate,
      (coordinates.SigmaLevels, coordinates.HybridLevels),
  )
  lowest = lambda f: f.isel({source_levels: -1})
  p_source = source_levels.pressure_centers(surface_pressure, sim_units)
  p_low = lowest(p_source)
  t_low = lowest(temperature)
  z_low = lowest(geopotential)
  p_target = target_levels.pressure_centers(surface_pressure, sim_units)
  g = sim_units.gravity_acceleration
  gamma = sim_units.nondimensionalize(
      STANDARD_LAPSE_RATE * typing.units.kelvin / typing.units.meter
  )
  exponent = sim_units.ideal_gas_constant * gamma / g

  def _extrapolate(t_interp, z_interp, p, p_low, t_low, z_low):
    ratio_pow = (p / p_low) ** exponent
    below = p > p_low
    t_ext = jnp.where(below, t_low * ratio_pow, t_interp)
    z_ext = jnp.where(
        below, z_low + (g / gamma) * t_low * (1 - ratio_pow), z_interp
    )
    return t_ext, z_ext

  # Interpolated fields are passed first so that outputs follow their dims.
  t_ext, z_ext = cx.cmap(_extrapolate)(
      outputs['temperature'],
      outputs['geopotential'],
      p_target,
      p_low,
      t_low,
      z_low,
  )
  return outputs | {'temperature': t_ext, 'geopotential': z_ext}


def primitive_equations_to_uvtz(
    inputs: dict[str, cx.Field],
    ylm_map: spherical_harmonics.FixedYlmMapping,
    levels: (
        coordinates.PressureLevels
        | coordinates.SigmaLevels
        | coordinates.HybridLevels
    ),
    orography: orographies.Orography,
    sim_units: units.SimUnits,
    include_surface_pressure: bool = False,
    below_ground_extrapolation: Literal['linear', 'lapse_rate'] = 'linear',
    pressure_interpolation_space: Literal['direct', 'log'] = 'direct',
) -> dict[str, cx.Field]:
  """Converts primitive equations state to pressure level representation.

  This function transforms an atmospheric state described in terms of
  temperature variation, divergence, vorticity, surface pressure and tracers
  on sigma levels to wind components, temperature, geopotential and tracers
  on fixed pressure-level coordinates.

  Args:
    inputs: State represented using primitive equations variables.
    ylm_map: Spherical harmonics mapping that defines modal-nodal conversion.
    levels: Vertical levels to interpolate "uvtz, ..." variables to.
    orography: Orography module.
    sim_units: Simulation units object.
    include_surface_pressure: Whether to include surface pressure in the output.
    below_ground_extrapolation: How to extrapolate temperature and geopotential
      to levels below the lowest model level. `linear` extrapolates linearly in
      pressure using the two lowest levels; `lapse_rate` uses a standard
      atmosphere lapse rate starting from the lowest model level.
    pressure_interpolation_space: Whether to interpolate (and extrapolate above
      the top level) linearly in pressure (`direct`) or in log-pressure
      (`log`). See `interpolators.LinearOnPressure.interpolation_space`.

  Returns:
    State represented as zonal, medidional wind, temperature, geopotential and
    tracers interpolated to vertical `levels`.
  """
  inputs = inputs.copy()  # avoid mutating inputs.
  vorticity, divergence = inputs.pop('vorticity'), inputs.pop('divergence')
  u, v = spherical_harmonics.vor_div_to_uv_nodal(vorticity, divergence, ylm_map)
  log_surface_pressure = inputs.pop('log_surface_pressure')
  surface_pressure = cx.cpmap(jnp.exp)(ylm_map.to_nodal(log_surface_pressure))

  nodal_inputs = ylm_map.to_nodal(inputs)  # includes temperature and tracers.
  geopotential = get_geopotential(
      nodal_inputs, orography, sim_units, surface_pressure=surface_pressure
  )
  temperature = nodal_inputs.pop('temperature')

  surface_pressure, temperature, geopotential, nodal_inputs = (
      ylm_map.mesh.with_sharding_constraint(
          (surface_pressure, temperature, geopotential, nodal_inputs),
          ('physics', 'dycore'),
      )
  )

  regrid_constant = interpolators.LinearOnPressure(
      levels,
      'constant',
      sim_units=sim_units,
      interpolation_space=pressure_interpolation_space,
  )
  regrid_linear = interpolators.LinearOnPressure(
      levels,
      'linear',
      sim_units=sim_units,
      interpolation_space=pressure_interpolation_space,
  )
  # closest regridding options to those used in ERA5.
  # use constant extrapolation for `u, v, tracers`.
  # use linear extrapolation for `z, t`.
  # google reference: http://shortn/_X09ZAU1jsx.
  ps_dict = {'surface_pressure': surface_pressure}
  winds = {'u_component_of_wind': u, 'v_component_of_wind': v}
  outputs = regrid_constant(winds | nodal_inputs | ps_dict)
  outputs |= regrid_linear(
      {'temperature': temperature, 'geopotential': geopotential} | ps_dict
  )
  if below_ground_extrapolation == 'lapse_rate':
    outputs = _lapse_rate_below_lowest_level(
        outputs, temperature, geopotential, surface_pressure, levels, sim_units
    )
  elif below_ground_extrapolation != 'linear':
    raise ValueError(
        f'Unknown {below_ground_extrapolation=}, expected linear or lapse_rate.'
    )
  if include_surface_pressure:
    outputs['surface_pressure'] = surface_pressure
  return outputs
