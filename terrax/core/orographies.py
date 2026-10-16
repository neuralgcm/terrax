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

"""Modules that hold orographic data."""

import coordax as cx
from flax import nnx
import jax.numpy as jnp
from terrax.core import coordinates
from terrax.core import interpolators
from terrax.core import scales
from terrax.core import spatial_filters
from terrax.core import spherical_harmonics
from terrax.core import units
from terrax.core import xarray_utils
import xarray


_OROGRAPHY_KEY = 'orography'
_GEOPOTENTIAL_AT_SURFACE_KEY = 'geopotential_at_surface'


def _nodal_orography_from_xarray(
    dataset: xarray.Dataset,
    sim_units: units.SimUnits,
) -> cx.Field:
  """Returns nondimensionalized nodal orography from `dataset`.

  Orography is read from the `orography` variable if present, and otherwise
  derived from `geopotential_at_surface`. Units are taken from the `units`
  attribute of the variable and default to meters.

  Args:
    dataset: dataset with `orography` or `geopotential_at_surface` variables
      on a `longitude`, `latitude` grid.
    sim_units: units used to nondimensionalize orography.

  Returns:
    Nondimensionalized orography on the grid of `dataset`.
  """
  if _OROGRAPHY_KEY in dataset:
    orography = dataset[_OROGRAPHY_KEY]
  else:
    geopotential = dataset[_GEOPOTENTIAL_AT_SURFACE_KEY]
    orography = geopotential / scales.GRAVITY_ACCELERATION.magnitude
  orography = orography.transpose('longitude', 'latitude')
  data_units = units.parse_units(orography.attrs.get('units', 'meter'))
  orography = orography.copy(
      data=sim_units.nondimensionalize(orography.values * data_units)
  )
  return xarray_utils.from_xarray(orography)


class OrographyVariable(nnx.Variable):
  """Variable class for orography data."""


class ModalOrography(nnx.Module):
  """Orogrphay module that provoides elevation in modal representation."""

  def __init__(
      self,
      *,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    self.ylm_map = ylm_map
    modal_shape_1d = (ylm_map.ylm_grid.fields['mask'].data.sum(),)
    self.orography = OrographyVariable(initializer(rngs, modal_shape_1d))

  @property
  def nodal_orography(self) -> cx.Field:
    return self.ylm_map.to_nodal(self.modal_orography)

  @property
  def modal_orography(self) -> cx.Field:
    """Returns orography converted to modal representation with filtering."""
    ylm_grid = self.ylm_map.modal_grid
    mask = ylm_grid.fields['mask']
    modal_orography_2d = jnp.zeros(ylm_grid.shape)
    return cx.field(
        modal_orography_2d.at[mask.data].set(self.orography[...]), ylm_grid
    )

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      **kwargs,
  ):
    """Updates ``self.orography`` with filtered orography from dataset."""
    data_ylm_map = kwargs['data_ylm_map']
    sim_units = kwargs['sim_units']
    spatial_filter = kwargs.get('spatial_filter', None)

    if spatial_filter is None:
      spatial_filter = lambda x: x
    nodal_orography = _nodal_orography_from_xarray(dataset, sim_units)
    data_grid = nodal_orography.coordinate
    if data_grid != data_ylm_map.nodal_grid:
      raise ValueError(
          f'{data_grid=} does not match {data_ylm_map.nodal_grid=}.'
      )
    if not isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      nodal_orography = spatial_filter(nodal_orography)
    modal_orography = data_ylm_map.to_modal(nodal_orography)
    interpolator = interpolators.SpectralRegridder(self.ylm_map.modal_grid)
    modal_orography = interpolator(modal_orography)
    if isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      modal_orography = spatial_filter.filter_modal(modal_orography)
    modal_orography_data = modal_orography.data
    self.orography.set_value(modal_orography_data[
        self.ylm_map.modal_grid.fields['mask'].data
    ])


class ModalOrographyWithCorrection(ModalOrography):
  """ModalOrography module with learned correction in modal representation."""

  def __init__(
      self,
      *,
      ylm_map: spherical_harmonics.FixedYlmMapping,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      correction_scale: float,
      correction_param_type: nnx.Param = nnx.Param,
      correction_initializer: nnx.initializers.Initializer = (
          nnx.initializers.truncated_normal()
      ),
      rngs: nnx.Rngs,
  ):
    super().__init__(ylm_map=ylm_map, initializer=initializer, rngs=rngs)
    self.correction_scale = correction_scale
    self.correction = correction_param_type(
        correction_initializer(rngs.params(), self.orography.shape)
    )

  @property
  def modal_orography(self) -> cx.Field:
    """Returns orography converted to modal representation with filtering."""
    ylm_grid = self.ylm_map.modal_grid
    mask = ylm_grid.fields['mask']
    modal_orography_2d = jnp.zeros(mask.shape)
    modal_orography_1d = (
        self.orography[...] + self.correction[...] * self.correction_scale
    )
    return cx.field(
        modal_orography_2d.at[mask.data].set(modal_orography_1d), ylm_grid
    )


class Orography(nnx.Module):
  """Orography module that provides elevation in real space."""

  def __init__(
      self,
      *,
      grid: coordinates.LonLatGrid,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    self.grid = grid
    self.orography = OrographyVariable(initializer(rngs, grid.shape))

  @property
  def nodal_orography(self) -> cx.Field:
    return cx.field(self.orography[...], self.grid)

  def update_from_xarray(
      self,
      dataset: xarray.Dataset,
      **kwargs,
  ):
    """Updates ``self.orography`` with filtered orography from dataset."""
    sim_units = kwargs['sim_units']
    spatial_filter = kwargs.get('spatial_filter', None)

    if spatial_filter is None:
      spatial_filter = lambda x: x
    nodal_orography = _nodal_orography_from_xarray(dataset, sim_units)
    data_grid = nodal_orography.coordinate
    nodal_orography = spatial_filter(nodal_orography)
    if data_grid != self.grid:
      raise ValueError(f'{data_grid=} does not match {self.grid=}.')
    self.orography.set_value(nodal_orography.data)
