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

"""Coordinate systems that describe how data & model states are discretized."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any, Iterable, Literal, Self, Sequence, TYPE_CHECKING, cast

import coordax as cx
import coordax.experimental
from dinosaur import fourier
from dinosaur import hybrid_coordinates
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import typing
from terrax.core import units
import treescope
import xarray

if TYPE_CHECKING:
  # import only under TYPE_CHECKING to avoid circular dependency
  # pylint: disable=g-bad-import-order
  # TODO(dkochkov): consider moving UnshardedCoordinate to coordax.
  from terrax.core import parallelism


SphericalHarmonicsImpl = spherical_harmonic.SphericalHarmonicsImpl
RealSphericalHarmonics = spherical_harmonic.RealSphericalHarmonics
FastSphericalHarmonics = spherical_harmonic.FastSphericalHarmonics
P = jax.sharding.PartitionSpec


SphericalHarmonicsMethodNames = Literal['real', 'fast']
SPHERICAL_HARMONICS_METHODS = {
    'real': RealSphericalHarmonics,
    'fast': FastSphericalHarmonics,
}
# factor for globally constant conversion from real to YLM basis.
_YLM_CONSTANT_NORMALIZATION_FACTOR = 3.5449077


def _in_treescope_abbreviation_mode() -> bool:
  """Returns True if treescope.abbreviation is set by context or globally."""
  return treescope.abbreviation_threshold.get() is not None


@dataclasses.dataclass(frozen=True)
class ArrayKey:
  """Wrapper for a numpy array to make it hashable."""

  value: np.ndarray

  def __eq__(self, other):
    return (
        isinstance(self, ArrayKey)
        and self.value.dtype == other.value.dtype
        and self.value.shape == other.value.shape
        and (self.value == other.value).all()
    )

  def __hash__(self) -> int:
    return hash((self.value.shape, self.value.tobytes()))


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class TimeDelta(cx.Coordinate):
  """Coordinates that discretize data along static relative time."""

  deltas: np.ndarray

  def __post_init__(self):
    deltas = np.asarray(self.deltas)
    if not np.issubdtype(deltas.dtype, np.timedelta64):
      raise ValueError(f'deltas must be a timedelta array, got {deltas.dtype=}')
    object.__setattr__(self, 'deltas', deltas.astype('timedelta64[s]'))

  @property
  def dims(self) -> tuple[str, ...]:
    return ('timedelta',)

  @property
  def shape(self):
    return self.deltas.shape

  @property
  def fields(self):
    return {'timedelta': cx.field(self.deltas, self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    return coordax.experimental.map_indexers_using_ticks(
        self, indexers, ticks_are_sorted=True, method=method
    )

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    key = self.dims[0] if self.dims[0] in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return cx.Scalar()

    return TimeDelta(self.deltas[indexer])

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    variables['timedelta'] = xarray.Variable(('timedelta',), self.deltas)
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'timedelta':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "timedelta"')
    if 'timedelta' not in coords:
      return cx.coords.NoCoordinateMatch(
          'no associated coordinate for timedelta'
      )

    data = coords['timedelta'].data
    if data.ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'timedelta coordinate is not a 1D array'
      )

    if not np.issubdtype(data.dtype, np.timedelta64):
      return cx.coords.NoCoordinateMatch(
          f'data must be a timedelta array, got {data.dtype=}'
      )

    return cls(deltas=data)

  def _components(self):
    return (ArrayKey(self.deltas),)

  def __eq__(self, other):
    return (
        isinstance(other, TimeDelta)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __getitem__(self, key: slice) -> Self:
    return type(self)(self.deltas[key])

  def __repr__(self):
    if _in_treescope_abbreviation_mode():
      return treescope.render_to_text(self)
    else:
      with treescope.abbreviation_threshold.set_scoped(1):
        with treescope.using_expansion_strategy(9, 80):
          return treescope.render_to_text(self)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    """Treescope handler for Field."""
    dts = [str(x) for x in self.deltas.tolist()]
    if len(dts) < 6:
      deltas = '[' + ', '.join(dts) + ']'
    else:
      deltas = '[' + ', '.join(dts[:2])
      deltas += ', ..., '
      deltas += ', '.join(dts[-2:]) + ']'
    heading = f'<{type(self).__name__}'
    return treescope.rendering_parts.siblings(
        heading, treescope.rendering_parts.text(deltas), '>'
    )


#
# Grid-like and spherical harmonic coordinate systems
#


def _mesh_to_dinosaur_spmd_mesh(
    dims: tuple[str, ...],
    mesh: parallelism.Mesh | None = None,
    partition_schema_key: str | None = None,
) -> jax.sharding.Mesh | None:
  """Returns spmd_mesh in dinosaur.spherical_harmonic.Grid format."""
  if mesh is not None:
    dim_to_axes = {d: ax for d, ax in zip(dims, ['x', 'y'])}
    # TODO(dkochkov): modify dinosaur.spherical_harmonic.Grid to not
    # assume level dimension in implementation details.
    dim_to_axes['level'] = 'z'
    return mesh.rearrange_spmd_mesh(partition_schema_key, dim_to_axes)
  return None


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LonLatGrid(cx.Coordinate):
  """Coordinates that discretize data as point values on lon-lat grid."""

  longitude_nodes: int
  latitude_nodes: int
  latitude_spacing: str = 'gauss'
  longitude_offset: float = 0.0
  lon_lat_padding: tuple[int, int] = (0, 0)

  @property
  def _ylm_grid(self) -> spherical_harmonic.Grid:
    """Unpadded spherical harmonic grid with matching nodal values."""
    return spherical_harmonic.Grid(
        longitude_wavenumbers=0,
        total_wavenumbers=0,
        longitude_nodes=self.longitude_nodes,
        latitude_nodes=self.latitude_nodes,
        latitude_spacing=self.latitude_spacing,
        longitude_offset=self.longitude_offset,
    )

  @property
  def dims(self) -> tuple[str, ...]:
    return ('longitude', 'latitude')

  @property
  def shape(self):
    unpadded_shape = (self.longitude_nodes, self.latitude_nodes)
    return tuple(x + y for x, y in zip(unpadded_shape, self.lon_lat_padding))

  @property
  def fields(self):
    lon_pad, lat_pad = self.lon_lat_padding
    lons = np.rad2deg(np.pad(self._ylm_grid.longitudes, (0, lon_pad)))
    lats = np.rad2deg(np.pad(self._ylm_grid.latitudes, (0, lat_pad)))
    return {
        'longitude': cx.field(lons, cx.SelectedAxis(self, axis=0)),
        'latitude': cx.field(lats, cx.SelectedAxis(self, axis=1)),
    }

  @functools.cached_property
  def cos_lat(self) -> cx.Field:
    padded_lats = np.pad(self._ylm_grid.latitudes, (0, self.lon_lat_padding[1]))
    return cx.field(np.cos(padded_lats), cx.SelectedAxis(self, axis=1))

  def _remap_indexers_to_labeled_axes(
      self, indexers: dict[str | cx.Coordinate, Any]
  ) -> tuple[
      dict[str | cx.Coordinate, Any], tuple[cx.LabeledAxis, cx.LabeledAxis]
  ]:
    """Remaps indexers to labeled axes."""
    longitudes_axis = cx.LabeledAxis('longitude', self.fields['longitude'].data)
    latitudes_axis = cx.LabeledAxis('latitude', self.fields['latitude'].data)
    remap: dict[str | cx.Coordinate, cx.Coordinate] = {
        self.axes[0]: longitudes_axis,
        self.axes[1]: latitudes_axis,
    }
    axes = (longitudes_axis, latitudes_axis)
    indexers = {  # swap self.axes to LabeledAxes for both keys and idx values.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    return indexers, axes

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: coordax.experimental.SelMethod = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    # When slicing by longitude/latitude, we remap respective axes to
    # LabeledAxis, since sliced LonLatGrid is not necessarily a LonLatGrid.
    # This allows us to reuse 1d indexing logic. When returning, keys must be
    # mapped back to ensure that associated dimensions can be identified.
    indexers, (lon_ax, lat_ax) = self._remap_indexers_to_labeled_axes(indexers)
    mapped, consumed = {}, set()
    if 'longitude' in indexers or lon_ax in indexers:
      sub_mapped, sub_consumed = lon_ax.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)

    if 'latitude' in indexers or lat_ax in indexers:
      sub_mapped, sub_consumed = lat_ax.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)

    # Remap LabeledAxes keys back to self.axes to ensure keys consistency.
    from_labeled: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        l_ax: ax for l_ax, ax in zip((lon_ax, lat_ax), self.axes)
    }
    consumed = {from_labeled.get(ax, ax) for ax in consumed}
    mapped = {from_labeled.get(k, k): v for k, v in mapped.items()}
    return mapped, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    indexers, l_axes = self._remap_indexers_to_labeled_axes(indexers)
    l_isel = cx.coords.compose(*l_axes).isel(indexers)
    # If selected axis was not sliced, we preserve the original axis.
    labeled_to_orig: dict[cx.Coordinate, cx.Coordinate] = {
        ax: orig for ax, orig in zip(l_axes, self.axes)
    }
    return cx.coords.compose(*[labeled_to_orig.get(c, c) for c in l_isel.axes])

  def integrate(
      self,
      x: cx.Field,
      dims: str | Sequence[str] = ('longitude', 'latitude'),
      radius: float = 1.0,
  ) -> cx.Field:
    """Integrates `x` over dimensions in `dims` using quadrature.

    Args:
      x: Field to integrate.
      dims: Dimensions to integrate. Can be 'longitude', 'latitude', or a
        sequence of thereof.
      radius: Radius of sphere, defaults to 1.0.

    Returns:
      Field with all or a subset of dimensions of LonLatGrid integrated out.
    """
    _, lon_w = fourier.quadrature_nodes(self.longitude_nodes)
    _, lat_w = spherical_harmonic.get_latitude_nodes(
        self.latitude_nodes, self.latitude_spacing
    )

    def _integrate_lon(array):
      return radius**2 * jnp.sum(array[: self.longitude_nodes] * lon_w)

    def _integrate_lat(array):
      return radius**2 * jnp.sum(array[: self.latitude_nodes] * lat_w)

    def _integrate(array):
      unpadded = array[: self.longitude_nodes, : self.latitude_nodes]
      return radius**2 * jnp.sum(unpadded * lat_w * lon_w)

    dims_tuple = tuple(sorted({dims} if isinstance(dims, str) else set(dims)))

    match dims_tuple:
      case ('longitude',):
        return cx.cmap(_integrate_lon)(x.untag(cx.SelectedAxis(self, axis=0)))
      case ('latitude',):
        return cx.cmap(_integrate_lat)(x.untag(cx.SelectedAxis(self, axis=1)))
      case ('latitude', 'longitude'):
        return cx.cmap(_integrate)(x.untag(self))
      case _:
        raise ValueError(
            'dims must be "longitude", "latitude", or a sequence '
            f'thereof, got {dims}'
        )

  def mean(
      self,
      x: cx.Field,
      dims: str | Sequence[str] = ('longitude', 'latitude'),
  ) -> cx.Field:
    """Computes mean of `x` over dimensions in `dims`.

    Args:
      x: Field to average.
      dims: Dimensions to average over. Can be 'longitude', 'latitude', or a
        sequence of thereof.

    Returns:
      Field with all or a subset of dimensions of LonLatGrid averaged out.
    """
    integral = self.integrate(x, dims, radius=1.0)
    dims_tuple = tuple(sorted({dims} if isinstance(dims, str) else set(dims)))
    match dims_tuple:
      case ('longitude',):
        return integral / (2 * np.pi)
      case ('latitude',):
        return integral / 2.0
      case ('latitude', 'longitude'):
        return integral / (4 * np.pi)
      case _:
        raise ValueError(
            'dims must be "longitude", "latitude", or a sequence '
            f'thereof, got {dims}'
        )

  def __lt__(self, other: LonLatGrid) -> bool:
    """Custom comparison operator for sorting LonLatGrids."""

    # Implementing __lt__ enables LonLatGrid to be keys in a dict
    # and be compatible with jax.tree. operations.
    def _to_tuple(x):
      xt = (
          x.longitude_nodes,
          x.latitude_nodes,
          x.latitude_spacing,
          x.longitude_offset,
          x.lon_lat_padding,
      )
      return xt

    if not isinstance(other, LonLatGrid):
      return NotImplemented
    self_compare_values_in_order = _to_tuple(self)
    other_compare_values_in_order = _to_tuple(other)
    return self_compare_values_in_order < other_compare_values_in_order

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    return cls(
        longitude_nodes=ylm_grid.longitude_nodes,
        latitude_nodes=ylm_grid.latitude_nodes,
        latitude_spacing=ylm_grid.latitude_spacing,
        longitude_offset=ylm_grid.longitude_offset,
        lon_lat_padding=ylm_grid.nodal_padding,
    )

  @classmethod
  def construct(
      cls,
      gaussian_nodes: int | None = None,
      latitude_spacing: str = 'gauss',
      longitude_offset: float = 0.0,
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
      *,
      longitude_nodes: int | None = None,
      latitude_nodes: int | None = None,
  ) -> LonLatGrid:
    """Constructs a `LonLatGrid` with specified number of nodes.

    Args:
      gaussian_nodes: number of nodes between the equator and a pole. Mutually
        exclusive with `(longitude_nodes, latitude_nodes)`.
      latitude_spacing: either 'gauss', 'equiangular', 'equiangular_with_poles'.
        This determines the spacing of grid points in the latitudinal direction.
      longitude_offset: the value of the first longitude node, in radians.
      mesh: optional Mesh that specifies necessary grid padding.
      partition_schema_key: key indicating a partition schema on `mesh` to infer
        padding details. Used only if an appropriate `mesh` is passed in.
      longitude_nodes: explicit number of longitude nodes; must be specified
        together with `latitude_nodes` when `gaussian_nodes` is omitted.
      latitude_nodes: explicit number of latitude nodes; must be specified
        together with `longitude_nodes` when `gaussian_nodes` is omitted.

    Returns:
      Constructed LonLatGrid object.
    """
    if gaussian_nodes is not None:
      if longitude_nodes is not None or latitude_nodes is not None:
        raise ValueError(
            '`gaussian_nodes` and `(longitude_nodes, latitude_nodes)`, are'
            ' mutually exclusive, please specify only one.'
        )
      longitude_nodes = 4 * gaussian_nodes
      latitude_nodes = 2 * gaussian_nodes
    elif longitude_nodes is None or latitude_nodes is None:
      raise ValueError(
          'Either `gaussian_nodes` or both `(longitude_nodes, latitude_nodes)`'
          ' must be specified.'
      )
    dims = ('longitude', 'latitude')
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=0,
        total_wavenumbers=0,
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=FastSphericalHarmonics,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  @classmethod
  def construct_equiangular_linear(
      cls,
      max_wavenumber: int,
      longitude_offset: float = 0.0,
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> LonLatGrid:
    """Constructs an equiangular linear `LonLatGrid`.

    This creates a grid with equiangular_with_poles latitude spacing, where
    latitude nodes are uniformly spaced and include both poles. The grid has
    2 * max_wavenumber longitude nodes and max_wavenumber + 1 latitude nodes.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      longitude_offset: the value of the first longitude node, in radians.
      mesh: optional Mesh that specifies necessary grid padding.
      partition_schema_key: key indicating a partition schema on `mesh` to infer
        padding details. Used only if an appropriate `mesh` is passed in.

    Returns:
      Constructed LonLatGrid object.
    """
    dims = ('longitude', 'latitude')
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=0,
        total_wavenumbers=0,
        longitude_nodes=2 * max_wavenumber,
        latitude_nodes=max_wavenumber + 1,
        latitude_spacing='equiangular_with_poles',
        longitude_offset=longitude_offset,
        spherical_harmonics_impl=FastSphericalHarmonics,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=16, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=24, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=32, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=64, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=80, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=90, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=128, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=160, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=256, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=320, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=16, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=24, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=32, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=48, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=64, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=80, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=90, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=128, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=320, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> LonLatGrid:
    return cls.construct(gaussian_nodes=640, **kwargs)

  # EL* grids use equiangular linear (equiangular_with_poles) latitude spacing.
  # These grids include both poles and have uniform latitude spacing.

  @classmethod
  def EL120(cls, **kwargs) -> LonLatGrid:
    """Equiangular linear grid resolving up to wavenumber 120."""
    return cls.construct_equiangular_linear(max_wavenumber=120, **kwargs)

  @classmethod
  def EL180(cls, **kwargs) -> LonLatGrid:
    """Equiangular linear grid resolving up to wavenumber 180 (1 deg)."""
    return cls.construct_equiangular_linear(max_wavenumber=180, **kwargs)

  @classmethod
  def EL720(cls, **kwargs) -> LonLatGrid:
    """Equiangular linear grid resolving up to wavenumber 720 (0.25 deg)."""
    return cls.construct_equiangular_linear(max_wavenumber=720, **kwargs)

  @classmethod
  def EL1800(cls, **kwargs) -> LonLatGrid:
    """Equiangular linear grid resolving up to wavenumber 1800 (0.1 deg)."""
    return cls.construct_equiangular_linear(max_wavenumber=1800, **kwargs)

  # F* grids are regular Gaussian grids named by the number of latitude nodes
  # between a pole and the equator, i.e. F<N> has shape (4N, 2N). This follows
  # ECMWF's "F<N>" naming for full (regular) Gaussian grids. They are named by
  # nodal resolution rather than by spectral truncation.
  #
  # Correspondence with spectral names: F16 == T21 == TL31, F24 == T31 == TL47,
  # F32 == T42 == TL63, F48 == TL95, F64 == T85 == TL127.
  #
  # TODO(dkochkov): Consider upstreaming coarse grids to dinosaur.

  @classmethod
  def F2(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 8 x 4 nodes (~45 deg)."""
    return cls.construct(gaussian_nodes=2, **kwargs)

  @classmethod
  def F4(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 16 x 8 nodes (~22.5 deg)."""
    return cls.construct(gaussian_nodes=4, **kwargs)

  @classmethod
  def F6(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 24 x 12 nodes (~15 deg)."""
    return cls.construct(gaussian_nodes=6, **kwargs)

  @classmethod
  def F8(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 32 x 16 nodes (~11.25 deg)."""
    return cls.construct(gaussian_nodes=8, **kwargs)

  @classmethod
  def F12(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 48 x 24 nodes (~7.5 deg)."""
    return cls.construct(gaussian_nodes=12, **kwargs)

  @classmethod
  def F16(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 64 x 32 nodes (~5.6 deg), same as T21."""
    return cls.construct(gaussian_nodes=16, **kwargs)

  @classmethod
  def F24(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 96 x 48 nodes (~3.75 deg), same as T31."""
    return cls.construct(gaussian_nodes=24, **kwargs)

  @classmethod
  def F32(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 128 x 64 nodes (~2.8 deg), same as TL63."""
    return cls.construct(gaussian_nodes=32, **kwargs)

  @classmethod
  def F48(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 192 x 96 nodes (~1.9 deg), same as TL95."""
    return cls.construct(gaussian_nodes=48, **kwargs)

  @classmethod
  def F64(cls, **kwargs) -> LonLatGrid:
    """Gaussian grid with 256 x 128 nodes (~1.4 deg), same as TL127."""
    return cls.construct(gaussian_nodes=64, **kwargs)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        lon_lat_padding=self.lon_lat_padding,
    )
    variables['longitude'].attrs = metadata
    variables['latitude'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    if dims[:2] != ('longitude', 'latitude'):
      return cx.coords.NoCoordinateMatch(
          "leading dimensions are not ('longitude', 'latitude')"
      )

    if coords['longitude'].dims != ('longitude',):
      return cx.coords.NoCoordinateMatch('longitude is not a 1D coordinate')

    if coords['latitude'].dims != ('latitude',):
      return cx.coords.NoCoordinateMatch('latitude is not a 1D coordinate')

    lon = coords['longitude'].data
    lat = coords['latitude'].data

    if lon.max() < 2 * np.pi:
      return cx.coords.NoCoordinateMatch(
          f'expected longitude values in degrees, got {lon}'
      )

    if np.allclose(np.diff(lat), lat[1] - lat[0]):
      if np.isclose(max(lat), 90.0):
        latitude_spacing = 'equiangular_with_poles'
      else:
        latitude_spacing = 'equiangular'
    else:
      latitude_spacing = 'gauss'

    longitude_offset = float(np.deg2rad(coords['longitude'].data[0]))
    longitude_nodes = coords.sizes['longitude']
    latitude_nodes = coords.sizes['latitude']

    lon_lat_padding = coords['longitude'].attrs.get('lon_lat_padding', (0, 0))
    result = cls(
        longitude_nodes=longitude_nodes,
        latitude_nodes=latitude_nodes,
        latitude_spacing=latitude_spacing,
        longitude_offset=longitude_offset,
        lon_lat_padding=(int(lon_lat_padding[0]), int(lon_lat_padding[1])),
    )
    result_lat = np.rad2deg(result._ylm_grid.latitudes)
    if not np.allclose(result_lat, lat, atol=1e-3):
      return cx.coords.NoCoordinateMatch(
          f'inferred latitudes with spacing={latitude_spacing!r} do not '
          f' match coordinate data: {result_lat} vs {lat}'
      )

    result_lon = np.rad2deg(result._ylm_grid.longitudes)
    if not np.allclose(result_lon, lon, atol=1e-3):
      return cx.coords.NoCoordinateMatch(
          'inferred longitudes do not match coordinate data:'
          f' {result_lon} vs {lon}'
      )

    return result


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SphericalHarmonicGrid(cx.Coordinate):
  """Coordinates that discretize data as spherical harmonic coefficients."""

  longitude_wavenumbers: int
  total_wavenumbers: int
  spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast'
  wavenumber_padding: tuple[int, int] = (0, 0)

  @property
  def _ylm_grid(self) -> spherical_harmonic.Grid:
    method = SPHERICAL_HARMONICS_METHODS[self.spherical_harmonics_method]
    return spherical_harmonic.Grid(
        longitude_wavenumbers=self.longitude_wavenumbers,
        total_wavenumbers=self.total_wavenumbers,
        longitude_nodes=0,
        latitude_nodes=0,
        latitude_spacing='gauss',
        longitude_offset=0.0,
        spherical_harmonics_impl=method,
    )

  @property
  def dims(self) -> tuple[str, ...]:
    return ('longitude_wavenumber', 'total_wavenumber')

  @property
  def shape(self) -> tuple[int, ...]:
    unpadded_shape = self._ylm_grid.modal_shape
    return tuple(x + y for x, y in zip(unpadded_shape, self.wavenumber_padding))

  @property
  def fields(self):
    unpadded_ms, unpadded_ls = self._ylm_grid.modal_axes
    m_pad, l_pad = self.wavenumber_padding
    ms = np.pad(unpadded_ms, (0, m_pad))
    ls = np.pad(unpadded_ls, (0, l_pad))
    axes_fields = {
        k: cx.field(v, cx.SelectedAxis(self, i))
        for i, (k, v) in enumerate(zip(self.dims, [ms, ls]))
    }
    unpadded_mask = self._ylm_grid.mask
    mask = np.pad(unpadded_mask, ((0, m_pad), (0, l_pad)))
    mask_field = cx.field(mask, self)
    return axes_fields | {'mask': mask_field}

  def _remap_indexers_to_labeled_axes(
      self, indexers: dict[str | cx.Coordinate, Any]
  ) -> tuple[
      dict[str | cx.Coordinate, Any], tuple[cx.LabeledAxis, cx.LabeledAxis]
  ]:
    """Remaps indexers to labeled axes."""
    ms_ax = cx.LabeledAxis(
        'longitude_wavenumber', self.fields['longitude_wavenumber'].data
    )
    ls_ax = cx.LabeledAxis(
        'total_wavenumber', self.fields['total_wavenumber'].data
    )
    remap: dict[str | cx.Coordinate, cx.Coordinate] = {
        self.axes[0]: ms_ax,
        self.axes[1]: ls_ax,
    }
    axes = (ms_ax, ls_ax)
    indexers = {  # swap self.axes to LabeledAxes for both keys and idx values.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    return indexers, axes

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    # When slicing by longitude/total wavenumber, we remap respective axes to
    # LabeledAxis, since sliced SphericalHarmonicGrid is not necessarily a
    # SphericalHarmonicGrid. This allows us to reuse 1d indexing logic. When
    # returning, keys must be mapped back to ensure that associated dimensions
    # can be identified.
    indexers, (ms_ax, ls_ax) = self._remap_indexers_to_labeled_axes(indexers)
    mapped, consumed = {}, set()
    if 'longitude_wavenumber' in indexers or ms_ax in indexers:
      sub_mapped, sub_consumed = ms_ax.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)

    if 'total_wavenumber' in indexers or ls_ax in indexers:
      sub_mapped, sub_consumed = ls_ax.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)

    # Remap LabeledAxes keys back to self.axes to ensure keys consistency.
    from_labeled: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        l_ax: ax for l_ax, ax in zip((ms_ax, ls_ax), self.axes)
    }
    consumed = {from_labeled.get(ax, ax) for ax in consumed}
    mapped = {from_labeled.get(k, k): v for k, v in mapped.items()}
    return mapped, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    indexers, l_axes = self._remap_indexers_to_labeled_axes(indexers)
    l_isel = cx.coords.compose(*l_axes).isel(indexers)
    # If selected axis was not sliced, we preserve the original axis.
    labeled_to_orig: dict[cx.Coordinate, cx.Coordinate] = {
        ax: orig for ax, orig in zip(l_axes, self.axes)
    }
    return cx.coords.compose(*[labeled_to_orig.get(c, c) for c in l_isel.axes])

  def add_constant(
      self,
      x: cx.Field,
      c: float | jax.Array | cx.Field,
  ) -> cx.Field:
    """Adds the constant `c` to the field `x` in the spectral basis."""
    if not cx.is_field(c):
      c = cx.field(jnp.squeeze(c))
    assert isinstance(c, cx.Field)  # make pytype happy.
    if c.positional_shape:
      raise ValueError(
          f'Adding non-scalar constants without axes is not supported. {c=}'
      )
    if any(d in self.dims for d in c.dims):
      raise ValueError(
          f'Coordinate {c} cannot have any of the dimensions {self.dims}.'
      )
    if not set(c.dims).issubset(x.dims):
      raise ValueError(
          'Introduction of new axes via add_constant is not supported. For'
          f' that coordinate {c} must have dimensions that are subset of the'
          f' dimensions {x.dims=} but it is not.'
      )

    def add_fn(x, y):
      x = jnp.asarray(x)
      return x.at[0, 0].add(_YLM_CONSTANT_NORMALIZATION_FACTOR * y)

    x = x.untag(self)
    return cx.cmap(add_fn, out_axes=x.dims)(x, c).tag(self)

  def clip_wavenumbers(self, x: cx.Field, n: int) -> cx.Field:
    if n <= 0:
      raise ValueError(f'`n` must be >= 0; got {n}.')

    def clip(x):
      # Multiplication by the mask is significantly faster than directly using
      # `x.at[..., -n:].set(0)`
      num_zeros = n + self.wavenumber_padding[-1]
      mask = jnp.ones(self.shape[-1], x.dtype).at[-num_zeros:].set(0)
      return x * mask

    return cx.cpmap(clip)(x.untag(self)).tag(self)

  def __lt__(self, other: SphericalHarmonicGrid) -> bool:
    """Custom comparison operator for sorting SphericalHarmonicGrids."""

    # Implementing __lt__ enables SphericalHarmonicGrid to be keys in a dict
    # and be compatible with jax.tree. operations.
    def _to_tuple(x):
      xt = (x.total_wavenumbers, x.longitude_wavenumbers, x.wavenumber_padding)
      return xt

    if not isinstance(other, SphericalHarmonicGrid):
      return NotImplemented
    # Sort by total, then longitude, then padding.
    self_compare_values_in_order = _to_tuple(self)
    other_compare_values_in_order = _to_tuple(other)
    return self_compare_values_in_order < other_compare_values_in_order

  @classmethod
  def from_dinosaur_grid(
      cls,
      ylm_grid: spherical_harmonic.Grid,
  ):
    cls_to_method = {v: k for k, v in SPHERICAL_HARMONICS_METHODS.items()}
    method_name = cls_to_method[ylm_grid.spherical_harmonics_impl]  # pyrefly: ignore[bad-index]
    method_name = cast(SphericalHarmonicsMethodNames, method_name)
    return cls(
        longitude_wavenumbers=ylm_grid.longitude_wavenumbers,
        total_wavenumbers=ylm_grid.total_wavenumbers,
        spherical_harmonics_method=method_name,
        wavenumber_padding=ylm_grid.modal_padding,
    )

  @classmethod
  def with_wavenumbers(
      cls,
      longitude_wavenumbers: int,
      spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast',
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` by specifying only wavenumbers."""
    dims = ('longitude_wavenumber', 'total_wavenumber')
    method = SPHERICAL_HARMONICS_METHODS[spherical_harmonics_method]
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=longitude_wavenumbers + 1,
        longitude_nodes=0,
        latitude_nodes=0,
        spherical_harmonics_impl=method,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  @classmethod
  def construct(
      cls,
      max_wavenumber: int,
      spherical_harmonics_method: SphericalHarmonicsMethodNames = 'fast',
      mesh: parallelism.Mesh | None = None,
      partition_schema_key: str | None = None,
  ) -> SphericalHarmonicGrid:
    """Constructs a `SphericalHarmonicGrid` with max_wavenumber.

    Args:
      max_wavenumber: maximum wavenumber to resolve.
      spherical_harmonics_method: name of the Yₗᵐ implementation to use.
      mesh: optional Mesh that specifies necessary grid padding.
      partition_schema_key: key indicating a partition schema on `mesh` to infer
        padding details. Used only if an appropriate `mesh` is passed in.

    Returns:
      Constructed SphericalHarmonicGrid object.
    """
    dims = ('longitude_wavenumber', 'total_wavenumber')
    method = SPHERICAL_HARMONICS_METHODS[spherical_harmonics_method]
    ylm_grid = spherical_harmonic.Grid(
        longitude_wavenumbers=max_wavenumber + 1,
        total_wavenumbers=max_wavenumber + 2,
        longitude_nodes=0,
        latitude_nodes=0,
        spherical_harmonics_impl=method,
        spmd_mesh=_mesh_to_dinosaur_spmd_mesh(dims, mesh, partition_schema_key),
    )
    return cls.from_dinosaur_grid(ylm_grid=ylm_grid)

  # The factory methods below return "standard" grids that appear in the
  # literature. See, e.g. https://doi.org/10.5194/tc-12-1499-2018 and
  # https://www.ecmwf.int/en/forecasts/documentation-and-support/data-spatial-coordinate-systems

  # The number in these names correspond to the maximum resolved wavenumber,
  # which is one less than the number of wavenumbers used in the Grid
  # constructor. An additional total wavenumber is added because the top
  # wavenumber is clipped from the initial state and each calculation of
  # explicit tendencies.

  # The names for these factory methods (including capilatization) are
  # standard in the literature.
  # pylint:disable=invalid-name

  # T* grids can model quadratic terms without aliasing, because the maximum
  # total wavenumber is <= 2/3 of the number of latitudinal nodes. ECMWF
  # sometimes calls these "TQ" (truncated quadratic) grids.

  @classmethod
  def T21(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=21, **kwargs)

  @classmethod
  def T31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, **kwargs)

  @classmethod
  def T42(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=42, **kwargs)

  @classmethod
  def T85(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=85, **kwargs)

  @classmethod
  def T106(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=106, **kwargs)

  @classmethod
  def T119(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=119, **kwargs)

  @classmethod
  def T170(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=170, **kwargs)

  @classmethod
  def T213(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=213, **kwargs)

  @classmethod
  def T340(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=340, **kwargs)

  @classmethod
  def T425(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=425, **kwargs)

  # TL* grids do not truncate any frequencies, and hence can only model linear
  # terms exactly. ECMWF used "TL" (truncated linear) grids for semi-Lagrangian
  # advection (which eliminates quadratic terms) up to 2016, which it switched
  # to "cubic" grids for resolutions above TL1279:
  # https://www.ecmwf.int/sites/default/files/elibrary/2016/17262-new-grid-ifs.pdf

  @classmethod
  def TL31(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=31, **kwargs)

  @classmethod
  def TL47(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=47, **kwargs)

  @classmethod
  def TL63(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=63, **kwargs)

  @classmethod
  def TL95(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=95, **kwargs)

  @classmethod
  def TL127(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=127, **kwargs)

  @classmethod
  def TL159(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=159, **kwargs)

  @classmethod
  def TL179(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=179, **kwargs)

  @classmethod
  def TL255(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=255, **kwargs)

  @classmethod
  def TL639(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=639, **kwargs)

  @classmethod
  def TL1279(cls, **kwargs) -> SphericalHarmonicGrid:
    return cls.construct(max_wavenumber=1279, **kwargs)

  # EL* grids use equiangular linear (equiangular_with_poles) latitude spacing.
  # The spectral representation is the same as TL grids with the same
  # wavenumber.

  @classmethod
  def EL120(cls, **kwargs) -> SphericalHarmonicGrid:
    """Equiangular linear grid resolving up to wavenumber 120."""
    return cls.construct(max_wavenumber=120, **kwargs)

  @classmethod
  def EL180(cls, **kwargs) -> SphericalHarmonicGrid:
    """Equiangular linear grid resolving up to wavenumber 180."""
    return cls.construct(max_wavenumber=180, **kwargs)

  @classmethod
  def EL720(cls, **kwargs) -> SphericalHarmonicGrid:
    """Equiangular linear grid resolving up to wavenumber 720."""
    return cls.construct(max_wavenumber=720, **kwargs)

  @classmethod
  def EL1800(cls, **kwargs) -> SphericalHarmonicGrid:
    """Equiangular linear grid resolving up to wavenumber 1800."""
    return cls.construct(max_wavenumber=1800, **kwargs)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        wavenumber_padding=self.wavenumber_padding,
        spherical_harmonics_method=self.spherical_harmonics_method,
    )
    variables['longitude_wavenumber'].attrs = metadata
    variables['total_wavenumber'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:

    if dims[:2] != ('longitude_wavenumber', 'total_wavenumber'):
      return cx.coords.NoCoordinateMatch(
          "leading dimensions are not ('longitude_wavenumber',"
          " 'total_wavenumber')"
      )

    if coords['longitude_wavenumber'].dims != ('longitude_wavenumber',):
      return cx.coords.NoCoordinateMatch(
          'longitude_wavenumber is not a 1D coordinate'
      )

    if coords['total_wavenumber'].dims != ('total_wavenumber',):
      return cx.coords.NoCoordinateMatch(
          'total_wavenumber is not a 1D coordinate'
      )

    longitude_wavenumbers = (coords.sizes['longitude_wavenumber'] + 1) // 2
    wavenumber_padding = coords['longitude_wavenumber'].attrs.get(
        'wavenumber_padding', (0, 0)
    )
    spherical_harmonics_method = coords['longitude_wavenumber'].attrs.get(
        'spherical_harmonics_method', 'fast'
    )
    candidate = cls(
        longitude_wavenumbers=longitude_wavenumbers,
        total_wavenumbers=coords.sizes['total_wavenumber'],
        spherical_harmonics_method=spherical_harmonics_method,
        wavenumber_padding=(
            int(wavenumber_padding[0]),
            int(wavenumber_padding[1]),
        ),
    )

    expected = candidate.fields['longitude_wavenumber'].data
    got = coords['longitude_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.coords.NoCoordinateMatch(
          'inferred longitude wavenumbers do not match coordinate data:'
          f' {expected} vs {got}. Perhaps you attempted to restore coordinate '
          ' data from FastSphericalHarmonics, which does not support '
          'restoration?'
      )

    expected = candidate.fields['total_wavenumber'].data
    got = coords['total_wavenumber'].data
    if not np.array_equal(expected, got):
      return cx.coords.NoCoordinateMatch(
          f'inferred total wavenumbers do not match coordinate data: {expected}'
          f' vs {got}. Perhaps you attempted to restore coordinate '
          ' data from FastSphericalHarmonics, which does not support '
          'restoration?'
      )

    return candidate


_EARTH_RADIUS_KM = 6371.0


class PatchCoordinate(cx.Coordinate):
  """Base class for 2D local patch coordinates."""

  @property
  @abc.abstractmethod
  def dims(self) -> tuple[str, str]:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, int]:
    raise NotImplementedError

  def _remap_indexers_to_labeled_axes(
      self, indexers: dict[str | cx.Coordinate, Any]
  ) -> tuple[
      dict[str | cx.Coordinate, Any], tuple[cx.LabeledAxis, cx.LabeledAxis]
  ]:
    """Remaps SelectedAxis indexers to LabeledAxes for 1D indexing."""
    dim0, dim1 = self.dims
    ax0 = cx.LabeledAxis(dim0, np.asarray(self.fields[dim0].data))
    ax1 = cx.LabeledAxis(dim1, np.asarray(self.fields[dim1].data))
    sel0, sel1 = self.axes
    remap: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        sel0: ax0,
        sel1: ax1,
    }
    axes = (ax0, ax1)
    indexers = {
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    return indexers, axes

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: coordax.experimental.SelMethod = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    """Maps indexers along the two patch axes using 1D LabeledAxes."""
    dim0, dim1 = self.dims
    indexers, (ax0, ax1) = self._remap_indexers_to_labeled_axes(indexers)
    mapped, consumed = {}, set()
    if dim0 in indexers or ax0 in indexers:
      sub_mapped, sub_consumed = ax0.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)
    if dim1 in indexers or ax1 in indexers:
      sub_mapped, sub_consumed = ax1.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)
    from_labeled: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        l_ax: ax for l_ax, ax in zip((ax0, ax1), self.axes)
    }
    consumed = {from_labeled.get(ax, ax) for ax in consumed}
    mapped = {from_labeled.get(k, k): v for k, v in mapped.items()}
    return mapped, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    indexers, l_axes = self._remap_indexers_to_labeled_axes(indexers)
    l_isel = cx.coords.compose(*l_axes).isel(indexers)
    labeled_to_orig: dict[cx.Coordinate, cx.Coordinate] = {
        ax: orig for ax, orig in zip(l_axes, self.axes)
    }
    return cx.coords.compose(
        *[labeled_to_orig.get(c, c) for c in l_isel.axes]
    )

  def _reduce_mean(
      self,
      x: cx.Field,
      weights: cx.Field | None = None,
      *,
      skipna: bool = True,
  ) -> cx.Field:
    """Computes unweighted or weighted mean of `x` across `self`."""
    if not cx.contains_dims(x, self):
      raise ValueError(f'{x=} does not contain all axes of {self=}.')
    if weights is not None:
      w = weights.broadcast_like(x.coordinate)
      if not skipna:

        def _weighted_mean(vals, w_arr):
          w_sum = jnp.sum(w_arr)
          has_valid = w_sum > 0.0
          safe_w_sum = jnp.where(has_valid, w_sum, 1.0)
          return jnp.where(
              has_valid, jnp.sum(vals * w_arr) / safe_w_sum, jnp.nan
          )

        return cx.cmap(_weighted_mean)(x.untag(self), w.untag(self))

      def _weighted_nanmean(vals, w_arr):
        valid = jnp.isfinite(vals) & jnp.isfinite(w_arr)
        safe_vals = jnp.where(valid, vals, 0.0)
        safe_w = jnp.where(valid, w_arr, 0.0)
        w_sum = jnp.sum(safe_w)
        has_valid = w_sum > 0.0
        safe_w_sum = jnp.where(has_valid, w_sum, 1.0)
        return jnp.where(
            has_valid,
            jnp.sum(safe_vals * safe_w) / safe_w_sum,
            jnp.nan,
        )

      return cx.cmap(_weighted_nanmean)(x.untag(self), w.untag(self))

    reduce_fn = jnp.nanmean if skipna else jnp.mean
    return cx.cmap(reduce_fn)(x.untag(self))


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SwathPatch(PatchCoordinate):
  """2D satellite swath patch coordinate along ('scanline', 'fov').

  Represents a rectangular window of `scanlines` along-track scans by `fov`
  cross-track fields-of-view from a scanning satellite instrument. Exposes
  cross-track scan angles and spherical surface-area footprint weights along
  the `'fov'` axis.

  Attributes:
    scanlines: Number of along-track scanlines in the patch.
    fov: Number of cross-track fields-of-view per scanline.
    max_scan_angle: Maximum cross-track scan angle from nadir in degrees at the
      outermost FOV cell centers.
    satellite_altitude_km: Nominal satellite orbital altitude above Earth's
      surface in kilometers.
  """

  scanlines: int
  fov: int
  max_scan_angle: float
  satellite_altitude_km: float

  def __post_init__(self):
    if self.scanlines < 1 or self.fov < 1:
      raise ValueError(
          'scanlines and fov must be >= 1, got'
          f' {self.scanlines=}, {self.fov=}.'
      )
    if not (0.0 <= self.max_scan_angle < 90.0):
      raise ValueError(
          'max_scan_angle must be in [0, 90) degrees, got'
          f' {self.max_scan_angle=}.'
      )
    if self.satellite_altitude_km < 0.0:
      raise ValueError(
          'Underground satellites are not supported yet, got'
          f' {self.satellite_altitude_km=}.'
      )
    r_ratio = 1.0 + self.satellite_altitude_km / _EARTH_RADIUS_KM
    sin_zenith_max = r_ratio * np.sin(np.deg2rad(self.max_scan_angle))
    if sin_zenith_max >= 1.0:
      horizon_deg = np.rad2deg(np.arcsin(1.0 / r_ratio))
      raise ValueError(
          f'{self.max_scan_angle=} exceeds geometric horizon'
          f' ({horizon_deg:.2f} deg) for {self.satellite_altitude_km=} km.'
      )

  @property
  def dims(self) -> tuple[str, str]:
    return ('scanline', 'fov')

  @property
  def shape(self) -> tuple[int, int]:
    return (self.scanlines, self.fov)

  def _tuple(self):
    return (
        self.scanlines,
        self.fov,
        self.max_scan_angle,
        self.satellite_altitude_km,
    )

  def __lt__(self, other) -> bool:
    if not isinstance(other, SwathPatch):
      return NotImplemented
    return self._tuple() < other._tuple()

  @property
  def fields(self) -> dict[str, cx.Field]:
    scanline_axis, fov_axis = self.axes
    return {
        'scanline': cx.field(np.arange(self.scanlines), scanline_axis),
        'fov': cx.field(np.arange(self.fov), fov_axis),
    }

  @classmethod
  def atms(cls, scanlines: int = 135) -> Self:
    """Returns a SwathPatch configured with nominal JPSS ATMS geometry."""
    return cls(
        scanlines=scanlines,
        fov=96,
        max_scan_angle=52.725,
        satellite_altitude_km=833.0,
    )

  @functools.cached_property
  def scan_angles(self) -> cx.Field:
    """Nominal cross-track scan angles in degrees along 'fov'."""
    _, fov_axis = self.axes
    with jax.ensure_compile_time_eval():
      if self.fov == 1:
        angles = jnp.zeros(1, dtype=jnp.float32)
      else:
        angles = jnp.linspace(
            -self.max_scan_angle,
            self.max_scan_angle,
            self.fov,
            dtype=jnp.float32,
        )
      return cx.field(angles, fov_axis)

  @functools.cached_property
  def footprint_weights(self) -> cx.Field:
    """Normalized relative footprint area weights along 'fov' (mean = 1)."""
    _, fov_axis = self.axes
    r_sat = _EARTH_RADIUS_KM + self.satellite_altitude_km
    r_ratio = r_sat / _EARTH_RADIUS_KM

    def _normalize_weights(scan_deg):
      scan_rad = jnp.deg2rad(jnp.abs(scan_deg))
      sin_zenith = jnp.clip(r_ratio * jnp.sin(scan_rad), 0.0, 1.0)
      cos_zenith = jnp.sqrt(jnp.maximum(1.0 - sin_zenith**2, 1e-12))
      zenith_rad = jnp.arcsin(sin_zenith)
      psi = zenith_rad - scan_rad
      slant_range = r_sat * jnp.cos(scan_rad) - _EARTH_RADIUS_KM * cos_zenith
      raw_area = (
          jnp.cos(psi)
          * (slant_range / _EARTH_RADIUS_KM)
          / jnp.maximum(cos_zenith, 1e-6)
      )
      return raw_area / jnp.mean(raw_area)

    with jax.ensure_compile_time_eval():
      return cx.cmap(_normalize_weights)(
          self.scan_angles.untag(fov_axis)
      ).tag(fov_axis)

  def mean(
      self,
      x: cx.Field,
      *,
      area_weighted: bool = False,
      skipna: bool = True,
  ) -> cx.Field:
    """Computes the mean of `x` across the ('scanline', 'fov') axes."""
    weights = self.footprint_weights if area_weighted else None
    return self._reduce_mean(x, weights=weights, skipna=skipna)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        max_scan_angle=self.max_scan_angle,
        satellite_altitude_km=self.satellite_altitude_km,
    )
    variables['scanline'].attrs = metadata
    variables['fov'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    if dims[:2] != ('scanline', 'fov'):
      return cx.coords.NoCoordinateMatch(
          f"leading dimensions {dims[:2]!r} are not ('scanline', 'fov')"
      )
    if 'scanline' not in coords or 'fov' not in coords:
      return cx.coords.NoCoordinateMatch(
          "missing 'scanline' or 'fov' coordinate variables"
      )
    if coords['scanline'].dims != ('scanline',):
      return cx.coords.NoCoordinateMatch('scanline is not a 1D coordinate')
    if coords['fov'].dims != ('fov',):
      return cx.coords.NoCoordinateMatch('fov is not a 1D coordinate')

    scanlines = coords.sizes['scanline']
    fov = coords.sizes['fov']
    if not np.array_equal(coords['scanline'].data, np.arange(scanlines)):
      return cx.coords.NoCoordinateMatch(
          'scanline coordinate is not sequential integers 0..scanlines-1'
      )
    if not np.array_equal(coords['fov'].data, np.arange(fov)):
      return cx.coords.NoCoordinateMatch(
          'fov coordinate is not sequential integers 0..fov-1'
      )

    attrs = coords['scanline'].attrs or coords['fov'].attrs
    if 'max_scan_angle' not in attrs or 'satellite_altitude_km' not in attrs:
      return cx.coords.NoCoordinateMatch(
          'max_scan_angle or satellite_altitude_km not in "scanline" or "fov"'
          ' attributes'
      )
    return cls(
        scanlines=scanlines,
        fov=fov,
        max_scan_angle=float(attrs['max_scan_angle']),
        satellite_altitude_km=float(attrs['satellite_altitude_km']),
    )


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LonLatPatch(PatchCoordinate):
  """2D index-aligned sub-window of LonLatGrid along ('patch_lon', 'patch_lat').

  Represents a fixed-size `(longitude_nodes, latitude_nodes)` rectangular patch
  sliced from a parent `LonLatGrid`. Because the coordinate dimensions are node
  offsets `(0..longitude_nodes-1, 0..latitude_nodes-1)` rather than fixed global
  degrees, multiple patches at different geographic locations can be stacked and
  vectorized along a batch axis while preserving exact alignment with the parent
  `LonLatGrid`.

  Attributes:
    longitude_nodes: Number of grid nodes along `'patch_lon'`.
    latitude_nodes: Number of grid nodes along `'patch_lat'`.
  """

  longitude_nodes: int
  latitude_nodes: int

  def __post_init__(self):
    if self.longitude_nodes < 1 or self.latitude_nodes < 1:
      raise ValueError(
          'longitude_nodes and latitude_nodes must be >= 1, got'
          f' {self.longitude_nodes=}, {self.latitude_nodes=}.'
      )

  @property
  def dims(self) -> tuple[str, str]:
    return ('patch_lon', 'patch_lat')

  @property
  def shape(self) -> tuple[int, int]:
    return (self.longitude_nodes, self.latitude_nodes)

  def _tuple(self):
    return (self.longitude_nodes, self.latitude_nodes)

  def __lt__(self, other) -> bool:
    if not isinstance(other, LonLatPatch):
      return NotImplemented
    return self._tuple() < other._tuple()

  @property
  def fields(self) -> dict[str, cx.Field]:
    lon_axis, lat_axis = self.axes
    return {
        'patch_lon': cx.field(np.arange(self.longitude_nodes), lon_axis),
        'patch_lat': cx.field(np.arange(self.latitude_nodes), lat_axis),
    }

  def mean(
      self,
      x: cx.Field,
      *,
      latitude: cx.Field | None = None,
      skipna: bool = True,
  ) -> cx.Field:
    """Computes the mean of `x` across ('patch_lon', 'patch_lat').

    Args:
      x: Field containing `self` in its coordinates.
      latitude: Optional latitude field in degrees containing `self` (or
        `self.axes[1]`) used for `cos(latitude)` spherical area weighting.
      skipna: Whether to ignore NaN values when averaging.

    Returns:
      Field with `('patch_lon', 'patch_lat')` reduced.
    """
    if latitude is None:
      return self._reduce_mean(x, weights=None, skipna=skipna)
    cos_lat = cx.cmap(
        lambda lat_deg: jnp.maximum(jnp.cos(jnp.deg2rad(lat_deg)), 0.0)
    )(latitude)
    return self._reduce_mean(x, weights=cos_lat, skipna=skipna)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    if dims[:2] != ('patch_lon', 'patch_lat'):
      return cx.coords.NoCoordinateMatch(
          f"leading dimensions {dims[:2]!r} are not ('patch_lon', 'patch_lat')"
      )
    if 'patch_lon' not in coords or 'patch_lat' not in coords:
      return cx.coords.NoCoordinateMatch(
          "missing 'patch_lon' or 'patch_lat' coordinate variables"
      )
    if coords['patch_lon'].dims != ('patch_lon',):
      return cx.coords.NoCoordinateMatch('patch_lon is not a 1D coordinate')
    if coords['patch_lat'].dims != ('patch_lat',):
      return cx.coords.NoCoordinateMatch('patch_lat is not a 1D coordinate')
    n_lon = coords.sizes['patch_lon']
    n_lat = coords.sizes['patch_lat']
    if not np.array_equal(coords['patch_lon'].data, np.arange(n_lon)):
      return cx.coords.NoCoordinateMatch(
          'patch_lon is not sequential integers 0..longitude_nodes-1'
      )
    if not np.array_equal(coords['patch_lat'].data, np.arange(n_lat)):
      return cx.coords.NoCoordinateMatch(
          'patch_lat is not sequential integers 0..latitude_nodes-1'
      )
    return cls(longitude_nodes=n_lon, latitude_nodes=n_lat)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LocalGridPatch(PatchCoordinate):
  """2D uniform local rotated-sphere grid patch along ('patch_x', 'patch_y').

  Defines a regular `(x_nodes, y_nodes)` grid with spacing `(dx_deg, dy_deg)` in
  degrees centered at `(0, 0)` on a local rotated equator. When embedded at any
  geographic center `(center_lon, center_lat)` via
  `patch_geometry.local_patch_lon_lat`, the patch is free of polar meridian
  convergence and dateline discontinuities.

  Attributes:
    x_nodes: Number of grid nodes along `'patch_x'`.
    y_nodes: Number of grid nodes along `'patch_y'`.
    dx_deg: Angular spacing in degrees along `'patch_x'`.
    dy_deg: Angular spacing in degrees along `'patch_y'`.
  """

  x_nodes: int
  y_nodes: int
  dx_deg: float
  dy_deg: float

  def __post_init__(self):
    if self.x_nodes < 1 or self.y_nodes < 1:
      raise ValueError(
          f'x_nodes and y_nodes must be >= 1, got {self.x_nodes=},'
          f' {self.y_nodes=}.'
      )
    if self.dx_deg <= 0.0 or self.dy_deg <= 0.0:
      raise ValueError(
          f'dx_deg and dy_deg must be > 0, got {self.dx_deg=}, {self.dy_deg=}.'
      )
    span_x = (self.x_nodes - 1) * self.dx_deg
    span_y = (self.y_nodes - 1) * self.dy_deg
    if span_x >= 360.0 or span_y >= 180.0:
      raise ValueError(
          f'LocalGridPatch angular span ({span_x=}, {span_y=}) must be'
          ' < (360, 180) degrees.'
      )

  @property
  def dims(self) -> tuple[str, str]:
    return ('patch_x', 'patch_y')

  @property
  def shape(self) -> tuple[int, int]:
    return (self.x_nodes, self.y_nodes)

  def _tuple(self):
    return (self.x_nodes, self.y_nodes, self.dx_deg, self.dy_deg)

  def __lt__(self, other) -> bool:
    if not isinstance(other, LocalGridPatch):
      return NotImplemented
    return self._tuple() < other._tuple()

  @property
  def fields(self) -> dict[str, cx.Field]:
    x_axis, y_axis = self.axes
    half_x = 0.5 * (self.x_nodes - 1) * self.dx_deg
    half_y = 0.5 * (self.y_nodes - 1) * self.dy_deg
    xs = (
        np.zeros(1, dtype=np.float32)
        if self.x_nodes == 1
        else np.linspace(-half_x, half_x, self.x_nodes, dtype=np.float32)
    )
    ys = (
        np.zeros(1, dtype=np.float32)
        if self.y_nodes == 1
        else np.linspace(-half_y, half_y, self.y_nodes, dtype=np.float32)
    )
    return {
        'patch_x': cx.field(xs, x_axis),
        'patch_y': cx.field(ys, y_axis),
    }

  @functools.cached_property
  def area_weights(self) -> cx.Field:
    """Normalized spherical area weights cos(patch_y) along 'patch_y'."""
    _, y_axis = self.axes

    def _norm_cos(eta_deg):
      cos_y = jnp.cos(jnp.deg2rad(eta_deg))
      return cos_y / jnp.mean(cos_y)

    with jax.ensure_compile_time_eval():
      return cx.cmap(_norm_cos)(self.fields['patch_y'].untag(y_axis)).tag(
          y_axis
      )

  def mean(
      self,
      x: cx.Field,
      *,
      area_weighted: bool = False,
      skipna: bool = True,
  ) -> cx.Field:
    """Computes the mean of `x` across ('patch_x', 'patch_y')."""
    weights = self.area_weights if area_weighted else None
    return self._reduce_mean(x, weights=weights, skipna=skipna)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(dx_deg=self.dx_deg, dy_deg=self.dy_deg)
    variables['patch_x'].attrs = metadata
    variables['patch_y'].attrs = metadata
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    if dims[:2] != ('patch_x', 'patch_y'):
      return cx.coords.NoCoordinateMatch(
          f"leading dimensions {dims[:2]!r} are not ('patch_x', 'patch_y')"
      )
    if 'patch_x' not in coords or 'patch_y' not in coords:
      return cx.coords.NoCoordinateMatch(
          "missing 'patch_x' or 'patch_y' coordinate variables"
      )
    if coords['patch_x'].dims != ('patch_x',):
      return cx.coords.NoCoordinateMatch('patch_x is not a 1D coordinate')
    if coords['patch_y'].dims != ('patch_y',):
      return cx.coords.NoCoordinateMatch('patch_y is not a 1D coordinate')

    attrs = coords['patch_x'].attrs or coords['patch_y'].attrs
    if 'dx_deg' not in attrs or 'dy_deg' not in attrs:
      return cx.coords.NoCoordinateMatch(
          'dx_deg or dy_deg not in "patch_x" or "patch_y" attributes'
      )
    candidate = cls(
        x_nodes=coords.sizes['patch_x'],
        y_nodes=coords.sizes['patch_y'],
        dx_deg=float(attrs['dx_deg']),
        dy_deg=float(attrs['dy_deg']),
    )
    if not np.allclose(
        coords['patch_x'].data,
        np.asarray(candidate.fields['patch_x'].data),
        atol=1e-5,
    ):
      return cx.coords.NoCoordinateMatch('patch_x values do not match')
    if not np.allclose(
        coords['patch_y'].data,
        np.asarray(candidate.fields['patch_y'].data),
        atol=1e-5,
    ):
      return cx.coords.NoCoordinateMatch('patch_y values do not match')
    return candidate


#
# Vertical level coordinates
#


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SigmaLevels(cx.Coordinate):
  """Coordinates that discretize data as fraction of the surface pressure."""

  boundaries: np.ndarray
  sigma_levels: sigma_coordinates.SigmaCoordinates = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __init__(self, boundaries: Iterable[float] | np.ndarray):
    boundaries = np.asarray(boundaries, np.float32)
    object.__setattr__(self, 'boundaries', boundaries)
    self.__post_init__()

  def __post_init__(self):
    sigma_levels = sigma_coordinates.SigmaCoordinates(
        boundaries=self.boundaries
    )
    object.__setattr__(self, 'sigma_levels', sigma_levels)

  @property
  def dims(self) -> tuple[str, ...]:
    return ('sigma',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.sigma_levels.centers.shape

  @property
  def fields(self):
    boundaries, centers = self.boundaries, self.sigma_levels.centers
    return {
        'sigma': cx.field(centers, self),
        'sigma_boundaries': cx.field(boundaries, SigmaBoundaries(boundaries)),
    }

  @functools.cached_property
  def sigma_boundaries(self) -> SigmaBoundaries:
    return SigmaBoundaries(boundaries=self.boundaries)

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.sigma_levels.centers)
    remap: dict[str | cx.Coordinate, cx.Coordinate] = {self: sigma_labeled}
    indexers = {  # remaped to sigma_labeled.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    i_indexers, consumed = sigma_labeled.map_indexers(indexers, method)
    from_labeled: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        sigma_labeled: self
    }
    consumed = {from_labeled.get(k, k) for k in consumed}
    i_indexers = {from_labeled.get(k, k): v for k, v in i_indexers.items()}
    return i_indexers, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.sigma_levels.centers)
    remap: dict[str | cx.Coordinate, cx.Coordinate] = {self: sigma_labeled}
    indexers = {  # remaped to sigma_labeled.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    l_isel = sigma_labeled.isel(indexers)
    return self if l_isel == sigma_labeled else l_isel

  def integrate(self, x: cx.Field) -> cx.Field:
    """Integrates `x` over the sigma levels."""
    sigma_integrate = functools.partial(
        sigma_coordinates.sigma_integral,
        coordinates=self.sigma_levels,
        axis=0,
        keepdims=False,
    )
    return cx.cmap(sigma_integrate)(x.untag(self))

  def integrate_over_pressure(
      self,
      x: cx.Field,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    """Integrates `x` over the sigma levels weighted by surface pressure."""
    del sim_units  # unused
    return surface_pressure * self.integrate(x)

  def integrate_cumulative(
      self,
      x: cx.Field,
      downward: bool = True,
      cumsum_method: str = 'dot',
      sharding: jax.sharding.NamedSharding | None = None,
  ) -> cx.Field:
    """Integrates `x` over the sigma levels."""
    sigma_cumulative_integrate = functools.partial(
        sigma_coordinates.cumulative_sigma_integral,
        coordinates=self.sigma_levels,
        axis=0,
        downward=downward,
        cumsum_method=cumsum_method,
        sharding=sharding,
    )
    return cx.cpmap(sigma_cumulative_integrate)(x.untag(self)).tag(self)

  @property
  def centers(self):
    return self.sigma_levels.centers

  def pressure_centers(
      self,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    """Returns pressure at layer centers given `surface_pressure`.

    Args:
      surface_pressure: surface pressure, in the same units as the output.
      sim_units: unused, accepted for API compatibility with other levels.

    Returns:
      Pressure at sigma level centers.
    """
    del sim_units  # unused, output has units of `surface_pressure`.
    return self.fields['sigma'] * surface_pressure

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.boundaries),)

  def __eq__(self, other):
    return (
        isinstance(other, SigmaLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def to_sigma_boundaries(self) -> SigmaBoundaries:
    return SigmaBoundaries(boundaries=self.boundaries)

  @classmethod
  def from_sigma_boundaries(cls, sigma_boundaries: SigmaBoundaries) -> Self:
    return cls(boundaries=sigma_boundaries.boundaries)

  @classmethod
  def from_dinosaur_sigma_levels(
      cls,
      sigma_levels: sigma_coordinates.SigmaCoordinates,
  ):
    return cls(boundaries=sigma_levels.boundaries)

  @classmethod
  def equidistant(
      cls,
      layers: int,
  ) -> SigmaLevels:
    sigma_levels = sigma_coordinates.SigmaCoordinates.equidistant(layers)
    boundaries = sigma_levels.boundaries
    return cls(boundaries=boundaries)

  @classmethod
  def from_centers(cls, centers: np.ndarray) -> Self:
    sigma_levels = sigma_coordinates.SigmaCoordinates.from_centers(centers)
    boundaries = sigma_levels.boundaries
    return cls(boundaries=boundaries)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'sigma':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "sigma"')

    if coords['sigma'].ndim != 1:
      return cx.coords.NoCoordinateMatch('sigma coordinate is not a 1D array')

    centers = coords['sigma'].data
    candidate = cls.from_centers(centers)
    actual_centers = candidate.sigma_levels.centers
    if not np.array_equal(actual_centers, centers):
      return cx.coords.NoCoordinateMatch(
          'inferred sigma boundaries do not exactly match coordinate data:'
          f' {actual_centers} vs {centers}.'
      )
    return candidate


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SigmaBoundaries(SigmaLevels):
  """Coordinate that represents boundaries of SigmaLevels."""

  @property
  def dims(self) -> tuple[str, ...]:
    return ('sigma_boundaries',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.boundaries.shape

  @property
  def fields(self):
    boundaries, centers = self.boundaries, self.sigma_levels.centers
    return {
        'sigma_boundaries': cx.field(boundaries, self),
        'sigma': cx.field(centers, SigmaLevels(boundaries)),
    }

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.boundaries)
    remap: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        self: sigma_labeled
    }
    indexers = {  # remaped to sigma_labeled.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    i_indexers, consumed = sigma_labeled.map_indexers(indexers, method)
    from_labeled: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        sigma_labeled: self
    }
    consumed = {from_labeled.get(k, k) for k in consumed}
    i_indexers = {from_labeled.get(k, k): v for k, v in i_indexers.items()}
    return i_indexers, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    sigma_labeled = cx.LabeledAxis(self.dims[0], self.boundaries)
    remap: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        self: sigma_labeled
    }
    indexers = {  # remaped to sigma_labeled.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    l_isel = sigma_labeled.isel(indexers)
    return self if l_isel == sigma_labeled else l_isel

  def __eq__(self, other):
    return (
        isinstance(other, SigmaBoundaries)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash((self._components(), 'boundaries'))

  def to_sigma_levels(self) -> SigmaLevels:
    return SigmaLevels(boundaries=self.boundaries)

  @classmethod
  def from_sigma_levels(cls, sigma_levels: SigmaLevels) -> Self:
    return cls(boundaries=sigma_levels.boundaries)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'sigma_boundaries':
      return cx.coords.NoCoordinateMatch(
          f'dimension {dim!r} != "sigma_boundaries"'
      )
    sigma_dim = dim.removesuffix('_boundaries')
    sigma_levels = SigmaLevels.from_xarray((sigma_dim,), coords)
    if isinstance(sigma_levels, cx.coords.NoCoordinateMatch):
      return sigma_levels
    return cls.from_sigma_levels(sigma_levels)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class PressureLevels(cx.Coordinate):
  """Coordinates that discretize data per pressure levels."""

  centers: np.ndarray
  pressure_levels: vertical_interpolation.PressureCoordinates = (
      dataclasses.field(init=False, repr=False, compare=False)
  )

  def __init__(self, centers: Iterable[float] | np.ndarray):
    centers = np.asarray(centers, dtype=np.float32)
    object.__setattr__(self, 'centers', centers)
    self.__post_init__()

  def __post_init__(self):
    pressure_levels = vertical_interpolation.PressureCoordinates(
        centers=self.centers
    )
    object.__setattr__(self, 'pressure_levels', pressure_levels)

  @property
  def dims(self) -> tuple[str, ...]:
    return ('pressure',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.centers.shape

  @property
  def fields(self):
    return {'pressure': cx.field(self.centers, self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    return coordax.experimental.map_indexers_using_ticks(
        self,
        indexers,
        ticks_are_sorted=True,
        method=method,
    )

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    key = self.dims[0] if self.dims[0] in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return cx.Scalar()

    return PressureLevels(self.centers[indexer])

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.centers),)

  def __eq__(self, other):
    return (
        isinstance(other, PressureLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def pressure_centers(
      self,
      surface_pressure: cx.Field | None = None,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    """Returns pressure at levels in Pa (nondimensionalized if `sim_units`).

    Args:
      surface_pressure: unused, accepted for API compatibility with other
        levels.
      sim_units: if provided, the output is nondimensionalized.

    Returns:
      Pressure at pressure levels.
    """
    del surface_pressure  # unused, pressure levels are fixed.
    pressure = self.centers * 100  # In Pascal.
    if sim_units is not None:
      pressure = sim_units.nondimensionalize(pressure * typing.units.Pa)
    return cx.field(pressure, self)

  @classmethod
  def from_dinosaur_pressure_levels(
      cls,
      pressure_levels: vertical_interpolation.PressureCoordinates,
  ):
    return cls(centers=pressure_levels.centers)

  @classmethod
  def with_era5_levels(cls):
    """PressureLevels with standard 37 ERA5 pressure levels."""
    return cls(
        centers=[
            1,
            2,
            3,
            5,
            7,
            10,
            20,
            30,
            50,
            70,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
        ]
    )

  @classmethod
  def with_13_era5_levels(cls):
    """PressureLevels with commonly used subset of 13 ERA5 pressure levels."""
    centers = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    return cls(centers=centers)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim not in {'level', 'pressure'}:
      return cx.coords.NoCoordinateMatch(
          f'dimension {dim!r} is not "pressure" or "level"'
      )
    if coords[dim].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'pressure coordinate is not a 1D array'
      )
    centers = coords[dim].data
    if centers[0] < 0:
      return cx.coords.NoCoordinateMatch(
          f'pressure levels cannot be negative, got: {centers}'
      )
    if centers[-1] > 1025:
      return cx.coords.NoCoordinateMatch(
          f'pressure levels must end before 1025, got: {centers}'
      )
    return cls(centers=centers)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class HybridLevels(cx.Coordinate):
  """Coordinates with hybrid sigma-pressure levels."""

  a_boundaries: np.ndarray
  b_boundaries: np.ndarray
  hybrid_levels: hybrid_coordinates.HybridCoordinates = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __init__(
      self,
      a_boundaries: Iterable[float] | np.ndarray,
      b_boundaries: Iterable[float] | np.ndarray,
  ):
    a_boundaries = np.asarray(a_boundaries, np.float32)
    b_boundaries = np.asarray(b_boundaries, np.float32)
    object.__setattr__(self, 'a_boundaries', a_boundaries)
    object.__setattr__(self, 'b_boundaries', b_boundaries)
    self.__post_init__()

  def __post_init__(self):
    hybrid_levels = hybrid_coordinates.HybridCoordinates(
        a_boundaries=self.a_boundaries, b_boundaries=self.b_boundaries
    )
    object.__setattr__(self, 'hybrid_levels', hybrid_levels)

  @property
  def dims(self) -> tuple[str, ...]:
    return ('hybrid',)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.hybrid_levels.layers,)

  @property
  def fields(self):
    # Use index starting at 1 to match ECMWF's IFS convention.
    a = (self.a_boundaries[:-1] + self.a_boundaries[1:]) / 2.0
    b = (self.b_boundaries[:-1] + self.b_boundaries[1:]) / 2.0
    return {
        'hybrid': cx.field(np.arange(1, self.shape[0] + 1), self),
        'a': cx.field(a, self),
        'b': cx.field(b, self),
    }

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    hybrid_labeled = cx.LabeledAxis(self.dims[0], self.fields['hybrid'].data)
    remap: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        self: hybrid_labeled
    }
    idxrs = {remap.get(k, k): v for k, v in indexers.items()}
    return hybrid_labeled.isel(idxrs)

  def asdict(self) -> dict[str, Any]:
    return {
        'a_boundaries': self.a_boundaries.tolist(),
        'b_boundaries': self.b_boundaries.tolist(),
    }

  def _components(self):
    return (ArrayKey(self.a_boundaries), ArrayKey(self.b_boundaries))

  def __eq__(self, other):
    return (
        isinstance(other, HybridLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def pressure_boundaries(
      self,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    a_boundaries, b_boundaries = self.a_boundaries, self.b_boundaries
    a_boundaries = a_boundaries * 100  # Convert to Pascal.
    if sim_units is not None:
      a_boundaries = sim_units.nondimensionalize(a_boundaries * typing.units.Pa)

    def _boundaries(p_surface: jax.Array) -> jax.Array:
      return a_boundaries + b_boundaries * p_surface  # pyrefly: ignore[bad-return]

    # TODO(dkochkov): Consider adding HybridLevelsBoundaries coordinate and
    # tagging this output with it.
    return cx.cmap(_boundaries, out_axes=(..., *surface_pressure.dims))(
        surface_pressure
    )

  def pressure_centers(
      self,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None = None,
  ) -> cx.Field:
    """Returns pressure at layer centers given `surface_pressure`."""
    a = self.fields['a'] * 100  # In Pascal.
    if sim_units is not None:
      a = cx.field(sim_units.nondimensionalize(a.data * typing.units.Pa), self)
    return a + self.fields['b'] * surface_pressure

  def integrate_over_pressure(
      self,
      x: cx.Field,
      surface_pressure: cx.Field,
      sim_units: units.SimUnits | None,
  ) -> cx.Field:
    a, b = self.hybrid_levels.a_boundaries, self.hybrid_levels.b_boundaries
    if sim_units is not None:
      a_nondim = sim_units.nondimensionalize(a * typing.units.hPa)
    else:
      a_nondim = a * 100  # default hPa -> Pa conversion.
    nondim_hybrid_levels = hybrid_coordinates.HybridCoordinates(a_nondim, b)  # pyrefly: ignore[bad-argument-type]
    integral_fn = functools.partial(
        hybrid_coordinates.integral_over_pressure,
        coordinates=nondim_hybrid_levels,
        axis=0,
        keepdims=False,
    )
    return cx.cmap(integral_fn)(x.untag(self), surface_pressure)

  def to_xarray(self) -> dict[str, xarray.Variable]:
    variables = super().to_xarray()
    metadata = dict(
        a_boundaries=self.a_boundaries.tolist(),
        b_boundaries=self.b_boundaries.tolist(),
    )
    variables['hybrid'].attrs = metadata
    return variables

  @classmethod
  def from_coefficients(
      cls,
      a_coefficients: list[float] | np.ndarray,
      b_coefficients: list[float] | np.ndarray,
      p0_in_hpa: float = 1000.0,
  ) -> Self:
    hybrid_levels = hybrid_coordinates.HybridCoordinates.from_coefficients(
        a_coeffs=a_coefficients, b_coeffs=b_coefficients, p0=p0_in_hpa
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def from_dinosaur_hybrid_levels(
      cls,
      hybrid_levels: hybrid_coordinates.HybridCoordinates,
  ) -> Self:
    return cls(
        a_boundaries=hybrid_levels.a_boundaries,
        b_boundaries=hybrid_levels.b_boundaries,
    )

  @classmethod
  def with_n_levels(
      cls,
      n_levels,
      sigma_exponent: float = 1.5,
      stretch_exponent: float = 0.5,
  ) -> Self:
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_levels(
        n_levels=n_levels,
        sigma_exponent=sigma_exponent,
        stretch_exponent=stretch_exponent,
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  # TODO(janniyuval) To remove after submitting cl/844884846
  @classmethod
  def analytic_ecmwf_like(
      cls,
      n_levels: int = 137,
      p0: float = 1013.25,
      p_top: float = 0.0,
      p_transition: float = 80.0,
      alpha_strat: float = 4.2,
      alpha_surf: float = 0.25,
      b_exponent: float = 1.3,
  ) -> Self:
    """Returns HybridLevels constructed using analytic_ecmwf_like strategy."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.analytic_ecmwf_like(
        n_levels=n_levels,
        p0=p0,
        p_top=p_top,
        p_transition=p_transition,
        alpha_strat=alpha_strat,
        alpha_surf=alpha_surf,
        b_exponent=b_exponent,
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def ecmwf137_interpolated(
      cls,
      n_levels: int,
      p_top: float | None = None,
  ) -> Self:
    """Returns HybridLevels interpolated from ECMWF 137 levels.

    Args:
      n_levels: number of layers.
      p_top: optional model top pressure in hPa. If given, ECMWF interfaces
        above `p_top` are dropped before interpolating. See
        `dinosaur.hybrid_coordinates.HybridCoordinates.ecmwf137_interpolated`.
    """
    hybrid_levels = hybrid_coordinates.HybridCoordinates.ecmwf137_interpolated(
        n_levels=n_levels,
        p_top=p_top,
    )
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def ECMWF137(cls) -> Self:
    """Returns HybridLevels with 137 levels from ECMWF's IFS."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.ECMWF137()
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def UFS127(cls) -> Self:
    """Returns HybridLevels with 127 levels from NOAA's UFS."""
    hybrid_levels = hybrid_coordinates.HybridCoordinates.UFS127()
    return cls.from_dinosaur_hybrid_levels(hybrid_levels)

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'hybrid':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "hybrid"')

    if 'hybrid' not in coords:
      return cx.coords.NoCoordinateMatch(
          'no associated coordinate for "hybrid"'
      )

    if coords['hybrid'].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          '"hybrid" coordinate is not a 1D array'
      )

    attrs = coords['hybrid'].attrs
    if 'a_boundaries' not in attrs or 'b_boundaries' not in attrs:
      return cx.coords.NoCoordinateMatch(
          'a_boundaries or b_boundaries not in "hybrid" attributes'
      )

    a_boundaries = np.asarray(attrs['a_boundaries'])
    b_boundaries = np.asarray(attrs['b_boundaries'])

    # Check that the hybrid coordinate is just an index.
    n_layers = len(a_boundaries) - 1
    if coords.sizes[dim] != n_layers:
      return cx.coords.NoCoordinateMatch('level dimension size mismatch')
    if not np.array_equal(coords['hybrid'].data, np.arange(1, n_layers + 1)):
      return cx.coords.NoCoordinateMatch(
          'hybrid coordinate is not a simple index'
      )

    return cls(a_boundaries=a_boundaries, b_boundaries=b_boundaries)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LayerLevels(cx.Coordinate):
  """Coordinates that discretize data by index of unstructured layer."""

  n_layers: int
  name: str = 'layer_index'

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.n_layers,)

  @property
  def fields(self):
    return {self.name: cx.field(np.arange(self.n_layers), self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    labeled_layers = cx.LabeledAxis(self.name, np.arange(self.n_layers))
    remap: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        self: labeled_layers
    }
    indexers = {  # remaped to labeled_layers.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    i_indexers, consumed = labeled_layers.map_indexers(indexers, method)
    from_labeled: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        labeled_layers: self
    }
    consumed = {from_labeled.get(k, k) for k in consumed}
    i_indexers = {from_labeled.get(k, k): v for k, v in i_indexers.items()}
    return i_indexers, consumed

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    labeled_layers = cx.LabeledAxis(self.name, np.arange(self.n_layers))
    remap: dict[str | cx.Coordinate, str | cx.Coordinate] = {
        self: labeled_layers
    }
    indexers = {  # remaped to labeled_layers.
        remap.get(k, k): remap.get(v, v) if cx.is_coord(v) else v
        for k, v in indexers.items()
    }
    l_isel = labeled_layers.isel(indexers)
    return self if l_isel == labeled_layers else l_isel

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'layer_index':
      return cx.coords.NoCoordinateMatch(f'dimension {dim!r} != "layer_index"')

    if coords['layer_index'].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'layer_index coordinate is not a 1D array'
      )

    n_layers = coords.sizes['layer_index']
    got = coords['layer_index'].data
    if not np.array_equal(got, np.arange(n_layers)):
      return cx.coords.NoCoordinateMatch(
          f'unexpected layer_index coordinate is not sequential integers: {got}'
      )
    return cls(n_layers=n_layers)


def validate_nonnegative_depth(centers: Iterable[float] | np.ndarray):
  centers = np.asarray(centers)
  if not np.min(centers) >= 0:
    raise ValueError('SoilLevels coordinate should have non-negative depth.')


def validate_increasing_depth(centers: Iterable[float] | np.ndarray):
  centers = np.asarray(centers)
  if not np.min(np.diff(centers)) > 0:
    raise ValueError(
        'SoilLevels coordinate should have monotonically increasing depth,'
        ' starting at the soil surface.'
    )


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SoilLevels(cx.Coordinate):
  """Coordinates that discretize the vertical depth of the soil layer.

  Attributes:
    centers: center depth of each soil layer, starting at the level closest to
      the soil surface. Must be monotonically increasing.
  """

  centers: np.ndarray

  def __init__(self, centers: Iterable[float] | np.ndarray):
    validate_nonnegative_depth(centers)
    validate_increasing_depth(centers)
    centers = np.asarray(centers, dtype=np.float32)
    object.__setattr__(self, 'centers', centers)

  @property
  def dims(self) -> tuple[str, ...]:
    return ('soil_levels',)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.centers.shape

  @property
  def fields(self):
    return {'soil_levels': cx.field(self.centers, self)}

  def map_indexers(
      self,
      indexers: dict[str | cx.Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | cx.Coordinate, Any], set[str | cx.Coordinate]]:
    return coordax.experimental.map_indexers_using_ticks(
        self,
        indexers,
        ticks_are_sorted=True,
        method=method,
    )

  def _isel(self, indexers: dict[str | cx.Coordinate, Any]) -> cx.Coordinate:
    key = self.dims[0] if self.dims[0] in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return cx.Scalar()

    return SoilLevels(self.centers[indexer])

  def asdict(self) -> dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def _components(self):
    return (ArrayKey(self.centers),)

  def __eq__(self, other):
    return (
        isinstance(other, SoilLevels)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __fiddle_serialize__(self):
    return {'centers': self.centers.tolist()}

  @classmethod
  def with_era5_levels(cls, layers=(0, 1, 2, 3)):
    era5_centers = [3.5, 17.5, 64, 194.5]
    centers = [era5_centers[i] for i in layers]
    return cls(centers=centers)

  @classmethod
  def from_xarray(
      cls,
      dims: tuple[str, ...],
      coords: xarray.Coordinates,
  ) -> Self | cx.coords.NoCoordinateMatch:
    dim = dims[0]
    if dim != 'soil_levels':
      return cx.coords.NoCoordinateMatch(
          f'Leading dimension {dim!r} != "soil_levels"'
      )
    name = dim
    if coords[name].ndim != 1:
      return cx.coords.NoCoordinateMatch(
          'SoilLevels coordinate is not a 1D array'
      )
    got = coords[name].data
    return cls(centers=got)


#
# Wrapper coordinates
#
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CoordinateWithPadding(cx.Coordinate):
  """Coordinate wrapper that represents a coordinate with padding."""

  coordinate: cx.Coordinate
  pad_sizes: dict[str, tuple[int, int]]

  def __post_init__(self):
    if set(self.coordinate.dims) != set(self.pad_sizes.keys()):
      raise ValueError(
          f'{self.pad_sizes=} are expected to specify pads for each dimension '
          f'in {self.coordinate=}'
      )

  @property
  def dims(self) -> tuple[str, ...]:
    return tuple(f'padded_{d}' for d in self.coordinate.dims)

  @property
  def shape(self) -> tuple[int, ...]:
    dims_and_shapes = zip(self.coordinate.dims, self.coordinate.shape)
    return tuple(s + sum(self.pad_sizes[d]) for d, s in dims_and_shapes)  # pyrefly: ignore[bad-index]

  @property
  def fields(self):
    return {}
