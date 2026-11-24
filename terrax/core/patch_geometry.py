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

"""Spherical and local-patch geometry functions for Terrax patch coordinates."""

import functools
from typing import overload

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates

_EARTH_RADIUS_KM = 6371.0


def _safe_hypot(a: jax.Array, b: jax.Array) -> jax.Array:
  """Computes `hypot(a, b)` safely with finite zero gradient at `(0, 0)`."""
  m = jnp.maximum(jnp.abs(a), jnp.abs(b))
  nonzero = m > 0.0
  safe_m = jnp.where(nonzero, m, 1.0)
  sq = (a / safe_m) ** 2 + (b / safe_m) ** 2
  r = safe_m * jnp.sqrt(jnp.where(nonzero, sq, 1.0))
  return jnp.where(nonzero, r, 0.0)


def slant_path_factor(zenith_angle_deg: cx.Field) -> cx.Field:
  """Computes optical air-mass / slant-path factor 1 / cos(zenith_angle_deg)."""

  def _sec_deg(z_deg):
    z_rad = jnp.deg2rad(jnp.clip(jnp.abs(z_deg), 0.0, 89.0))
    return 1.0 / jnp.cos(z_rad)

  return cx.cmap(_sec_deg)(zenith_angle_deg)


def zenith_from_scan_angle(
    scan_angle_deg: cx.Field,
    satellite_altitude_km: float = 833.0,
) -> cx.Field:
  """Computes spherical-Earth local satellite zenith angle from scan angle."""
  r_ratio = (_EARTH_RADIUS_KM + satellite_altitude_km) / _EARTH_RADIUS_KM

  def _to_zenith(scan_deg):
    scan_rad = jnp.deg2rad(jnp.abs(scan_deg))
    sin_zenith = jnp.clip(r_ratio * jnp.sin(scan_rad), 0.0, 1.0)
    return jnp.rad2deg(jnp.arcsin(sin_zenith))

  return cx.cmap(_to_zenith)(scan_angle_deg)


def lon_lat_to_cartesian(
    longitude: cx.Field,
    latitude: cx.Field,
    *,
    radius: float | cx.Field = 1.0,
) -> cx.Field:
  """Converts (longitude, latitude) in degrees to 3D Cartesian (x, y, z).

  Args:
    longitude: Longitude field in degrees.
    latitude: Latitude field in degrees.
    radius: Sphere radius (scalar float or Field broadcastable with `longitude`
      and `latitude`). Defaults to `1.0` (unit sphere).

  Returns:
    Field with a trailing `cx.LabeledAxis('cartesian', ['x', 'y', 'z'])` axis.
  """
  cartesian_axis = cx.LabeledAxis('cartesian', np.array(['x', 'y', 'z']))

  def _to_xyz(lon_deg, lat_deg, r):
    lon_rad = jnp.deg2rad(lon_deg)
    lat_rad = jnp.deg2rad(lat_deg)
    x = r * jnp.cos(lat_rad) * jnp.cos(lon_rad)
    y = r * jnp.cos(lat_rad) * jnp.sin(lon_rad)
    z = r * jnp.sin(lat_rad)
    return jnp.stack([x, y, z], axis=-1)

  if isinstance(radius, cx.Field):
    return cx.cmap(_to_xyz, out_axes='leading')(
        longitude, latitude, radius
    ).tag(cartesian_axis)
  return cx.cmap(functools.partial(_to_xyz, r=radius), out_axes='leading')(
      longitude, latitude
  ).tag(cartesian_axis)


def sub_satellite_track(
    patch: coordinates.SwathPatch,
    longitude: cx.Field,
    latitude: cx.Field,
) -> tuple[cx.Field, cx.Field]:
  """Computes the sub-satellite ground track (lon, lat) along `'scanline'`.

  Approximates the sub-satellite nadir point for each scanline from the central
  cross-track field-of-view (for odd `patch.fov`) or the spherical midpoint of
  the two central fields-of-view (for even `patch.fov`), reducing the `'fov'`
  axis.

  Args:
    patch: `SwathPatch` coordinate present on `longitude` and `latitude`.
    longitude: Footprint longitude field in degrees containing `patch`.
    latitude: Footprint latitude field in degrees containing `patch`.

  Returns:
    Tuple `(track_lon, track_lat)` in degrees with the `'fov'` axis reduced.
  """
  _, fov_axis = patch.axes
  mid = patch.fov // 2
  center_indices = (
      jnp.asarray([mid], dtype=jnp.int32)
      if patch.fov % 2 == 1
      else jnp.asarray([mid - 1, mid], dtype=jnp.int32)
  )

  def _sub_satellite_point(lon_1d, lat_1d):
    lon_c = jnp.deg2rad(lon_1d[center_indices])
    lat_c = jnp.deg2rad(lat_1d[center_indices])
    valid = jnp.isfinite(lon_c) & jnp.isfinite(lat_c)
    lon_safe = jnp.where(valid, lon_c, 0.0)
    lat_safe = jnp.where(valid, lat_c, 0.0)
    x = jnp.where(valid, jnp.cos(lat_safe) * jnp.cos(lon_safe), 0.0)
    y = jnp.where(valid, jnp.cos(lat_safe) * jnp.sin(lon_safe), 0.0)
    z = jnp.where(valid, jnp.sin(lat_safe), 0.0)
    xm = jnp.sum(x)
    ym = jnp.sum(y)
    zm = jnp.sum(z)
    r_xy = _safe_hypot(xm, ym)
    norm = _safe_hypot(r_xy, zm)
    is_valid = jnp.any(valid) & (norm > 1e-6)
    lon_deg = jnp.mod(jnp.rad2deg(jnp.arctan2(ym, xm)), 360.0)
    lat_deg = jnp.rad2deg(jnp.arctan2(zm, r_xy))
    return (
        jnp.where(is_valid, lon_deg, jnp.nan),
        jnp.where(is_valid, lat_deg, jnp.nan),
    )

  return cx.cmap(_sub_satellite_point)(
      longitude.untag(fov_axis), latitude.untag(fov_axis)
  )


def cartesian_positions(
    patch: coordinates.SwathPatch,
    longitude: cx.Field,
    latitude: cx.Field,
    *,
    satellite_altitude_km: float | None = None,
    altitude_exaggeration: float = 1.0,
) -> tuple[cx.Field, cx.Field]:
  """Computes 3D Cartesian ground footprint and satellite track positions.

  Args:
    patch: `SwathPatch` coordinate present on `longitude` and `latitude`.
    longitude: Footprint longitude field in degrees containing `patch`.
    latitude: Footprint latitude field in degrees containing `patch`.
    satellite_altitude_km: Optional orbital altitude in km; defaults to
      `patch.satellite_altitude_km`.
    altitude_exaggeration: Optional visual exaggeration factor applied to
      `satellite_altitude_km / _EARTH_RADIUS_KM` for 3D globe rendering.

  Returns:
    Tuple `(ground_xyz, sat_xyz)` tagged with
    `cx.LabeledAxis('cartesian', ['x', 'y', 'z'])`.
  """
  alt_km = (
      patch.satellite_altitude_km
      if satellite_altitude_km is None
      else satellite_altitude_km
  )
  r_sat = 1.0 + altitude_exaggeration * (alt_km / _EARTH_RADIUS_KM)
  ground_xyz = lon_lat_to_cartesian(longitude, latitude, radius=1.0)
  lon_track, lat_track = sub_satellite_track(patch, longitude, latitude)
  sat_xyz = lon_lat_to_cartesian(lon_track, lat_track, radius=r_sat)
  return ground_xyz, sat_xyz


def viewing_vectors(
    patch: coordinates.SwathPatch,
    longitude: cx.Field,
    latitude: cx.Field,
    *,
    satellite_altitude_km: float | None = None,
) -> cx.Field:
  """Computes 3D unit line-of-sight vectors from footprints to satellite.

  Args:
    patch: `SwathPatch` coordinate present on `longitude` and `latitude`.
    longitude: Footprint longitude field in degrees containing `patch`.
    latitude: Footprint latitude field in degrees containing `patch`.
    satellite_altitude_km: Optional orbital altitude in km; defaults to
      `patch.satellite_altitude_km`.

  Returns:
    Field with `patch` (plus any batch axes) and
    `cx.LabeledAxis('cartesian', ['x', 'y', 'z'])` containing unit vectors
    pointing from each ground footprint toward the satellite position along
    `'scanline'`.
  """
  cartesian_axis = cx.LabeledAxis('cartesian', np.array(['x', 'y', 'z']))
  ground_xyz, sat_xyz = cartesian_positions(
      patch,
      longitude,
      latitude,
      satellite_altitude_km=satellite_altitude_km,
  )
  diff = (sat_xyz.broadcast_like(ground_xyz.coordinate) - ground_xyz).untag(
      cartesian_axis
  )

  def _normalize(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    valid = jnp.all(jnp.isfinite(v), axis=-1, keepdims=True) & (norm > 1e-6)
    safe_norm = jnp.where(valid, norm, 1.0)
    return jnp.where(valid, v / safe_norm, jnp.nan)

  return cx.cpmap(_normalize)(diff).tag(cartesian_axis)


def spherical_centroid(
    longitude: cx.Field,
    latitude: cx.Field,
    coord: cx.Coordinate | None = None,
) -> tuple[cx.Field, cx.Field]:
  """Computes the unit-sphere centroid (lon, lat) in degrees across `coord`.

  Args:
    longitude: Longitude field in degrees.
    latitude: Latitude field in degrees.
    coord: Coordinate to reduce over. If None, extracts a `PatchCoordinate` from
      `longitude.coordinate`.

  Returns:
    Tuple `(lon_centroid, lat_centroid)` in degrees with `coord` reduced.
  """
  if coord is None:
    coord = cx.coords.extract(longitude.coordinate, coordinates.PatchCoordinate)

  def _centroid(lon_arr, lat_arr):
    valid = jnp.isfinite(lon_arr) & jnp.isfinite(lat_arr)
    lon_rad = jnp.deg2rad(jnp.where(valid, lon_arr, 0.0))
    lat_rad = jnp.deg2rad(jnp.where(valid, lat_arr, 0.0))
    x = jnp.where(valid, jnp.cos(lat_rad) * jnp.cos(lon_rad), 0.0)
    y = jnp.where(valid, jnp.cos(lat_rad) * jnp.sin(lon_rad), 0.0)
    z = jnp.where(valid, jnp.sin(lat_rad), 0.0)
    xm = jnp.sum(x)
    ym = jnp.sum(y)
    zm = jnp.sum(z)
    r_xy = _safe_hypot(xm, ym)
    norm = _safe_hypot(r_xy, zm)
    is_valid = jnp.any(valid) & (norm > 1e-6)
    lon_c = jnp.mod(jnp.rad2deg(jnp.arctan2(ym, xm)), 360.0)
    lat_c = jnp.rad2deg(jnp.arctan2(zm, r_xy))
    return (
        jnp.where(is_valid, lon_c, jnp.nan),
        jnp.where(is_valid, lat_c, jnp.nan),
    )

  return cx.cmap(_centroid)(longitude.untag(coord), latitude.untag(coord))


def great_circle_distance_deg(
    lon1: cx.Field,
    lat1: cx.Field,
    lon2: cx.Field,
    lat2: cx.Field,
) -> cx.Field:
  """Computes great-circle angular distance in degrees between two points."""

  def _ang_dist(lam1_deg, phi1_deg, lam2_deg, phi2_deg):
    lam1 = jnp.deg2rad(lam1_deg)
    phi1 = jnp.deg2rad(phi1_deg)
    lam2 = jnp.deg2rad(lam2_deg)
    phi2 = jnp.deg2rad(phi2_deg)
    dlam = lam2 - lam1
    y = _safe_hypot(
        jnp.cos(phi2) * jnp.sin(dlam),
        jnp.cos(phi1) * jnp.sin(phi2)
        - jnp.sin(phi1) * jnp.cos(phi2) * jnp.cos(dlam),
    )
    x = jnp.sin(phi1) * jnp.sin(phi2) + (
        jnp.cos(phi1) * jnp.cos(phi2) * jnp.cos(dlam)
    )
    return jnp.rad2deg(jnp.arctan2(y, x))

  return cx.cmap(_ang_dist)(lon1, lat1, lon2, lat2)


def patch_outline_lon_lat(
    longitude: cx.Field,
    latitude: cx.Field,
    patch: coordinates.PatchCoordinate | None = None,
) -> tuple[cx.Field, cx.Field]:
  """Extracts closed perimeter (lon, lat) loop for any 2D `PatchCoordinate`.

  Handles partially-filled patches (where trailing rows are NaN) by clamping
  perimeter row indices to the first and last valid rows while keeping a static
  output axis length `2 * n_rows + 2 * n_cols - 3`.

  Args:
    longitude: Longitude field in degrees containing a 2D `PatchCoordinate`.
    latitude: Latitude field in degrees containing a 2D `PatchCoordinate`.
    patch: Optional explicit `PatchCoordinate` to trace. If None, extracted from
      `longitude.coordinate`.

  Returns:
    Tuple `(lon_outline, lat_outline)` tagged with
    `cx.DummyAxis('outline_point', 2 * n_rows + 2 * n_cols - 3)`.
  """
  if patch is None:
    patch = cx.coords.extract(longitude.coordinate, coordinates.PatchCoordinate)
  n_rows, n_cols = patch.shape
  if n_rows < 2 or n_cols < 2:
    raise ValueError(
        'patch_outline_lon_lat requires patch shape >= (2, 2), got'
        f' {patch.shape=}.'
    )
  n_pts = 2 * n_rows + 2 * n_cols - 3
  outline_axis = cx.DummyAxis('outline_point', n_pts)

  def _extract_outline(lon_2d, lat_2d):
    row_valid = (
        jnp.isfinite(lon_2d[:, 0])
        & jnp.isfinite(lon_2d[:, -1])
        & jnp.isfinite(lat_2d[:, 0])
        & jnp.isfinite(lat_2d[:, -1])
    )
    any_valid = jnp.any(row_valid)
    row_indices = jnp.arange(n_rows)
    first_valid = jnp.min(jnp.where(row_valid, row_indices, n_rows - 1))
    last_valid = jnp.max(jnp.where(row_valid, row_indices, 0))

    right_rows = jnp.clip(row_indices[1:], first_valid, last_valid)
    left_rows = jnp.clip(row_indices[:-1][::-1], first_valid, last_valid)

    def _trace_one(arr_2d):
      top = arr_2d[first_valid, :]
      right = arr_2d[right_rows, -1]
      bottom = arr_2d[last_valid, :-1][::-1]
      left = arr_2d[left_rows, 0]
      loop = jnp.concatenate([top, right, bottom, left], axis=0)
      return jnp.where(any_valid, loop, jnp.nan)

    return _trace_one(lon_2d), _trace_one(lat_2d)

  lon_out, lat_out = cx.cmap(_extract_outline, out_axes='leading')(
      longitude.untag(patch), latitude.untag(patch)
  )
  return lon_out.tag(outline_axis), lat_out.tag(outline_axis)


def local_patch_lon_lat(
    patch: coordinates.LocalGridPatch,
    center_lon: cx.Field,
    center_lat: cx.Field,
    *,
    azimuth_deg: float | cx.Field = 0.0,
) -> tuple[cx.Field, cx.Field]:
  """Maps a `LocalGridPatch` to global (lon, lat) via SO(3) spherical rotation.

  Embeds the regular local grid `(patch_x, patch_y)` defined around `(0, 0)` on
  the rotated equator onto the sphere centered at `(center_lon, center_lat)`.
  Because the patch is constructed on its own rotated equator, physical grid
  spacing is quasi-uniform everywhere on Earth, including over the poles and
  across the prime meridian / dateline.

  Args:
    patch: `LocalGridPatch` defining local angular coordinates `('patch_x',
      'patch_y')`.
    center_lon: Patch center longitude in degrees (scalar or batch Field).
    center_lat: Patch center latitude in degrees (scalar or batch Field).
    azimuth_deg: Optional counter-clockwise tangent-plane rotation angle in
      degrees (0 means `patch_x` points locally East and `patch_y` points
      locally North at the patch center).

  Returns:
    Tuple `(longitude, latitude)` in degrees tagged with `patch` (plus any batch
    coordinates on `center_lon` and `center_lat`).
  """
  xi_deg = patch.fields['patch_x'].broadcast_like(patch)
  eta_deg = patch.fields['patch_y'].broadcast_like(patch)
  az_field = (
      azimuth_deg
      if isinstance(azimuth_deg, cx.Field)
      else cx.field(jnp.asarray(azimuth_deg, dtype=jnp.float32))
  )

  def _rotate_to_global(xi_d, eta_d, lon0_d, lat0_d, alpha_d):
    xi = jnp.deg2rad(xi_d)
    eta = jnp.deg2rad(eta_d)
    lon0 = jnp.deg2rad(lon0_d)
    lat0 = jnp.deg2rad(lat0_d)
    alpha = jnp.deg2rad(alpha_d)

    # Unit vector in local rotated-equator frame.
    x_l = jnp.cos(eta) * jnp.cos(xi)
    y_l = jnp.cos(eta) * jnp.sin(xi)
    z_l = jnp.sin(eta)

    # 1. Tangent-plane rotation around x_l by alpha.
    x_1 = x_l
    y_1 = y_l * jnp.cos(alpha) - z_l * jnp.sin(alpha)
    z_1 = y_l * jnp.sin(alpha) + z_l * jnp.cos(alpha)

    # 2. Pitch (0, 0) up to latitude lat0.
    x_2 = x_1 * jnp.cos(lat0) - z_1 * jnp.sin(lat0)
    y_2 = y_1
    z_2 = x_1 * jnp.sin(lat0) + z_1 * jnp.cos(lat0)

    # 3. Rotate around z-axis to longitude lon0.
    x = x_2 * jnp.cos(lon0) - y_2 * jnp.sin(lon0)
    y = x_2 * jnp.sin(lon0) + y_2 * jnp.cos(lon0)
    z = z_2

    lon_out = jnp.mod(jnp.rad2deg(jnp.arctan2(y, x)), 360.0)
    lat_out = jnp.rad2deg(jnp.arctan2(z, _safe_hypot(x, y)))
    return lon_out, lat_out

  seen_dims: set[str | None] = set(patch.dims)
  batch_axes = []
  for f_in in (center_lon, center_lat, az_field):
    for ax in cx.coords.canonicalize(f_in.coordinate):
      if not set(ax.dims) & seen_dims:
        batch_axes.append(ax)
        seen_dims.update(ax.dims)
  out_coord = cx.coords.compose(*batch_axes, patch)
  return cx.cmap(_rotate_to_global, out_axes=out_coord.dims)(
      xi_deg, eta_deg, center_lon, center_lat, az_field
  )


def lon_lat_to_local_patch(
    longitude: cx.Field,
    latitude: cx.Field,
    center_lon: cx.Field,
    center_lat: cx.Field,
    *,
    azimuth_deg: float | cx.Field = 0.0,
) -> tuple[cx.Field, cx.Field]:
  """Projects global (longitude, latitude) into `(patch_x, patch_y)` degrees.

  Applies the exact inverse SO(3) rotation of `local_patch_lon_lat` to map
  arbitrary global coordinates `(longitude, latitude)` into local rotated-sphere
  coordinates `(xi_deg, eta_deg)` relative to `(center_lon, center_lat)`.

  Args:
    longitude: Global longitude field in degrees.
    latitude: Global latitude field in degrees.
    center_lon: Target patch center longitude in degrees.
    center_lat: Target patch center latitude in degrees.
    azimuth_deg: Tangent-plane rotation angle in degrees matching
      `local_patch_lon_lat`.

  Returns:
    Tuple `(patch_x_deg, patch_y_deg)` in degrees in `(-180, 180] x [-90, 90]`.
  """
  az_field = (
      azimuth_deg
      if isinstance(azimuth_deg, cx.Field)
      else cx.field(jnp.asarray(azimuth_deg, dtype=jnp.float32))
  )

  def _rotate_to_local(lon_d, lat_d, lon0_d, lat0_d, alpha_d):
    valid = jnp.isfinite(lon_d) & jnp.isfinite(lat_d)
    lon = jnp.deg2rad(jnp.where(valid, lon_d, 0.0))
    lat = jnp.deg2rad(jnp.where(valid, lat_d, 0.0))
    lon0 = jnp.deg2rad(lon0_d)
    lat0 = jnp.deg2rad(lat0_d)
    alpha = jnp.deg2rad(alpha_d)

    # 1. Undo z-rotation by -lon0.
    dlon = lon - lon0
    x_2 = jnp.cos(lat) * jnp.cos(dlon)
    y_2 = jnp.cos(lat) * jnp.sin(dlon)
    z_2 = jnp.sin(lat)

    # 2. Undo latitude pitch by -lat0.
    x_1 = x_2 * jnp.cos(lat0) + z_2 * jnp.sin(lat0)
    y_1 = y_2
    z_1 = -x_2 * jnp.sin(lat0) + z_2 * jnp.cos(lat0)

    # 3. Undo tangent-plane rotation by -alpha.
    x_l = x_1
    y_l = y_1 * jnp.cos(alpha) + z_1 * jnp.sin(alpha)
    z_l = -y_1 * jnp.sin(alpha) + z_1 * jnp.cos(alpha)

    xi_deg = jnp.rad2deg(jnp.arctan2(y_l, x_l))
    eta_deg = jnp.rad2deg(jnp.arctan2(z_l, _safe_hypot(x_l, y_l)))
    return (
        jnp.where(valid, xi_deg, jnp.nan),
        jnp.where(valid, eta_deg, jnp.nan),
    )

  return cx.cmap(_rotate_to_local)(
      longitude, latitude, center_lon, center_lat, az_field
  )


def lon_lat_patch_indices(
    grid: coordinates.LonLatGrid,
    patch: coordinates.LonLatPatch,
    center_lon: cx.Field,
    center_lat: cx.Field,
) -> tuple[cx.Field, cx.Field]:
  """Computes exact `LonLatGrid` index slices for a `LonLatPatch` window.

  Wraps periodically along longitude and clamps the latitude window to stay
  within `[0, grid.latitude_nodes - patch.latitude_nodes]`.

  Args:
    grid: Source `LonLatGrid` from which to slice the patch.
    patch: Target `LonLatPatch` specifying `(longitude_nodes, latitude_nodes)`.
    center_lon: Desired patch center longitude in degrees.
    center_lat: Desired patch center latitude in degrees.

  Returns:
    Tuple `(lon_indices, lat_indices)` tagged with `patch.axes[0]`
    (`'patch_lon'`) and `patch.axes[1]` (`'patch_lat'`) respectively.
  """
  if grid.lon_lat_padding != (0, 0):
    raise ValueError(f'Padded source grid is not supported: {grid=}.')
  n_lon, n_lat = grid.shape
  w_lon, w_lat = patch.shape
  if w_lon > n_lon or w_lat > n_lat:
    raise ValueError(
        f'{patch.shape=} cannot exceed source {grid.shape=}.'
    )
  lon_src = grid.fields['longitude'].data
  lat_src = grid.fields['latitude'].data
  patch_lon_axis, patch_lat_axis = patch.axes

  def _lon_slice(c_lon):
    signed_diff = jnp.mod(c_lon - lon_src + 180.0, 360.0) - 180.0
    center_idx = jnp.argmin(jnp.abs(signed_diff))
    if w_lon % 2 == 0:
      shift = jnp.where(
          signed_diff[center_idx] >= 0.0, (w_lon // 2) - 1, w_lon // 2
      )
    else:
      shift = w_lon // 2
    start_idx = center_idx - shift
    return jnp.mod(start_idx + jnp.arange(w_lon, dtype=jnp.int32), n_lon)

  def _lat_slice(c_lat):
    signed_diff = c_lat - lat_src
    center_idx = jnp.argmin(jnp.abs(signed_diff))
    if w_lat % 2 == 0:
      shift = jnp.where(
          signed_diff[center_idx] >= 0.0, (w_lat // 2) - 1, w_lat // 2
      )
    else:
      shift = w_lat // 2
    start_idx = jnp.clip(center_idx - shift, 0, n_lat - w_lat)
    return start_idx + jnp.arange(w_lat, dtype=jnp.int32)

  lon_idx = cx.cmap(_lon_slice, out_axes='leading')(center_lon).tag(
      patch_lon_axis
  )
  lat_idx = cx.cmap(_lat_slice, out_axes='leading')(center_lat).tag(
      patch_lat_axis
  )
  return lon_idx, lat_idx


@overload
def extract_lon_lat_patch(
    inputs: cx.Field,
    patch: coordinates.LonLatPatch,
    center_lon: cx.Field,
    center_lat: cx.Field,
    *,
    skip_missing: bool = False,
    include_lon_lat: bool = True,
) -> cx.Field:
  ...


@overload
def extract_lon_lat_patch(
    inputs: dict[str, cx.Field],
    patch: coordinates.LonLatPatch,
    center_lon: cx.Field,
    center_lat: cx.Field,
    *,
    skip_missing: bool = False,
    include_lon_lat: bool = True,
) -> dict[str, cx.Field]:
  ...


def extract_lon_lat_patch(
    inputs: cx.Field | dict[str, cx.Field],
    patch: coordinates.LonLatPatch,
    center_lon: cx.Field,
    center_lat: cx.Field,
    *,
    skip_missing: bool = False,
    include_lon_lat: bool = True,
) -> cx.Field | dict[str, cx.Field]:
  """Extracts index-aligned `LonLatPatch` windows from `LonLatGrid` fields.

  Slices `LonLatGrid` fields onto `patch` without interpolation by gathering
  consecutive grid nodes around `(center_lon, center_lat)`.

  Args:
    inputs: A `cx.Field` or dict of `cx.Field`s on a `LonLatGrid`.
    patch: Target `LonLatPatch` coordinate.
    center_lon: Patch center longitude in degrees (scalar or batch Field).
    center_lat: Patch center latitude in degrees (scalar or batch Field).
    skip_missing: If True, returns fields without a `LonLatGrid` unchanged.
    include_lon_lat: If True and `inputs` is a dict, includes `'longitude'` and
      `'latitude'` fields on `patch` in the returned dict.

  Returns:
    Field or dict of Fields with `LonLatGrid` replaced by `patch` (and any batch
    coordinates on `center_lon` and `center_lat`).
  """
  if center_lon.coordinate != center_lat.coordinate:
    raise ValueError(
        'center_lon and center_lat must have the same coordinate, got'
        f' {center_lon.coordinate=} vs {center_lat.coordinate=}.'
    )

  indices_cache: dict[coordinates.LonLatGrid, tuple[cx.Field, cx.Field]] = {}

  def _get_indices(
      source_grid: coordinates.LonLatGrid,
  ) -> tuple[cx.Field, cx.Field]:
    if source_grid not in indices_cache:
      indices_cache[source_grid] = lon_lat_patch_indices(
          source_grid, patch, center_lon, center_lat
      )
    return indices_cache[source_grid]

  def _slice_field(
      f: cx.Field,
  ) -> tuple[cx.Field, coordinates.LonLatGrid | None]:
    try:
      source_grid = cx.coords.extract(f.coordinate, coordinates.LonLatGrid)
    except ValueError:
      if skip_missing:
        return f, None
      raise
    lon_idx, lat_idx = _get_indices(source_grid)
    patch_lon_axis, patch_lat_axis = patch.axes

    def _gather_2d(grid_2d, i_lon, i_lat):
      grid_2d = jnp.asarray(grid_2d)
      return grid_2d[i_lon[:, None], i_lat[None, :]]

    batch_axes = [
        c
        for c in cx.coords.canonicalize(center_lon.coordinate)
        if not cx.contains_dims(f, c)
    ]
    replacement = cx.coords.compose(
        *batch_axes, cx.DummyAxis(None, 1), cx.DummyAxis(None, 1)
    )
    out_template = cx.coords.replace_axes(
        f.coordinate, source_grid, replacement
    )
    sliced = cx.cmap(_gather_2d, out_axes=out_template.dims)(
        f.untag(source_grid),
        lon_idx.untag(patch_lon_axis),
        lat_idx.untag(patch_lat_axis),
    ).tag(patch)
    return sliced, source_grid

  if isinstance(inputs, cx.Field):
    sliced_f, _ = _slice_field(inputs)
    return sliced_f

  outputs = {}
  inferred_grid = None
  for k, v in inputs.items():
    outputs[k], g = _slice_field(v)
    if g is not None:
      inferred_grid = g

  if include_lon_lat and inferred_grid is not None:
    lon_idx, lat_idx = _get_indices(inferred_grid)
    lon_1d = inferred_grid.fields['longitude'].data
    lat_1d = inferred_grid.fields['latitude'].data
    lon_sub = cx.cmap(lambda idx: jnp.asarray(lon_1d)[idx])(lon_idx)
    lat_sub = cx.cmap(lambda idx: jnp.asarray(lat_1d)[idx])(lat_idx)
    target_coord = cx.coords.compose(center_lon.coordinate, patch)
    outputs['longitude'] = lon_sub.broadcast_like(target_coord)
    outputs['latitude'] = lat_sub.broadcast_like(target_coord)
  return outputs


def patch_blending_weights(
    patch: coordinates.PatchCoordinate,
    taper_fraction: float = 0.25,
    min_weight: float = 1e-3,
) -> cx.Field:
  """Computes 2D separable raised-cosine (Tukey) feathering weights on `patch`.

  Weights are `1.0` in the interior core of `patch` and taper smoothly toward
  `min_weight` near all four boundaries so overlapping patches blend without
  seam discontinuities when stitched onto a target grid.

  Args:
    patch: 2D `PatchCoordinate` (`SwathPatch`, `LonLatPatch`, or
      `LocalGridPatch`).
    taper_fraction: Fraction of each dimension in `[0, 0.5]` tapered at each
      edge (`0.0` gives uniform ones; `0.5` gives a full Hann window).
    min_weight: Strictly positive floor at the outermost boundary pixels so
      boundary cells covered by only a single patch still receive valid values.

  Returns:
    Field on `patch` with values in `[min_weight, 1.0]`.
  """
  if not (0.0 <= taper_fraction <= 0.5):
    raise ValueError(
        f'taper_fraction must be in [0, 0.5], got {taper_fraction=}.'
    )

  def _tukey_1d(n: int) -> jax.Array:
    if n <= 1 or taper_fraction == 0.0:
      return jnp.ones((n,), dtype=jnp.float32)
    u = jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)
    edge_dist = jnp.minimum(u, 1.0 - u)
    phase = jnp.clip(edge_dist / taper_fraction, 0.0, 1.0)
    w = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
    return min_weight + (1.0 - min_weight) * w

  n0, n1 = patch.shape
  with jax.ensure_compile_time_eval():
    w0 = _tukey_1d(n0)
    w1 = _tukey_1d(n1)
    w2d = w0[:, None] * w1[None, :]
    return cx.field(w2d, patch)


def fibonacci_sphere_lon_lat(
    num_points: int,
    *,
    axis_name: str = 'patch_index',
) -> tuple[cx.Field, cx.Field]:
  """Generates quasi-uniformly spaced (lon, lat) centers on the sphere.

  Args:
    num_points: Number of points to place on the sphere (`>= 1`).
    axis_name: Name of the `LabeledAxis` tagging the returned fields.

  Returns:
    Tuple `(center_lon, center_lat)` in degrees tagged with
    `cx.LabeledAxis(axis_name, np.arange(num_points))`.
  """
  if num_points < 1:
    raise ValueError(f'num_points must be >= 1, got {num_points=}.')
  axis = cx.LabeledAxis(axis_name, np.arange(num_points))
  with jax.ensure_compile_time_eval():
    idx = jnp.arange(num_points, dtype=jnp.float32) + 0.5
    golden_angle = jnp.pi * (3.0 - jnp.sqrt(5.0))
    z = 1.0 - (2.0 * idx) / float(num_points)
    r_xy = jnp.sqrt(
        jnp.maximum((4.0 * idx / num_points) * (1.0 - idx / num_points), 0.0)
    )
    lat_deg = jnp.rad2deg(jnp.arctan2(z, r_xy))
    lon_deg = jnp.mod(jnp.rad2deg(golden_angle * idx), 360.0)
    return cx.field(lon_deg, axis), cx.field(lat_deg, axis)
