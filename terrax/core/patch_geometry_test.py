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

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
from terrax.core import coordinates
from terrax.core import patch_geometry


class PatchGeometryTest(parameterized.TestCase):

  def test_slant_path_and_zenith_from_scan_angle(self):
    patch = coordinates.SwathPatch.atms(scanlines=4)
    _, fov_axis = patch.axes
    zenith = patch_geometry.zenith_from_scan_angle(
        patch.scan_angles, satellite_altitude_km=patch.satellite_altitude_km
    )
    self.assertEqual(zenith.coordinate, fov_axis)
    self.assertGreater(zenith.data[-1], 62.0)
    self.assertLess(zenith.data[-1], 65.0)

    sec_z = patch_geometry.slant_path_factor(zenith)
    expected_sec = cx.cpmap(lambda z: 1.0 / jnp.cos(jnp.deg2rad(z)))(zenith)
    cx.testing.assert_fields_allclose(sec_z, expected_sec, rtol=1e-5)

  def test_spherical_centroid_and_outline_across_patch_types(self):
    patch = coordinates.SwathPatch(
        scanlines=6,
        fov=4,
        max_scan_angle=52.725,
        satellite_altitude_km=833.0,
    )
    lons_1d = np.array([358.0, 359.0, 1.0, 2.0], dtype=np.float32)
    lats_1d = np.array([-10.0, -6.0, -2.0, 2.0, 6.0, 10.0], dtype=np.float32)
    lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)
    lon_f = cx.field(lon_2d, patch)
    lat_f = cx.field(lat_2d, patch)

    with self.subTest('spherical_centroid_across_prime_meridian'):
      lon_c, lat_c = patch_geometry.spherical_centroid(lon_f, lat_f)
      wrapped_c = cx.cpmap(lambda x: jnp.mod(x + 180.0, 360.0) - 180.0)(lon_c)
      cx.testing.assert_fields_allclose(
          wrapped_c, cx.field(np.float32(0.0)), atol=1e-4
      )
      cx.testing.assert_fields_allclose(
          lat_c, cx.field(np.float32(0.0)), atol=1e-4
      )

    with self.subTest('patch_outline_lon_lat_full_and_partial'):
      lon_out, lat_out = patch_geometry.patch_outline_lon_lat(lon_f, lat_f)
      expected_pts = 2 * patch.scanlines + 2 * patch.fov - 3
      outline_axis = cx.DummyAxis('outline_point', expected_pts)
      self.assertEqual(lon_out.coordinate, outline_axis)
      self.assertEqual(lat_out.coordinate, outline_axis)
      np.testing.assert_allclose(lon_out.data[0], lon_out.data[-1])
      np.testing.assert_allclose(lat_out.data[0], lat_out.data[-1])

      partial_lat = lat_2d.copy()
      partial_lat[4:, :] = np.nan
      partial_lon = lon_2d.copy()
      partial_lon[4:, :] = np.nan
      lon_p_out, lat_p_out = patch_geometry.patch_outline_lon_lat(
          cx.field(partial_lon, patch), cx.field(partial_lat, patch)
      )
      self.assertTrue(np.all(np.isfinite(lon_p_out.data)))
      self.assertTrue(np.all(np.isfinite(lat_p_out.data)))
      np.testing.assert_allclose(np.max(lat_p_out.data), lats_1d[3])

    with self.subTest('cartesian_positions_sub_satellite_and_viewing_vectors'):
      ground_xyz, sat_xyz = patch_geometry.cartesian_positions(
          patch, lon_f, lat_f
      )
      cartesian_axis = cx.LabeledAxis('cartesian', np.array(['x', 'y', 'z']))
      scanline_axis, _ = patch.axes
      self.assertEqual(
          ground_xyz.coordinate, cx.coords.compose(patch, cartesian_axis)
      )
      self.assertEqual(
          sat_xyz.coordinate, cx.coords.compose(scanline_axis, cartesian_axis)
      )

      lon_track, lat_track = patch_geometry.sub_satellite_track(
          patch, lon_f, lat_f
      )
      wrapped_lon = cx.cmap(lambda x: jnp.mod(x + 180.0, 360.0) - 180.0)(
          lon_track
      )
      cx.testing.assert_fields_allclose(
          wrapped_lon,
          cx.field(np.zeros(6, dtype=np.float32), scanline_axis),
          atol=1e-4,
      )
      cx.testing.assert_fields_allclose(
          lat_track,
          cx.field(lats_1d, scanline_axis),
          atol=5e-3,
      )
      nan_f = cx.field(np.full(patch.shape, np.nan, dtype=np.float32), patch)
      lon_n_nan, lat_n_nan = patch_geometry.sub_satellite_track(
          patch, nan_f, nan_f
      )
      self.assertTrue(np.all(np.isnan(lon_n_nan.data)))
      self.assertTrue(np.all(np.isnan(lat_n_nan.data)))

      view_vec = patch_geometry.viewing_vectors(patch, lon_f, lat_f)
      self.assertEqual(
          view_vec.coordinate, cx.coords.compose(patch, cartesian_axis)
      )
      view_norm = cx.cmap(jnp.linalg.norm)(view_vec.untag(cartesian_axis))
      cx.testing.assert_fields_allclose(
          view_norm,
          cx.field(np.ones(patch.shape, dtype=np.float32), patch),
          rtol=1e-5,
      )

  def test_polar_and_antipodal_float32_precision_and_gradients(self):
    # At the North Pole (lat0 = 90 deg), small offsets of 0.005 and 0.010 deg
    # retain full float32 precision and finite pole gradients via arctan2.
    pole_patch = coordinates.LocalGridPatch(
        x_nodes=5, y_nodes=1, dx_deg=0.005, dy_deg=0.005
    )
    _, lat_pole = patch_geometry.local_patch_lon_lat(
        pole_patch, cx.field(np.float32(0.0)), cx.field(np.float32(90.0))
    )
    # Nodes along patch_x are [-0.010, -0.005, 0.0, +0.005, +0.010] deg from
    # the pole -> true latitudes [89.990, 89.995, 90.0, 89.995, 89.990].
    np.testing.assert_allclose(
        lat_pole.data[:, 0],
        np.array([89.990, 89.995, 90.0, 89.995, 89.990], dtype=np.float32),
        atol=2e-5,
    )
    pole_grad = jax.grad(
        lambda lat0: jnp.sum(
            patch_geometry.local_patch_lon_lat(
                pole_patch, cx.field(np.float32(0.0)), cx.field(lat0)
            )[1].data
        )
    )(np.float32(90.0))
    self.assertTrue(np.isfinite(float(pole_grad)))

    # Near-antipodal distance (179.990 deg) and 0 deg gradient via Vincenty.
    dist_antipodal = patch_geometry.great_circle_distance_deg(
        cx.field(np.float32(0.0)),
        cx.field(np.float32(0.0)),
        cx.field(np.float32(179.990)),
        cx.field(np.float32(0.0)),
    )
    np.testing.assert_allclose(
        float(dist_antipodal.data), 179.990, atol=2e-5
    )
    zero_dist_grad = jax.grad(
        lambda lon2: patch_geometry.great_circle_distance_deg(
            cx.field(np.float32(10.0)),
            cx.field(np.float32(20.0)),
            cx.field(lon2),
            cx.field(np.float32(20.0)),
        ).data
    )(np.float32(10.0))
    self.assertTrue(np.isfinite(float(zero_dist_grad)))

    # Batched center_lat and azimuth_deg with scalar center_lon.
    lat_ax = cx.SizedAxis('transect', 2)
    az_ax = cx.SizedAxis('az', 3)
    lon_b, lat_b = patch_geometry.local_patch_lon_lat(
        pole_patch,
        cx.field(np.float32(30.0)),
        cx.field(np.array([-20.0, 20.0], dtype=np.float32), lat_ax),
        azimuth_deg=cx.field(
            np.array([0.0, 45.0, 90.0], dtype=np.float32), az_ax
        ),
    )
    self.assertEqual(
        lon_b.coordinate, cx.coords.compose(lat_ax, az_ax, pole_patch)
    )
    self.assertEqual(lon_b.coordinate, lat_b.coordinate)

  @parameterized.named_parameters(
      dict(testcase_name='equator_prime_meridian', lon0=0.0, lat0=0.0, az=0.0),
      dict(testcase_name='dateline_midlat', lon0=359.0, lat0=45.0, az=25.0),
      dict(testcase_name='north_pole', lon0=120.0, lat0=89.0, az=0.0),
      dict(testcase_name='south_pole', lon0=240.0, lat0=-88.0, az=-15.0),
  )
  def test_local_patch_so3_forward_and_inverse_roundtrip(
      self, lon0: float, lat0: float, az: float
  ):
    local_patch = coordinates.LocalGridPatch(
        x_nodes=9, y_nodes=7, dx_deg=1.5, dy_deg=1.0
    )
    c_lon = cx.field(np.float32(lon0))
    c_lat = cx.field(np.float32(lat0))
    lon_g, lat_g = patch_geometry.local_patch_lon_lat(
        local_patch, c_lon, c_lat, azimuth_deg=az
    )
    self.assertEqual(lon_g.coordinate, local_patch)
    self.assertEqual(lat_g.coordinate, local_patch)

    # Center node (4, 3) has (patch_x, patch_y) == (0, 0), so its global coords
    # must equal (lon0, lat0).
    np.testing.assert_allclose(lon_g.data[4, 3], lon0, atol=1e-4)
    np.testing.assert_allclose(lat_g.data[4, 3], lat0, atol=1e-4)

    # Inverse SO(3) rotation must recover (patch_x, patch_y) across the patch.
    xi_rec, eta_rec = patch_geometry.lon_lat_to_local_patch(
        lon_g, lat_g, c_lon, c_lat, azimuth_deg=az
    )
    expected_xi = local_patch.fields['patch_x'].broadcast_like(local_patch)
    expected_eta = local_patch.fields['patch_y'].broadcast_like(local_patch)
    cx.testing.assert_fields_allclose(xi_rec, expected_xi, atol=5e-4)
    cx.testing.assert_fields_allclose(eta_rec, expected_eta, atol=5e-4)

  def test_extract_lon_lat_patch_wrapping_and_exactness(self):
    grid = coordinates.LonLatGrid.T21()
    lon_1d = grid.fields['longitude'].data
    lat_1d = grid.fields['latitude'].data
    field_2d = lon_1d[:, None] + 10.0 * lat_1d[None, :]
    f_grid = cx.field(field_2d.astype(np.float32), grid)

    ll_patch = coordinates.LonLatPatch(longitude_nodes=8, latitude_nodes=6)
    patch_axis = cx.LabeledAxis('patch_index', np.arange(2))
    c_lon = cx.field(np.array([0.0, 180.0], dtype=np.float32), patch_axis)
    c_lat = cx.field(np.array([0.0, 88.0], dtype=np.float32), patch_axis)

    extracted = patch_geometry.extract_lon_lat_patch(
        {'f': f_grid}, ll_patch, c_lon, c_lat, include_lon_lat=True
    )
    expected_coord = cx.coords.compose(patch_axis, ll_patch)
    self.assertEqual(extracted['f'].coordinate, expected_coord)
    self.assertEqual(extracted['longitude'].coordinate, expected_coord)
    self.assertEqual(extracted['latitude'].coordinate, expected_coord)

    # Extracted values must match the analytic formula on extracted lon/lat
    # with zero interpolation error.
    expected_f = extracted['longitude'] + 10.0 * extracted['latitude']
    cx.testing.assert_fields_allclose(extracted['f'], expected_f, rtol=1e-6)

    # Padded LonLatGrid must be rejected cleanly.
    padded_grid = coordinates.LonLatGrid(
        longitude_nodes=64, latitude_nodes=32, lon_lat_padding=(2, 2)
    )
    with self.assertRaisesRegex(ValueError, 'Padded source grid'):
      patch_geometry.lon_lat_patch_indices(
          padded_grid, ll_patch, c_lon, c_lat
      )

    # Even window width brackets c_lon / c_lat symmetrically rather than
    # shifting exclusively left.
    even_grid = coordinates.LonLatGrid(
        longitude_nodes=36,
        latitude_nodes=19,
        latitude_spacing='equiangular_with_poles',
    )  # 10 deg spacing: lon = [0, 10, 20, ...], lat = [-90, -80, ..., 20, ...]
    even_patch = coordinates.LonLatPatch(longitude_nodes=2, latitude_nodes=2)
    lon_idx, lat_idx = patch_geometry.lon_lat_patch_indices(
        even_grid,
        even_patch,
        cx.field(np.float32(14.0)),
        cx.field(np.float32(14.0)),
    )
    np.testing.assert_array_equal(lon_idx.data, [1, 2])  # [10, 20] deg
    np.testing.assert_array_equal(lat_idx.data, [10, 11])  # [10, 20] deg

  def test_patch_blending_weights_and_fibonacci_sphere(self):
    patch = coordinates.LocalGridPatch(
        x_nodes=11, y_nodes=11, dx_deg=1.0, dy_deg=1.0
    )
    w = patch_geometry.patch_blending_weights(patch, taper_fraction=0.3)
    self.assertEqual(w.coordinate, patch)
    # Center is 1.0, corners are min_weight (1e-3).
    np.testing.assert_allclose(float(w.data[5, 5]), 1.0, rtol=1e-5)
    self.assertLess(float(w.data[0, 0]), 0.01)
    self.assertGreater(float(w.data[0, 0]), 0.0)

    c_lon, c_lat = patch_geometry.fibonacci_sphere_lon_lat(24)
    self.assertEqual(c_lon.shape, (24,))
    self.assertEqual(c_lat.shape, (24,))
    self.assertTrue(np.all((c_lon.data >= 0.0) & (c_lon.data < 360.0)))
    self.assertTrue(np.all((c_lat.data >= -90.0) & (c_lat.data <= 90.0)))


if __name__ == '__main__':
  absltest.main()
