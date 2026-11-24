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

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import numpy as np
from terrax.core import boundaries
from terrax.core import coordinates


class BoundariesTest(parameterized.TestCase):
  """Tests that boundary conditions correctly pad and trim boundary cells."""

  def test_lon_lat_boundary_values(self):
    """Tests that boundary ghost cells contain expected values."""
    bc = boundaries.LonLatBoundary()
    grid = coordinates.LonLatGrid.T21()  # [64, 32].
    grid_sizes = np.prod(grid.shape)
    # reshape to lat,lon and transpose to make range increment along longitude.
    f = cx.field(np.arange(grid_sizes).reshape(grid.shape[::-1]).T, grid)
    pad_sizes = {'longitude': (1, 1), 'latitude': (2, 2)}
    padded_f = bc.pad(f, pad_sizes)

    self.assertEqual(padded_f.shape, (66, 36))  # 64 + 1 + 1, 32 + 2 + 2.
    np.testing.assert_array_equal(
        padded_f.data[0, 2:-2], f.data[-1, :], err_msg='left_lon_padding'
    )
    np.testing.assert_array_equal(
        padded_f.data[-1, 2:-2], f.data[0, :], err_msg='right_lon_padding'
    )
    np.testing.assert_array_equal(
        padded_f.data[1:-1, -3], f.data[:, -1], err_msg='preserved_top_boundary'
    )
    np.testing.assert_array_equal(
        padded_f.data[1:-1, 2], f.data[:, 0], err_msg='preserved_bot_boundary'
    )
    # latitude boundary should be padded with values on the opposite side of the
    # pole. For 64 longitude points the diameter opposite side of 0 will be 32,
    # with 33, ... being to the left of it (clockwise direction) that pairs
    # with the following neighbors of `0` (i.e. `1`, `2`, ...).
    np.testing.assert_array_equal(
        padded_f.data[1:-1, 1],
        np.roll(f.data[:, 0], 32),
        err_msg='lat_boundary_nearest',
    )
    np.testing.assert_array_equal(
        padded_f.data[1:-1, 0],
        np.roll(f.data[:, 1], 32),
        err_msg='lat_boundary_next_nearest',
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='odd_padding',
          grid=coordinates.LonLatGrid.TL31(),
          pad_sizes={'longitude': (1, 1), 'latitude': (1, 1)},
          expected_padded_shape=(66, 34),
      ),
      dict(
          testcase_name='even_padding',
          grid=coordinates.LonLatGrid.T21(),
          pad_sizes={'longitude': (2, 2), 'latitude': (2, 2)},
          expected_padded_shape=(68, 36),
      ),
      dict(
          testcase_name='mixed_padding',
          grid=coordinates.LonLatGrid.T21(),
          pad_sizes={'longitude': (1, 4), 'latitude': (2, 3)},
          expected_padded_shape=(69, 37),
      ),
  )
  def test_lon_lat_boundary_pad_and_trim(
      self,
      grid: coordinates.LonLatGrid,
      pad_sizes: dict[str, tuple[int, int]],
      expected_padded_shape: tuple[int, ...],
  ):
    bc = boundaries.LonLatBoundary()
    rng = jax.random.PRNGKey(0)
    data = jax.random.normal(rng, grid.shape)

    with self.subTest('on_field_inputs'):
      f = cx.field(data, grid)
      padded = bc.pad(f, pad_sizes)
      self.assertEqual(padded.shape, expected_padded_shape)
      trimmed = bc.trim(padded)
      self.assertEqual(trimmed.shape, grid.shape)
      np.testing.assert_array_equal(trimmed.data, f.data)

    with self.subTest('on_array_inputs'):
      pad_width = tuple(pad_sizes[d] for d in grid.dims)
      padded = bc.pad_array(data, pad_width)
      self.assertEqual(padded.shape, expected_padded_shape)
      trimmed = bc.trim_array(padded, pad_width)
      self.assertEqual(trimmed.shape, data.shape)
      np.testing.assert_array_equal(trimmed, data)

  @parameterized.named_parameters(
      dict(testcase_name='edge_mode', mode='edge', constant_value=0.0),
      dict(testcase_name='constant_mode', mode='constant', constant_value=-1.0),
  )
  def test_patch_boundary_pad_and_trim(self, mode: str, constant_value: float):
    self.assertIs(boundaries.SwathBoundary, boundaries.PatchBoundary)
    bc = boundaries.PatchBoundary(mode=mode, constant_value=constant_value)
    patch = coordinates.SwathPatch(
        scanlines=12,
        fov=8,
        max_scan_angle=52.725,
        satellite_altitude_km=833.0,
    )
    pad_sizes = {'scanline': (2, 3), 'fov': (1, 2)}
    rng = jax.random.PRNGKey(42)
    data = jax.random.normal(rng, patch.shape)

    with self.subTest('on_field_inputs'):
      f = cx.field(data, patch)
      padded = bc.pad(f, pad_sizes)
      self.assertEqual(padded.shape, (17, 11))
      if mode == 'edge':
        np.testing.assert_array_equal(padded.data[0, 1:-2], f.data[0, :])
        np.testing.assert_array_equal(padded.data[-1, 1:-2], f.data[-1, :])
      else:
        np.testing.assert_array_equal(padded.data[0, :], constant_value)
      trimmed = bc.trim(padded)
      cx.testing.assert_fields_equal(trimmed, f)

    with self.subTest('on_local_grid_patch'):
      local_patch = coordinates.LocalGridPatch(
          x_nodes=8, y_nodes=6, dx_deg=0.5, dy_deg=0.5
      )
      f_local = cx.field(
          jax.random.normal(rng, local_patch.shape), local_patch
      )
      padded_local = bc.pad(f_local, {'patch_x': (1, 1), 'patch_y': (2, 2)})
      self.assertEqual(padded_local.shape, (10, 10))
      cx.testing.assert_fields_equal(bc.trim(padded_local), f_local)

    with self.subTest('on_multiple_padded_coordinates'):
      other_padded = coordinates.CoordinateWithPadding(
          cx.LabeledAxis('sensor_fov', np.arange(4)), {'sensor_fov': (1, 1)}
      )
      multi_coord = cx.coords.compose(other_padded, patch)
      f_multi = cx.field(
          jax.random.normal(rng, multi_coord.shape), multi_coord
      )
      padded_multi = bc.pad(f_multi, pad_sizes)
      cx.testing.assert_fields_equal(bc.trim(padded_multi), f_multi)

    with self.subTest('on_array_inputs'):
      pad_width = ((2, 3), (1, 2))
      padded_arr = bc.pad_array(data, pad_width)
      self.assertEqual(padded_arr.shape, (17, 11))
      trimmed_arr = bc.trim_array(padded_arr, pad_width)
      np.testing.assert_array_equal(trimmed_arr, data)


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
