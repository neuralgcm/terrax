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

from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import jax
import numpy as np
from terrax.metrics import aggregation
from terrax.metrics import evaluators
from terrax.metrics import stability_metrics


class IsFiniteTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='any_mode_no_nans',
          nan_mode=stability_metrics.NanMode.ANY,
          pred_data=np.array([1.0, 2.0, 3.0]),
          expected=1.0,
      ),
      dict(
          testcase_name='any_mode_some_nans',
          nan_mode=stability_metrics.NanMode.ANY,
          pred_data=np.array([1.0, np.nan, 3.0]),
          expected=0.0,
      ),
      dict(
          testcase_name='any_mode_all_nans',
          nan_mode=stability_metrics.NanMode.ANY,
          pred_data=np.array([np.nan, np.nan, np.nan]),
          expected=0.0,
      ),
      dict(
          testcase_name='all_mode_no_nans',
          nan_mode=stability_metrics.NanMode.ALL,
          pred_data=np.array([1.0, 2.0, 3.0]),
          expected=1.0,
      ),
      dict(
          testcase_name='all_mode_some_nans',
          nan_mode=stability_metrics.NanMode.ALL,
          pred_data=np.array([1.0, np.nan, 3.0]),
          expected=1.0,
      ),
      dict(
          testcase_name='all_mode_all_nans',
          nan_mode=stability_metrics.NanMode.ALL,
          pred_data=np.array([np.nan, np.nan, np.nan]),
          expected=0.0,
      ),
  )
  def test_nan_modes(self, nan_mode, pred_data, expected):
    """Tests IsFinite with event_dims=None (reduce all dims)."""
    dim = cx.SizedAxis('spatial', len(pred_data))
    predictions = {'x': cx.field(pred_data, dim)}
    targets = {'x': cx.field(np.ones_like(pred_data), dim)}
    stat = stability_metrics.IsFinite(nan_mode=nan_mode)
    result = stat.compute(predictions, targets)
    cx.testing.assert_fields_allclose(result['x'], cx.field(expected))

  def test_is_finite_with_event_dims(self):
    """Tests that event_dims reduces only specified dims, preserving batch."""
    batch = cx.SizedAxis('batch', 3)
    spatial = cx.SizedAxis('spatial', 2)
    # batch=0: finite, batch=1: has NaN, batch=2: finite
    pred_data = np.array([[1.0, 2.0], [1.0, np.nan], [3.0, 4.0]])
    predictions = {'x': cx.field(pred_data, batch, spatial)}
    targets = {'x': cx.field(np.ones((3, 2)), batch, spatial)}
    stat = stability_metrics.IsFinite(
        nan_mode=stability_metrics.NanMode.ANY,
        event_dims=('spatial',),
    )
    result = stat.compute(predictions, targets)
    cx.testing.assert_fields_allclose(
        result['x'], cx.field(np.array([1.0, 0.0, 1.0]), batch)
    )

  def test_is_finite_missing_event_dims_skipped(self):
    """Tests that event_dims not present on a variable are skipped."""
    batch = cx.SizedAxis('batch', 2)
    spatial = cx.SizedAxis('spatial', 3)
    # Surface variable (no 'level' dim). event_dims includes 'level'.
    pred_data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    predictions = {'x': cx.field(pred_data, batch, spatial)}
    targets = {'x': cx.field(np.ones((2, 3)), batch, spatial)}
    stat = stability_metrics.IsFinite(
        nan_mode=stability_metrics.NanMode.ANY,
        event_dims=('spatial', 'level'),  # 'level' not present on x.
    )
    result = stat.compute(predictions, targets)
    # Only 'spatial' is reduced. batch=0 has NaN, batch=1 does not.
    cx.testing.assert_fields_allclose(
        result['x'], cx.field(np.array([0.0, 1.0]), batch)
    )


class AllFiniteTest(parameterized.TestCase):

  def test_all_finite_with_event_dims(self):
    """Tests cross-variable check with batch dimension preserved."""
    batch = cx.SizedAxis('batch', 3)
    spatial = cx.SizedAxis('spatial', 2)
    x_data = np.array([[1.0, 2.0], [np.nan, 2.0], [3.0, 4.0]])
    y_data = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 6.0]])
    predictions = {
        'x': cx.field(x_data, batch, spatial),
        'y': cx.field(y_data, batch, spatial),
    }
    targets = {
        'x': cx.field(np.ones((3, 2)), batch, spatial),
        'y': cx.field(np.ones((3, 2)), batch, spatial),
    }
    stat = stability_metrics.AllFinite(
        nan_mode=stability_metrics.NanMode.ANY,
        event_dims=('spatial',),
    )
    result = stat.compute(predictions, targets)
    # batch=0: both finite -> 1, batch=1: x NaN -> 0, batch=2: y NaN -> 0
    cx.testing.assert_fields_allclose(
        result['all_variables'],
        cx.field(np.array([1.0, 0.0, 0.0]), batch),
    )


class StabilityEvaluatorTest(parameterized.TestCase):

  def test_evaluator_is_finite(self):
    """Tests IsFinite used inside an Evaluator with event_dims."""
    batch = cx.SizedAxis('batch', 3)
    spatial = cx.SizedAxis('spatial', 2)
    predictions = {
        'x': cx.field(
            np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]]), batch, spatial
        ),
        'y': cx.field(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), batch, spatial
        ),
    }
    targets = {
        'x': cx.field(np.ones((3, 2)), batch, spatial),
        'y': cx.field(np.ones((3, 2)), batch, spatial),
    }
    metric = stability_metrics.IsFinite(event_dims=('spatial',))
    ev = evaluators.Evaluator(
        metrics={'stability': metric},
        aggregators=aggregation.Aggregator(dims_to_reduce=['batch']),
    )
    values = ev.evaluate_metrics(predictions, targets)
    # x: batch 0 has NaN -> [0, 1, 1] -> mean = 2/3
    # y: all finite -> [1, 1, 1] -> mean = 1.0
    cx.testing.assert_fields_allclose(
        values['stability']['x'], cx.field(2.0 / 3.0)
    )
    cx.testing.assert_fields_allclose(
        values['stability']['y'], cx.field(1.0)
    )

  def test_evaluator_all_finite(self):
    """Tests AllFinite inside an Evaluator — cross-variable stability."""
    batch = cx.SizedAxis('batch', 4)
    spatial = cx.SizedAxis('spatial', 2)
    # x: NaN in batch=2 only. y: NaN in batch=1 only.
    # Per-variable: both 3/4 finite. But only batches 0,3 are fully stable.
    predictions = {
        'x': cx.field(
            np.array([[1, 2], [3, 4], [np.nan, 6], [7, 8.0]]), batch, spatial
        ),
        'y': cx.field(
            np.array([[1, 2], [np.nan, 4], [5, 6], [7, 8.0]]), batch, spatial
        ),
    }
    targets = {
        'x': cx.field(np.ones((4, 2)), batch, spatial),
        'y': cx.field(np.ones((4, 2)), batch, spatial),
    }
    metric = stability_metrics.AllFinite(event_dims=('spatial',))
    ev = evaluators.Evaluator(
        metrics={'stability': metric},
        aggregators=aggregation.Aggregator(dims_to_reduce=['batch']),
    )
    values = ev.evaluate_metrics(predictions, targets)
    # 2 out of 4 simulations fully stable = 0.5.
    cx.testing.assert_fields_allclose(
        values['stability']['all_variables'], cx.field(0.5)
    )


if __name__ == '__main__':
  jax.config.update('jax_traceback_filtering', 'off')
  absltest.main()
