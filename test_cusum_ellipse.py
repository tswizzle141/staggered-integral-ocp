"""Run: python -m unittest -v test_cusum_ellipse.py"""

import unittest

import numpy as np

from main import reference_state, ELLIPSE_A, ELLIPSE_B
from cusum_upgrade import (
    CUSUMAdaptiveQuantile,
    CoverageCalibratedCUSUM,
)


class IntegrationTests(unittest.TestCase):
    def test_ellipse_geometry_and_velocity(self):
        start = np.array([
            0., 0., 1., 0., 0., 0., 0., 0.,
        ])
        goal = start.copy()

        for t in np.linspace(0, 10, 101):
            ref = reference_state(
                t, start, goal, 10, "ellipse",
            )

            # Kiểm tra phương trình ellipse.
            value = (
                ((ref[0] - ELLIPSE_A) / ELLIPSE_A) ** 2
                + (ref[1] / ELLIPSE_B) ** 2
            )
            self.assertAlmostEqual(value, 1.)

            # Kiểm tra vận tốc bằng đạo hàm số của vị trí.
            eps = 1e-5

            if 0 < t < 10:
                ref_plus = reference_state(
                    t + eps, start, goal, 10, "ellipse",
                )
                ref_minus = reference_state(
                    t - eps, start, goal, 10, "ellipse",
                )

                derivative = (
                    ref_plus[:3] - ref_minus[:3]
                ) / (2 * eps)

                np.testing.assert_allclose(
                    ref[3:6],
                    derivative,
                    atol=1e-7,
                )

        # Quỹ đạo khép kín, vận tốc hai đầu bằng 0.
        np.testing.assert_allclose(
            reference_state(
                0, start, goal, 10, "ellipse",
            ),
            start,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            reference_state(
                10, start, goal, 10, "ellipse",
            ),
            start,
            atol=1e-12,
        )

    def test_prediction_before_update_and_persistent_correction(self):
        base = CUSUMAdaptiveQuantile(
            threads=1,
            q_init=0,
            n_min=2,
            n_max=5,
            h=0.1,
        )

        upgraded = CoverageCalibratedCUSUM(
            base,
            threads=1,
            score_max=10,
            eta_cal=0.5,
        )

        previous = 0.
        alarms = 0

        for score in [0., 0., 3., 3., 0.1, 0.1, 4.]:
            # Lấy dự đoán trước khi cập nhật.
            q_before = upgraded.get_quantile()
            upgraded.update(score)

            expected = max(
                0.,
                previous
                + 0.5 * (
                    int(score > q_before) - 0.1
                ),
            )

            # Correction phải theo đúng quy tắc cập nhật,
            # kể cả khi CUSUM báo động.
            self.assertAlmostEqual(
                upgraded.c[0],
                expected,
            )
            self.assertEqual(
                upgraded.events[-1]["q_before"],
                q_before,
            )

            previous = expected
            alarms += bool(base.events[-1]["alarm"])

        # Chuỗi kiểm tra phải thực sự kích hoạt alarm.
        self.assertGreater(alarms, 0)

    def test_thread_schedule_and_score_bound(self):
        base = CUSUMAdaptiveQuantile(threads=3)

        upgraded = CoverageCalibratedCUSUM(
            base,
            threads=3,
            score_max=2,
        )

        for score in [1., 0.2, 2., 0.3, 1., 0.1]:
            upgraded.update(score)

        self.assertEqual(
            [
                event["thread"]
                for event in upgraded.events
            ],
            [0, 1, 2, 0, 1, 2],
        )

        # Không được âm, vượt score_max, NaN hoặc vô hạn.
        for bad_score in [
            2.1,
            -1.,
            float("nan"),
            float("inf"),
        ]:
            with self.assertRaises(ValueError):
                upgraded.update(bad_score)

        # Các lần cập nhật bị từ chối không làm lệch thread.
        self.assertEqual(upgraded.k, 6)
        self.assertEqual(upgraded.base.k, 6)


if __name__ == "__main__":
    unittest.main()