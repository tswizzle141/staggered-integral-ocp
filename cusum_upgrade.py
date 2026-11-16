"""CUSUM upgrade extracted unchanged from the supplied benchmark_ocp3.py.

Use CoverageCalibratedCUSUM wrapping CUSUMAdaptiveQuantile.
The raw estimator is an internal proposal, not a comparison baseline.
"""
import math
from collections import deque

class _CUSUMQuantileThread:
    """Trang thai rieng cua mot thread."""

    def __init__(
        self, alpha, q_init, n_min, n_max,
        h, delta, beta, grow_every, two_sided
    ):
        self.alpha = alpha
        self.q = float(q_init)

        self.n_min = n_min
        self.n_max = n_max
        self.window = n_max

        self.h = h
        self.delta = delta
        self.beta = beta
        self.grow_every = grow_every
        self.two_sided = two_sided

        self.history = deque(maxlen=n_max)

        self.mu = None
        self.g_up = 0.0
        self.g_down = 0.0

        self.age = 0
        self.updates = 0
        self.alarm = ""

    def update(self, score):
        self.alarm = ""
        old_mu = self.mu

        if old_mu is None:
            self.mu = score

        elif self.updates >= self.n_min:
            # Dung reference truoc khi nhan score hien tai.
            self.g_up = max(
                0.0,
                self.g_up + score - old_mu - self.delta
            )
            self.g_down = max(
                0.0,
                self.g_down + old_mu - score - self.delta
            )

            if self.g_up > self.h:
                self.alarm = "up"

            elif self.two_sided and self.g_down > self.h:
                self.alarm = "down"

        self.history.append(score)
        self.updates += 1

        if self.alarm:
            # Bo phan lich su cu, chi giu n_min mau gan nhat.
            recent = list(self.history)[-self.n_min:]
            self.history.clear()
            self.history.extend(recent)

            self.window = self.n_min
            self.age = 0

            self.g_up = 0.0
            self.g_down = 0.0

            # Khoi tao reference moi tu score gay bao dong.
            self.mu = score

        else:
            if self.updates <= self.n_min:
                # Hoc muc binh thuong ban dau.
                self.mu = sum(self.history) / len(self.history)
            else:
                # Reference EWMA cap nhat cham.
                self.mu = (
                    (1 - self.beta) * self.mu
                    + self.beta * score
                )

            self.age += 1

            if self.age >= self.grow_every:
                self.window = min(
                    self.n_max, self.window + 1
                )
                self.age = 0

        # Quantile thuc nghiem tren cua so hien tai.
        values = sorted(
            list(self.history)[-self.window:]
        )

        rank = max(
            1,
            math.ceil((1 - self.alpha) * len(values))
        )

        self.q = float(values[rank - 1])
        return self.q

class CUSUMAdaptiveQuantile:
    """
    Giao dien tuong thich voi evaluate():
        get_quantile()
        update(score)

    CUSUM va cua so quantile doc lap cho tung thread.
    """

    def __init__(
        self,
        alpha=0.1,
        q_init=0.5,
        threads=10,
        n_min=8,
        n_max=100,
        h=1.0,
        delta=0.05,
        beta=0.02,
        grow_every=5,
        two_sided=True,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")

        counts = [threads, n_min, n_max, grow_every]

        if any(
            not isinstance(v, int) or v < 1
            for v in counts
        ):
            raise ValueError(
                "thread/window counts must be positive integers"
            )

        if n_min > n_max:
            raise ValueError("n_min must be <= n_max")

        if not all(
            math.isfinite(v)
            for v in [q_init, h, delta, beta]
        ):
            raise ValueError("parameters must be finite")

        if (
            q_init < 0 or h <= 0 or delta < 0
            or not 0 < beta <= 1
        ):
            raise ValueError("invalid CUSUM parameters")

        self.workers = [
            _CUSUMQuantileThread(
                alpha, q_init, n_min, n_max,
                h, delta, beta, grow_every, two_sided
            )
            for _ in range(threads)
        ]

        self.k = 0
        self.events = []

    def get_quantile(self):
        j = self.k % len(self.workers)
        return self.workers[j].q

    def update(self, score):
        score = float(score)

        if not math.isfinite(score) or score < 0:
            raise ValueError(
                "score must be finite and nonnegative"
            )

        j = self.k % len(self.workers)
        worker = self.workers[j]

        q = worker.update(score)

        self.events.append({
            "t": self.k,
            "thread": j,
            "alarm": worker.alarm,
            "window": worker.window,
            "samples": len(worker.history),
            "q_after": q,
            "reference_mean": worker.mu,
        })

        self.k += 1
        return q

class CoverageCalibratedCUSUM:
    """
    CUSUM proposes a quantile; a correction controls miscoverage.

    Assumptions:
      - Every score is in [0, score_max].
      - Base algorithm uses the same round-robin thread schedule.
      - Evaluate prediction BEFORE update(score).
      - Never reset correction states on CUSUM alarms.

    Guarantee:
      Long-run empirical score coverage is at least 1-alpha.
      This is NOT a strongly adaptive regret guarantee.
    """

    def __init__(
        self,
        base,
        alpha=0.1,
        threads=10,
        score_max=10.0,
        eta_cal=0.5,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")

        if not isinstance(threads, int) or threads < 1:
            raise ValueError(
                "threads must be a positive integer"
            )

        if any(
            not math.isfinite(v) or v <= 0
            for v in [score_max, eta_cal]
        ):
            raise ValueError(
                "score_max and eta_cal must be finite and positive"
            )

        if len(base.workers) != threads:
            raise ValueError(
                "base and calibrator must have the same threads"
            )

        self.base = base
        self.alpha = alpha
        self.B = float(score_max)
        self.eta_cal = float(eta_cal)

        # Persistent correction, one state per thread.
        self.c = [0.0] * threads
        self.k = 0
        self.events = []

    def get_quantile(self):
        j = self.k % len(self.c)

        b = float(self.base.get_quantile())

        if not math.isfinite(b):
            raise ValueError("base quantile must be finite")

        # Bound the proposal, then add coverage correction.
        b = min(self.B, max(0.0, b))

        return float(
            min(self.B, b + self.c[j])
        )

    def update(self, score):
        score = float(score)

        if (
            not math.isfinite(score)
            or not 0 <= score <= self.B
        ):
            raise ValueError(
                "score outside [0, score_max]; "
                "bound assumption violated"
            )

        j = self.k % len(self.c)

        # Prediction used to evaluate this score.
        q_before = self.get_quantile()
        miss = int(score > q_before)

        # Project correction onto nonnegative values.
        c_after = max(
            0.0,
            self.c[j]
            + self.eta_cal * (miss - self.alpha)
        )

        # CUSUM updates independently; its alarms do not reset c.
        b_after = float(self.base.update(score))
        self.c[j] = c_after

        q_after = min(
            self.B,
            max(0.0, min(self.B, b_after)) + c_after,
        )

        self.events.append({
            "t": self.k,
            "thread": j,
            "score": score,
            "q_before": q_before,
            "violation": miss,
            "correction_after": c_after,
            "q_after": q_after,
        })

        self.k += 1
        return float(q_after)

    def coverage_lower_bound(self, length):
        """Worst-case bound for any interval of this length."""
        if length < 1:
            raise ValueError("length must be positive")

        active_threads = min(len(self.c), length)

        penalty = (
            active_threads
            * (self.B + self.eta_cal)
            / (self.eta_cal * length)
        )

        return max(
            0.0,
            1 - self.alpha - penalty,
        )
