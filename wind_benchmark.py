"""Wind stress tests for the supplied main.py."""

import argparse
import csv
import importlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import main as app
from plant import Plant as OriginalPlant
from ssml import DEFAULT_EPOCHS, get_or_train_model


CASES = (
    "weak_step",
    "medium_step",
    "strong_step",
    "direction_flip",
    "gust_switch",
)


def schedule(case):
    """
    Mỗi phần tử:
        (tỷ lệ thời gian bắt đầu, vector vận tốc gió m/s)

    Vector gió nằm trong hệ tọa độ world.
    """
    low = (0.5, 0.0, 0.0)

    if case in CASES[:3]:
        speed = {
            "weak_step": 1.5,
            "medium_step": 3.0,
            "strong_step": 5.0,
        }[case]

        return [
            (0.00, low),
            (0.25, (speed, 0.0, 0.0)),
            (0.65, low),
        ]

    if case == "direction_flip":
        return [
            (0.00, low),
            (0.25, (5.0, 0.0, 0.0)),
            (0.50, (-5.0, 0.0, 0.0)),
            (0.75, (0.0, 5.0, 0.0)),
        ]

    if case == "gust_switch":
        return [
            (0.00, low),
            (0.25, (7.0, 0.0, 0.0)),
            (0.30, low),
            (0.40, (-7.0, 0.0, 0.0)),
            (0.45, low),
            (0.55, (0.0, 7.0, 0.0)),
            (0.60, low),
            (0.70, (0.0, -7.0, 0.0)),
            (0.75, low),
            (0.85, (5.0, 5.0, 0.0)),
            (0.90, low),
        ]

    raise ValueError(f"Unknown case: {case}")


class WindProfile:
    def __init__(self, case, duration, dt, seed):
        self.case = case
        self.duration = duration
        self.dt = dt
        self.steps = int(round(duration / dt))

        if (
            self.steps < 1
            or not np.isclose(self.steps * dt, duration)
        ):
            raise ValueError(
                "duration must be a positive multiple of dt_sim"
            )

        entries = schedule(case)
        indices = [
            int(round(fraction * self.steps))
            for fraction, _ in entries
        ]

        if (
            len(set(indices)) != len(indices)
            or indices[-1] >= self.steps
        ):
            raise ValueError(
                "duration is too short to resolve wind regimes"
            )

        self.starts = np.array(indices)
        self.vectors = np.array(
            [vector for _, vector in entries],
            dtype=float,
        )

        # Tạo trước nhiễu lực theo thời gian.
        # Không phụ thuộc số lần RK45 gọi dynamics.
        rng = np.random.default_rng(seed)
        self.noise = (
            rng.normal(size=(self.steps + 1, 3))
            * np.array([0.2, 0.2, 0.1])
        )

    def index(self, t):
        return int(np.clip(
            np.floor(t / self.dt + 1e-7),
            0,
            self.steps,
        ))

    def velocity(self, t):
        k = self.index(t)
        j = np.searchsorted(
            self.starts, k, side="right"
        ) - 1
        return self.vectors[j].copy()

    def noise_at(self, t):
        return self.noise[self.index(t)].copy()

    def regimes(self):
        ends = list(self.starts[1:]) + [self.steps]

        return [
            (
                int(start) * self.dt,
                int(end) * self.dt,
                vector.tolist(),
            )
            for start, end, vector in zip(
                self.starts, ends, self.vectors
            )
        ]


def plant_class(profile):
    class ControlledWindPlant(OriginalPlant):
        def wind_velocity(self, t, p):
            # Gió tuyệt đối, không giảm gần đích.
            return profile.velocity(t)

        def unmodeled_dynamics(self, t, p, v, angles):
            # Giữ luật quadratic drag của plant.py.
            phi, theta = angles
            cx, sx = np.cos(phi), np.sin(phi)
            cy, sy = np.cos(theta), np.sin(theta)

            R = np.array([
                [cy, 0, sy],
                [sx * sy, cx, -sx * cy],
                [-cx * sy, sx, cx * cy],
            ])

            v_relative = v - self.wind_velocity(t, p)
            v_body = R.T @ v_relative
            drag_matrix = np.diag([0.3, 0.3, 0.6])

            drag = (
                -self.m
                * R
                @ drag_matrix
                @ (v_body * np.abs(v_body))
            )

            return drag + profile.noise_at(t)

        def step(self, x, u, t, dt):
            # Giữ gió và nhiễu cố định trong một dt_sim.
            # Cú nhảy được áp dụng đúng tại biên bước tiếp theo.
            sol = solve_ivp(
                lambda _, state: self.dynamics(t, state, u),
                [t, t + dt],
                x,
                method="RK45",
            )

            if not sol.success:
                raise RuntimeError(sol.message)

            return sol.y[:, -1]

    return ControlledWindPlant


@contextmanager
def use_wind(profile):
    previous = app.Plant
    app.Plant = plant_class(profile)

    try:
        yield
    finally:
        app.Plant = previous


def metrics(scores, q, alpha):
    if len(scores) == 0:
        return {
            "samples": 0,
            "coverage": None,
            "mean_q": None,
            "pinball": None,
            "longest_miss_run": None,
            "min_rolling_50": None,
        }

    misses = scores > q
    error = scores - q

    longest = 0
    run = 0

    for miss in misses:
        run = run + 1 if miss else 0
        longest = max(longest, run)

    if len(scores) >= 50:
        rolling = np.convolve(
            (~misses).astype(float),
            np.ones(50) / 50,
            mode="valid",
        )
    else:
        rolling = np.array([])

    loss = (
        (1 - alpha) * np.maximum(error, 0)
        + alpha * np.maximum(-error, 0)
    )

    return {
        "samples": len(scores),
        "coverage": float(1 - misses.mean()),
        "mean_q": float(q.mean()),
        "pinball": float(loss.mean()),
        "longest_miss_run": longest,
        "min_rolling_50": (
            float(rolling.min())
            if rolling.size else None
        ),
    }


def analyze(
    directory,
    profile,
    summary,
    alpha,
    post_seconds,
):
    with np.load(
        directory / "tracking_data.npz"
    ) as data:
        times = data["score_time"]
        scores = data["score"]
        q = data["q_before"]

        intervals = [
            ("all", 0.0, profile.duration),
        ]

        for j, (start, end, _) in enumerate(
            profile.regimes()
        ):
            intervals.append((
                f"regime_{j}", start, end
            ))

            if j:
                intervals.append((
                    f"post_change_{j}",
                    start,
                    min(end, start + post_seconds),
                ))

        # main.py ghi timestamp ở CUỐI bước.
        # Phân loại theo thời điểm BẮT ĐẦU bước quan sát.
        observed_at = times - profile.dt
        rows = []

        for label, start, end in intervals:
            mask = (
                (observed_at >= start - 1e-8)
                & (observed_at < end - 1e-8)
            )

            rows.append({
                "case": profile.case,
                "algorithm": summary["algorithm"],
                "region": label,
                "start": start,
                "end": end,
                **metrics(scores[mask], q[mask], alpha),
            })

        overall = metrics(scores, q, alpha)

        summary.update({
            "case": profile.case,
            "score_pinball": overall["pinball"],
            "longest_score_miss_run": (
                overall["longest_miss_run"]
            ),
            "max_score": (
                float(scores.max())
                if scores.size else None
            ),
        })

    if summary["algorithm"] == "cusum_upgrade":
        with (
            directory / "cusum_diagnostics.csv"
        ).open(newline="", encoding="utf-8") as f:
            summary["cusum_alarm_count"] = sum(
                bool(row["alarm"])
                for row in csv.DictReader(f)
            )
    else:
        summary["cusum_alarm_count"] = None

    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    app.write_csv(
        directory / "phase_metrics.csv",
        rows,
    )

    return rows


def plot_diagnostics(
    destination,
    summaries,
    profile,
    alpha,
):
    fig, axes = plt.subplots(
        4, 1, figsize=(13, 12), sharex=True
    )

    grid = (
        np.arange(profile.steps + 1)
        * profile.dt
    )
    wind = np.array([
        profile.velocity(t) for t in grid
    ])

    for j, axis in enumerate("xyz"):
        axes[0].step(
            grid,
            wind[:, j],
            where="post",
            label=f"wind {axis}",
        )

    for summary in summaries:
        name = summary["algorithm"]

        with np.load(
            destination / name / "tracking_data.npz"
        ) as data:
            t = data["score_time"]
            scores = data["score"]
            q = data["q_before"]

            color = (
                "tab:blue"
                if name == "ocp"
                else "tab:orange"
            )
            label = app.ALGORITHM_LABELS[name]

            axes[1].plot(
                t,
                scores,
                color=color,
                alpha=0.3,
                linewidth=0.8,
                label=f"{name} score",
            )
            axes[1].plot(
                t,
                q,
                color=color,
                label=f"{name} q before update",
            )

            flags = (scores <= q).astype(float)

            if len(flags) >= 50:
                rolling = np.convolve(
                    flags,
                    np.ones(50) / 50,
                    mode="valid",
                )
                axes[2].plot(
                    t[49:],
                    rolling,
                    color=color,
                    label=label,
                )

            axes[3].plot(
                data["time"],
                data["position_error"],
                color=color,
                linestyle=(
                    "-" if name == "ocp" else "--"
                ),
                label=label,
            )

    axes[0].set_ylabel("Wind (m/s)")
    axes[1].set_ylabel("Score / quantile")
    axes[2].set_ylabel("Coverage (50 updates)")
    axes[2].set_ylim(-0.05, 1.05)

    axes[2].axhline(
        1 - alpha,
        color="black",
        linestyle="--",
        label="Target",
    )

    axes[3].set(
        ylabel="Position error (m)",
        xlabel="Time (s)",
    )

    for ax in axes:
        for start, _, _ in profile.regimes()[1:]:
            ax.axvline(
                start,
                color="grey",
                linestyle=":",
                alpha=0.5,
            )

        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        profile.case
        + ": prescribed wind, independent closed-loop runs"
    )
    fig.tight_layout()

    fig.savefig(
        destination / "wind_comparison.png",
        dpi=150,
    )
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--case",
        choices=(*CASES, "all"),
        default="all",
    )
    p.add_argument(
        "--scenario",
        choices=("adaptation_off", "adaptation_on"),
        default="adaptation_on",
    )
    p.add_argument(
        "--trajectory",
        choices=app.PATHS,
        default="circle",
    )
    p.add_argument(
        "--duration", type=float, default=60.0,
    )
    p.add_argument(
        "--algorithm",
        choices=("ocp", "cusum_upgrade", "both"),
        default="both",
    )
    p.add_argument(
        "--seed", type=int, default=1337,
    )
    p.add_argument(
        "--post-seconds", type=float, default=5.0,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    p.add_argument(
        "--retrain", action="store_true",
    )
    p.add_argument(
        "--gif", action="store_true",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/wind_tests"),
    )
    p.add_argument(
        "--cusum-score-max",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--cusum-eta-cal",
        type=float,
        default=None,
    )

    args = p.parse_args()

    if (
        not np.isfinite(args.duration)
        or args.duration <= 0
        or args.epochs < 1
    ):
        p.error(
            "duration must be finite and positive; "
            "epochs must be positive"
        )

    if (
        not np.isfinite(args.post_seconds)
        or args.post_seconds <= 0
    ):
        p.error(
            "post-seconds must be finite and positive"
        )

    source_cfg = importlib.import_module(
        f"scenarios.{args.scenario}"
    ).scenario

    cfg = replace(
        source_cfg,
        seed=args.seed,
    )

    cases = (
        CASES
        if args.case == "all"
        else (args.case,)
    )
    algorithms = (
        ("ocp", "cusum_upgrade")
        if args.algorithm == "both"
        else (args.algorithm,)
    )

    profiles = [
        WindProfile(
            case,
            args.duration,
            cfg.dt_sim,
            args.seed,
        )
        for case in cases
    ]

    opts = {
        "score_max": args.cusum_score_max,
        "eta_cal": (
            cfg.eta_ocp
            if args.cusum_eta_cal is None
            else args.cusum_eta_cal
        ),
        "n_min": 8,
        "n_max": 100,
        "h": 1.0,
        "delta": 0.05,
        "beta": 0.02,
        "grow_every": 5,
        "two_sided": True,
    }

    if "cusum_upgrade" in algorithms:
        app.CoverageCalibratedCUSUM(
            app.CUSUMAdaptiveQuantile(threads=1),
            threads=1,
            score_max=opts["score_max"],
            eta_cal=opts["eta_cal"],
        )

    args.weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoints = {}

    # Chuẩn bị offline SSML trước khi thay Plant.
    # Không train SSML trên gió của bài test.
    for algorithm in algorithms:
        checkpoints[algorithm] = (
            args.weights_dir
            / f"ssml_weights_{algorithm}.pt"
        )

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        get_or_train_model(
            weights_path=checkpoints[algorithm],
            epochs=args.epochs,
            force_retrain=args.retrain,
        )

    all_summaries = []
    all_phases = []
    failures = []

    root = (
        args.output_dir
        / args.scenario
        / args.trajectory
        / f"seed_{args.seed}"
    )
    root.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        dest = root / profile.case
        dest.mkdir(parents=True, exist_ok=True)

        metadata = {
            "units": "m/s",
            "duration": args.duration,
            "dt_sim": cfg.dt_sim,
            "seed": args.seed,
            "regimes": profile.regimes(),
            "note": (
                "Absolute wind, no goal attenuation; "
                "noise ZOH shared by both algorithms"
            ),
        }

        (dest / "wind_schedule.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        wind_rows = []

        for k in range(profile.steps):
            t = k * profile.dt
            v = profile.velocity(t)
            noise = profile.noise_at(t)

            wind_rows.append({
                "time": t,
                "wx": v[0],
                "wy": v[1],
                "wz": v[2],
                "noise_x": noise[0],
                "noise_y": noise[1],
                "noise_z": noise[2],
            })

        app.write_csv(
            dest / "wind_input.csv",
            wind_rows,
        )

        summaries = []

        for algorithm in algorithms:
            directory = dest / algorithm
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:
                with use_wind(profile):
                    summary = app.run_case(
                        cfg,
                        args.trajectory,
                        args.duration,
                        directory,
                        args.gif,
                        algorithm,
                        opts,
                        checkpoints[algorithm],
                    )

            except ValueError as exc:
                if (
                    "score outside [0, score_max]"
                    not in str(exc)
                ):
                    raise

                failure = {
                    "case": profile.case,
                    "algorithm": algorithm,
                    "status": "score_bound_violated",
                    "error": str(exc),
                }

                failures.append(failure)

                (directory / "failure.json").write_text(
                    json.dumps(failure, indent=2),
                    encoding="utf-8",
                )

                print(
                    f"FAILED {profile.case}/{algorithm}: "
                    f"{exc}"
                )
                continue

            all_phases.extend(
                analyze(
                    directory,
                    profile,
                    summary,
                    cfg.alpha_ocp,
                    args.post_seconds,
                )
            )

            summaries.append(summary)
            all_summaries.append(summary)

            if not summary["completed"]:
                failures.append({
                    "case": profile.case,
                    "algorithm": algorithm,
                    "status": "incomplete",
                })

        if summaries:
            app.save_comparison(
                dest,
                summaries,
                cfg.alpha_ocp,
            )
            plot_diagnostics(
                dest,
                summaries,
                profile,
                cfg.alpha_ocp,
            )

    app.write_csv(
        root / "all_summary.csv",
        all_summaries,
    )
    app.write_csv(
        root / "all_phase_metrics.csv",
        all_phases,
    )

    (root / "failures.json").write_text(
        json.dumps(failures, indent=2),
        encoding="utf-8",
    )

    print(f"Results: {root}")

    if failures:
        raise SystemExit(
            "Some stress tests failed; see failures.json. "
            "Partial runs are not passes."
        )


if __name__ == "__main__":
    main()