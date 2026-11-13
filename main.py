import argparse
import collections
import importlib
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plant import Plant
from controller import DynamicTubeMPC
from ocp import StaggeredDriftScoreOCP
from ssml import (
    get_or_train_model,
    flatten_params,
    assign_params,
    compute_jacobian,
    spectral_normalization_clip,
)


# Kích thước quỹ đạo, đơn vị mét.
SINE_AMPLITUDE = 0.5
CIRCLE_RADIUS = 1.0
INFINITY_X = 2.0
INFINITY_Y = 0.75

PATHS = ("straight", "circle", "infinity", "sine")


def reference_state(t, start, goal, duration, trajectory):
    """Tham chiếu [x, y, z, vx, vy, vz, roll, pitch]."""
    u = np.clip(float(t) / duration, 0.0, 1.0)

    # Vận tốc bằng 0 tại đầu và cuối.
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / duration

    theta = 2*np.pi*s
    omega = 2*np.pi*ds

    ref = np.zeros(8)
    ref[:3] = start[:3]

    if trajectory in ("straight", "sine"):
        ref[:3] += s * (goal[:3] - start[:3])
        ref[3:6] = ds * (goal[:3] - start[:3])

        if trajectory == "sine":
            ref[1] += SINE_AMPLITUDE * np.sin(theta)
            ref[4] += (
                SINE_AMPLITUDE * np.cos(theta) * omega
            )

    elif trajectory == "circle":
        # Bắt đầu tại start, tâm lệch sang phải một bán kính.
        ref[0] += CIRCLE_RADIUS * (1 - np.cos(theta))
        ref[1] += CIRCLE_RADIUS * np.sin(theta)

        ref[3] = CIRCLE_RADIUS * np.sin(theta) * omega
        ref[4] = CIRCLE_RADIUS * np.cos(theta) * omega

    elif trajectory == "infinity":
        # Số 8 nằm ngang; bắt đầu tại giao điểm giữa.
        ref[0] += INFINITY_X * np.sin(theta)
        ref[1] += INFINITY_Y * np.sin(2*theta)

        ref[3] = INFINITY_X * np.cos(theta) * omega
        ref[4] = (
            2 * INFINITY_Y * np.cos(2*theta) * omega
        )

    else:
        raise ValueError(f"Unknown trajectory: {trajectory}")

    return ref


def compute_dist_bound(q_k, L_d, T_p):
    # Giữ công thức trong main.py bạn gửi.
    return np.where(
        q_k < 0.5 * L_d * T_p**2,
        q_k / max(T_p, 1e-8) + 0.5 * L_d * T_p,
        np.sqrt(2.0 * L_d * q_k),
    )


def save_outputs(
    out,
    trajectory,
    times,
    actual,
    target,
    curve,
    bounds,
    residuals,
    weights,
    dt_sim,
    make_gif,
):
    errors = np.linalg.norm(
        actual[:, :3] - target[:, :3],
        axis=1,
    )

    print(
        f"{trajectory}: "
        f"RMSE={np.sqrt(np.mean(errors**2)):.3f} m; "
        f"max={errors.max():.3f} m"
    )

    np.savez(
        out / "tracking_data.npz",
        time=times,
        actual=actual,
        reference=target,
        position_error=errors,
    )

    # PNG: đường tham chiếu và đường thực tế.
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        curve[:, 0],
        curve[:, 1],
        "r--",
        lw=2,
        label="Reference trajectory",
    )
    ax.plot(
        actual[:, 0],
        actual[:, 1],
        "b-",
        lw=2,
        label="Actual trajectory",
    )

    ax.set(
        xlabel="X Position (m)",
        ylabel="Y Position (m)",
        title=f"Trajectory tracking: {trajectory}",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out / "top_down_tube.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Sai số bám vị trí 3D.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, errors)
    ax.set(
        xlabel="Time (s)",
        ylabel="3D position error (m)",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / "tracking_error.png", dpi=150)
    plt.close(fig)

    # Disturbance bound.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        times,
        0,
        bounds,
        alpha=0.4,
        label="SI-OCP Prediction Bound",
    )
    ax.plot(
        times,
        residuals,
        label="True disturbance residual",
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Disturbance norm",
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "si-ocp_vs_true.png", dpi=150)
    plt.close(fig)

    # Tham số neural network.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, weights, alpha=0.1, lw=1)
    ax.set(
        xlabel="Time (s)",
        ylabel="NN weight values",
    )
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / "nn_params_vs_time.png", dpi=150)
    plt.close(fig)

    if make_gif:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            curve[:, 0],
            curve[:, 1],
            "r--",
            lw=2,
            label="Reference trajectory",
        )
        line, = ax.plot(
            [],
            [],
            "b-",
            lw=2,
            label="Actual trajectory",
        )
        point, = ax.plot(
            [],
            [],
            "bo",
            markersize=5,
        )

        xy = np.vstack((curve[:, :2], actual[:, :2]))

        ax.set_xlim(
            xy[:, 0].min() - 0.5,
            xy[:, 0].max() + 0.5,
        )
        ax.set_ylim(
            xy[:, 1].min() - 0.5,
            xy[:, 1].max() + 0.5,
        )
        ax.set(
            xlabel="X Position (m)",
            ylabel="Y Position (m)",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        def update(k):
            line.set_data(
                actual[:k + 1, 0],
                actual[:k + 1, 1],
            )
            point.set_data(
                actual[k:k + 1, 0],
                actual[k:k + 1, 1],
            )
            ax.set_title(
                f"{trajectory} | t={times[k]:.2f}s"
            )
            return line, point

        # Khoảng 10 khung hình/giây.
        # Không thay đổi bước tính toán mô phỏng.
        skip = max(1, int(round(0.1 / dt_sim)))
        frames = list(range(0, len(times), skip))

        if frames[-1] != len(times) - 1:
            frames.append(len(times) - 1)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=1000 * dt_sim * skip,
            blit=False,
        )
        ani.save(
            out / "top_down_animation.gif",
            writer="pillow",
            fps=1.0 / (dt_sim * skip),
        )
        plt.close(fig)

    print(f"Saved: {out}")


def run_case(cfg, trajectory, duration, out, make_gif):
    out.mkdir(parents=True, exist_ok=True)

    # Mỗi test bắt đầu lại với cùng seed và weights.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dt_sim = cfg.dt_sim
    dt_mpc = cfg.dt_mpc
    n_substeps = int(round(dt_mpc / dt_sim))

    if (
        duration <= 0
        or n_substeps < 1
        or not np.isclose(n_substeps * dt_sim, dt_mpc)
    ):
        raise ValueError(
            "duration > 0; dt_mpc phải là bội nguyên của dt_sim"
        )

    steps = int(round(duration / dt_sim))

    if (
        steps < 1
        or not np.isclose(steps * dt_sim, duration)
    ):
        raise ValueError(
            "duration phải là bội nguyên của dt_sim"
        )

    plant = Plant(spatial_mode=cfg.spatial_wind)

    # Không có vật cản.
    controller = DynamicTubeMPC(
        plant,
        obstacles=[],
        H=cfg.mpc_horizon,
        dt=dt_mpc,
    )

    window = cfg.mpc_horizon * n_substeps
    T_p = window * dt_sim

    ocp = StaggeredDriftScoreOCP(
        alpha=cfg.alpha_ocp,
        eta_const=cfg.eta_ocp,
        N_threads=window,
        q_init=cfg.q_init_ocp,
    )

    model = get_or_train_model()
    theta0 = flatten_params(model).clone().detach()
    theta = theta0.clone().detach()

    x = np.copy(cfg.x0)
    u = np.array([0.0, 0.0, 9.81 * plant.m])

    # Giữ cách tính residual của main.py bạn gửi.
    u_old = u.copy()

    past_x = collections.deque(maxlen=window)
    past_f = collections.deque(maxlen=window)

    states = [x.copy()]
    times = [0.0]
    bounds = [cfg.dist_bound_init]
    residuals = [0.0]
    weights = [theta.numpy().copy()]

    covered = 0
    completed = True

    def ref(t):
        return reference_state(
            t,
            cfg.x0,
            cfg.x_goal,
            duration,
            trajectory,
        )

    print(
        f"\nScenario={cfg.name}; "
        f"trajectory={trajectory}; "
        f"duration={duration}s"
    )

    for k in range(steps):
        t = k * dt_sim

        if k % n_substeps == 0:
            horizon = np.array([
                ref(t + (j + 1) * dt_mpc)
                for j in range(controller.H)
            ])

            bound = float(
                compute_dist_bound(
                    ocp.get_quantile(),
                    cfg.ddot_bound,
                    T_p,
                )
            )

            tic = time.perf_counter()

            u, _, _, success = controller.compute_u(
                x,
                horizon,
                bound,
                model_nn=model,
            )

            if not success:
                print(
                    f"Solver failed at t={t:.2f}s; "
                    "test incomplete."
                )
                completed = False
                break

            if k % (20 * n_substeps) == 0:
                print(
                    f"t={t:.2f}s; "
                    f"MPC={time.perf_counter()-tic:.3f}s"
                )

        old = x.copy()
        x = plant.step(old, u, t, dt_sim)

        if not np.isfinite(x).all():
            completed = False
            print(
                f"Nonfinite state at t={t+dt_sim:.2f}s; "
                "test incomplete."
            )
            break

        x_in = np.concatenate((old[3:6], old[6:8]))

        with torch.no_grad():
            nn_acc = model(
                torch.tensor(x_in, dtype=torch.float32)
            ).numpy()

        pred = (
            plant.f(old)
            + plant.g_mat(old) @ u_old
            + np.concatenate((
                np.zeros(3),
                nn_acc,
                np.zeros(2),
            ))
        )

        true_error = (
            plant.dynamics(t, old, u_old) - pred
        )[3:6]

        error_acc = (
            (x[3:6] - old[3:6]) / dt_sim
            - pred[3:6]
        )

        past_x.append(old)
        past_f.append(pred.copy())

        if len(past_x) >= window:
            xb = np.array(past_x)
            fb = np.array(past_f)

            integral = np.cumsum(
                np.flip(fb, axis=0),
                axis=0,
            ) * dt_sim

            score = np.linalg.norm(
                xb[-1] - np.flip(xb, axis=0) - integral,
                axis=1,
            ).max()

            q = ocp.update(score)

            bound = float(
                compute_dist_bound(
                    q,
                    cfg.ddot_bound,
                    T_p,
                )
            )
        else:
            bound = cfg.dist_bound_init

        J = compute_jacobian(
            model, x_in
        ).detach().numpy()

        covered += int(
            np.linalg.norm(true_error) <= bound
        )

        theta_dot = (
            cfg.gamma_lr * (J.T @ error_acc)
            - cfg.lambd * (
                theta.numpy() - theta0.numpy()
            )
        )

        theta = theta + torch.tensor(
            theta_dot,
            dtype=torch.float32,
        ) * dt_sim

        assign_params(model, theta)
        spectral_normalization_clip(model)

        states.append(x.copy())
        times.append((k + 1) * dt_sim)
        bounds.append(bound)
        residuals.append(np.linalg.norm(true_error))
        weights.append(theta.numpy().copy())

    states = np.array(states)
    times = np.array(times)

    target = np.array([
        ref(t) for t in times
    ])
    curve = np.array([
        ref(t)
        for t in np.linspace(0, duration, 500)
    ])

    count = len(times) - 1

    print(
        f"Completed={completed}; "
        f"steps={count}/{steps}"
    )

    if count:
        print(
            f"Coverage={covered/count:.2%}; "
            f"target={1-cfg.alpha_ocp:.1%}"
        )

    save_outputs(
        out,
        trajectory,
        times,
        states,
        target,
        curve,
        bounds,
        residuals,
        weights,
        dt_sim,
        make_gif,
    )

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Drone trajectory tracking tests"
    )

    parser.add_argument(
        "--scenario",
        default="adaptation_off",
    )
    parser.add_argument(
        "--trajectory",
        choices=(*PATHS, "all"),
        default="sine",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = importlib.import_module(
        f"scenarios.{args.scenario}"
    ).scenario

    duration = (
        cfg.t_end
        if args.duration is None
        else args.duration
    )

    if not np.isfinite(duration) or duration <= 0:
        parser.error(
            "duration phải là số hữu hạn > 0"
        )

    paths = (
        PATHS
        if args.trajectory == "all"
        else (args.trajectory,)
    )

    all_completed = True

    for path in paths:
        out = (
            Path("output")
            / args.scenario
            / path
        )

        ok = run_case(
            cfg,
            path,
            duration,
            out,
            not args.no_gif,
        )

        all_completed = all_completed and ok

    if not all_completed:
        raise SystemExit(
            "Có test chưa chạy hết thời gian; xem log ở trên."
        )


if __name__ == "__main__":
    main()