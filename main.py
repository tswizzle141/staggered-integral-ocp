"""Compare SI-OCP and CUSUM control pipelines on shared quadrotor scenarios.

Both pipelines and the shared plant are defined in this file. SI-OCP retains
its baseline equations; CUSUM uses a wind observer and fixed hover tube MPC.
No companion source folder, source extraction, or network download is needed.
The default strong/mixed CUSUM profile uses empirically tuned simulation limits;
select --cusum-profile original to reproduce the original limits.
Solver defaults remain SI-OCP 300 and CUSUM 200 iterations, with no added CPU cap.

Run: python main.py
Default: 20 seconds of flight only (400 steps); no post-flight hover phase.
Dependencies: numpy, scipy, casadi, torch, matplotlib, pillow.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import collections
import csv
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass, asdict
from collections.abc import Callable
from types import SimpleNamespace
from pathlib import Path
import contextlib
import traceback

os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "siocp_cusum_cache")
)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ["XDG_CACHE_HOME"]) / "matplotlib")
)

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.signal import cont2discrete

torch.set_num_threads(1)

PINNED_COMMIT = "25fd56283b37849a08c859ab5be9a345bb2bfcb2"
AUTHOR_URL = "https://github.com/dcherenson/staggered-integral-ocp"
LABELS = {
    "siocp": "Author SI-OCP + SSML + DT-MPC",
    "cusum": "CUSUM + wind observer + fixed tube MPC",
}
COLORS = {"siocp": "tab:orange", "cusum": "tab:blue"}
DT = 0.05
RESIDUAL_ACCEL_MAX = np.array([1.2, 1.2, 0.6])  # fixed design assumption, m/s^2
ESTIMATE_MAX = np.array([0.6, 0.6, 0.3])
U_MIN, U_MAX = np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 30.0])


# SI-OCP baseline: dcherenson/staggered-integral-ocp at PINNED_COMMIT.


class Plant:
    """
    Quadcopter Simulation (3D with Attitude).
    State x = [px, py, pz, vx, vy, vz, phi, theta]^T
    Control u = [p, q, T]^T (Roll rate, Pitch rate, Net Thrust)
    """

    def __init__(
        self,
        spatial_mode: bool = False,
        noise_source: Callable[[float], np.ndarray] | None = None,
    ):
        self.m = 1.0  # kg
        self.g = np.array([0, 0, -9.81])
        self.spatial_mode = spatial_mode  # Toggle for online spatial gradient shift

        # Aerodynamic drag coefficients (velocity-dependent)
        self.k_drag = np.array([0.3, 0.3, 0.15])  # Drag per axis (N·s/m)
        # Angle-dependent coupling coefficients (phi/theta modulate disturbance)
        self.k_angle = 0.4  # Lateral force coupling from tilt
        self.noise_source = noise_source

    def wind_velocity(self, t, p):
        v_wind_x = 2.0 * np.sin(0.5 * t) + 1.0 * np.sin(2.0 * t)
        v_wind_y = 2.4 * np.cos(0.4 * t) + 1.2 * np.cos(1.8 * t)
        v_wind_z = 1.0 * np.sin(0.3 * t)

        if self.spatial_mode:
            v_wind_x += 0.5 * (p[0] - 3.0)
            v_wind_y += 0.5 * np.clip(p[1] - 0.5, -1.0, 1.0)
            v_wind_z += 0.2 * p[2]
        wind_vec = np.array([v_wind_x, v_wind_y, v_wind_z])

        goal_pos = np.array([6.0, 0.0, 1.0])
        dist = np.linalg.norm(p - goal_pos)
        scale = np.clip(dist / 2.0, 0.0, 1.0)
        return wind_vec * scale

    def unmodeled_dynamics(self, t, p, v, angles):
        """Realistic unmodeled drag using body-frame relative velocity."""
        phi, theta = angles
        # Additive noise
        if self.noise_source is None:
            noise = np.random.randn(3) * np.array([0.2, 0.2, 0.1])
        else:
            noise = np.asarray(self.noise_source(float(t)), dtype=float)
            if noise.shape != (3,) or not np.all(np.isfinite(noise)):
                raise ValueError("noise_source must return a finite length-3 vector")
        v_wind = self.wind_velocity(t, p)
        v_rel = v - v_wind

        # Rotation matrix R_X(phi) @ R_Y(theta) from body to world
        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        R = np.array([[cy, 0, sy], [sx * sy, cx, -sx * cy], [-cx * sy, sx, cx * cy]])

        v_b = R.T @ v_rel
        # Body frame quadratic drag coefficients
        D_body = np.diag([0.3, 0.3, 0.6])  # Higher drag vertically due to rotors

        # Drag force in world frame: F = - R * D * (v_b * |v_b|)
        drag_force = -self.m * R @ D_body @ (v_b * np.abs(v_b))

        return drag_force + noise

    def f(self, x: np.ndarray) -> np.ndarray:
        # Expected nominal drift dynamics [dp, dv, dAngles] where dv = g
        return np.array([x[3], x[4], x[5], 0.0, 0.0, -9.81, 0.0, 0.0])

    def g_mat(self, x: np.ndarray) -> np.ndarray:
        # Input u = [p, q, T]^T
        # dv = R(phi, theta) * [0, 0, T]^T / m
        # dAngles = [p, q]
        phi = x[6]
        theta = x[7]

        mat = np.zeros((8, 3))
        # Acceleration from Thrust
        mat[3, 2] = np.sin(theta) / self.m
        mat[4, 2] = -np.cos(theta) * np.sin(phi) / self.m
        mat[5, 2] = np.cos(theta) * np.cos(phi) / self.m

        # Angular rates from control
        mat[6, 0] = 1.0  # p -> dot{phi}
        mat[7, 1] = 1.0  # q -> dot{theta}
        return mat

    def Delta(self, x: np.ndarray, t: float) -> np.ndarray:
        """Full-state additive mismatch (true - nominal), including vel/angle terms."""
        d = self.unmodeled_dynamics(t, x[0:3], x[3:6], x[6:8])
        return np.concatenate((np.zeros(3), d / self.m))

    def dynamics(self, t, state, u):
        """True dynamics: includes velocity- and angle-dependent disturbance."""
        # state = [px, py, pz, vx, vy, vz, phi, theta]
        p = state[0:3]
        v = state[3:6]
        angles = state[6:8]
        phi, theta = angles

        # True disturbance depends on velocity and body angles
        d = self.unmodeled_dynamics(t, p, v, angles)

        dp = v
        # m a = m g + R * T + d
        dv = self.g + d / self.m
        # Control u = [p, q, T]
        p_rate = u[0]
        q_rate = u[1]
        thrust = u[2]

        dv[0] += thrust / self.m * np.sin(theta)
        dv[1] += thrust / self.m * (-np.cos(theta) * np.sin(phi))
        dv[2] += thrust / self.m * (np.cos(theta) * np.cos(phi))

        dAngles = np.array([p_rate, q_rate])

        return np.concatenate((dp, dv, dAngles))

    def step(self, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Computes the state derivative and integrates one step forward."""
        sol = solve_ivp(self.dynamics, [t, t + dt], x, args=(u,), method="RK45")
        return sol.y[:, -1]


class StaggeredDriftScoreOCP:
    """
    Staggered Online Conformal Prediction (OCP) for drift scores.

    Uses N independent threads to handle delayed feedback (where thread j
    handles every N-th timestep), preventing overlap between a prediction
    and its eventual ground-truth realization.

    Each thread tracks its own quantile estimate q_k using online gradient
    descent on the pinball loss (Eq 29 from the paper).
    """

    def __init__(
        self, alpha: float, eta_const: float, N_threads: int, q_init: float = 0.0
    ):
        """
        Args:
            alpha (float): User-specified mis-coverage rate in (0, 1). Target coverage = 1 - alpha.
            eta_const (float): Constant step size for quantile updates.
            N_threads (int): Number of independent OCP threads (matching the feedback delay/horizon).
            q_init (float): Initial value for all quantile estimates.
        """
        if not (0 < alpha < 1):
            raise ValueError("Mis-coverage rate alpha must be between 0 and 1.")

        self.alpha = alpha
        self.eta_const = eta_const
        self.N = N_threads
        self.qs = np.full(self.N, max(0.0, q_init))
        self.k_step = 0

    def update(self, S_k: float) -> float:
        """
        Updates the active thread's quantile estimate based on the current score.
        Thread j = k % N is updated at each step.

        Args:
            S_k: The non-conformity score (observed drift) at the current step.

        Returns:
            The newly updated quantile for the active thread.
        """
        j = self.k_step % self.N

        # Indicator function: 1 if score EXCEEDS quantile (violation), else 0
        indicator = 1.0 if S_k > self.qs[j] else 0.0

        # Pinball loss gradient update
        self.qs[j] = max(0.0, self.qs[j] + self.eta_const * (indicator - self.alpha))
        q_out = self.qs[j]

        self.k_step += 1
        return q_out

    def get_quantile(self) -> float:
        """Returns the quantile estimate of the currently active thread."""
        j = self.k_step % self.N
        return self.qs[j]

    def get_dist_bound_from_quantile(self, q_k: float, T: float, L_d: float) -> float:
        """
        Converts an integral OCP quantile q_k to a disturbance distance bound d_bar.

        Args:
            q_k: Current quantile estimate.
            T: Time window length (seconds).
            L_d: Lipschitz bound on the disturbance derivative.

        Returns:
            d_bar: Upper bound on the disturbance displacement over T.
        """
        d_triangle = np.sqrt(2.0 * L_d * q_k)
        triangle_base = 2.0 * d_triangle / L_d
        if triangle_base <= T:
            # Disturbance is locally 'small' relative to the window
            d_bar = d_triangle
        else:
            # Window is too short to contain the full disturbance growth triangle
            d_bar = (q_k / T) + 0.5 * L_d * T
        return d_bar


class DynamicTubeMPC:
    """
    Implements Dynamic Tube MPC (DTMPC) treating the tube geometry size (Phi)
    and control bandwidth (alpha) as decision variables. Supports 3D Space and SSML DNN.
    """

    def __init__(
        self, plant, obstacles: list, H: int = 5, dt: float = 0.1, eta: float = 0.1
    ):
        self.plant = plant
        self.obstacles = (
            obstacles  # List of dicts {'pos': np.array([x,y,z]), 'r': radius}
        )
        self.H = H
        self.dt = dt
        self.eta = 0.0
        self.z_min = 0.8  # Altitude floor (m)
        self.z_max = 1.2  # Altitude ceiling (m)

        self.u_max = 30.0  # Increased for z-axis gravity compensation
        self.alpha_min = 0.5
        self.alpha_max = 10.0
        self.Phi = 0.05  # Initial tube size
        self.use_terminal_vel = True  # Toggle terminal velocity constraints
        self._prev_opt = None  # Warm-start: previous solution
        self._z_prev = (
            None  # Ancillary controller: predicted nominal state at next step
        )

        # Ancillary feedback gains: u = v* + K(z_prev - x)
        # Mapped to the 8D quad's physical control channels [p_rate, q_rate, thrust]:
        #   z[2] (height) / z[5] (vz)  → thrust correction
        #   z[0] (x-pos) / z[3] (vx)  → pitch rate q (sin(theta) → ax)
        #   z[1] (y-pos) / z[4] (vy)  → roll rate p  (-cos(θ)sin(φ) → ay)
        self.K_pos_z = 3.0  # z-position error → thrust
        self.K_vel_z = 2.0  # z-velocity error → thrust
        self.K_pos_xy = 2.0  # x/y-position error → pitch/roll rate
        self.K_vel_xy = 3.5  # x/y-velocity error → pitch/roll rate

    def compute_u(
        self,
        x: np.ndarray,
        xd: np.ndarray,
        disturbance_bound: float,
        d_hat: np.ndarray = None,
        model_nn=None,
    ):
        import torch

        # Check divergence
        if np.any(np.isnan(x)):
            return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False

        # For point-to-point, we start with a small tube centered on the drone
        self.Phi = 0.05

        # Evaluate Network Once for Zero-Order Hold Disturbance Prediction over the horizon
        if model_nn is not None and x.shape[0] == 8:
            # Assume hover state initial for disturbance evaluation
            x_in = np.concatenate((x[3:6], x[6:8]))
            with torch.no_grad():
                f_nn_acc = model_nn(torch.tensor(x_in, dtype=torch.float32)).numpy()
        else:
            f_nn_acc = np.zeros(3)

        # ── Build symbolic NLP with CasADi + IPOPT ────────────────────────────
        import casadi as ca

        n_V = 3 * self.H  # virtual control variables
        n_alpha = self.H  # bandwidth variables
        n_opt = n_V + n_alpha

        OPT = ca.SX.sym("OPT", n_opt)
        # CasADi reshape is column-major; to match numpy row-major (step-first)
        # we reshape as (3, H) then transpose -> V_sym[j,:] = [p_j, q_j, T_j]
        V_sym = ca.reshape(OPT[:n_V], 3, self.H).T
        alpha_sym = OPT[n_V:]

        # Numeric constants into CasADi
        x0_ca = ca.DM(x)
        xd_ca = ca.DM(xd)
        d_bar = ca.DM(disturbance_bound)
        nn_acc = ca.DM(f_nn_acc)

        # Symbolic rollout
        z = x0_ca
        phi = ca.DM(self.Phi)
        cost = ca.DM(0.0)
        g_sym = []  # inequality constraints (>= 0)

        obs_pos_list = [ca.DM(o["pos"]) for o in self.obstacles]
        obs_rad_list = [o["r"] for o in self.obstacles]

        for j in range(self.H):
            Vj = V_sym[j, :].T  # 3×1

            # ── nominal 8D quad dynamics (linearised g_mat at current z) ──
            phi_z = z[6]
            theta_z = z[7]
            # Compute f(z) symbolically
            f_z = ca.vertcat(
                z[3],
                z[4],
                z[5],
                ca.DM(0.0),
                ca.DM(0.0),
                ca.DM(-9.81),
                ca.DM(0.0),
                ca.DM(0.0),
            )
            # g_mat(z) @ Vj  (only thrust rows depend on angles)
            m = self.plant.m
            sin_t = ca.sin(theta_z)
            cos_t = ca.cos(theta_z)
            sin_p = ca.sin(phi_z)
            cos_p = ca.cos(phi_z)
            gV = ca.vertcat(
                ca.DM(0.0),
                ca.DM(0.0),
                ca.DM(0.0),
                sin_t / m * Vj[2],
                -cos_t * sin_p / m * Vj[2],
                cos_t * cos_p / m * Vj[2],
                Vj[0],  # p -> phi_dot
                Vj[1],
            )  # q -> theta_dot

            # NN feed-forward (3 acc channels, pad to 8)
            nn_full = ca.vertcat(ca.DM([0, 0, 0]), nn_acc, ca.DM([0, 0]))
            d_h = ca.DM(np.zeros(8)) if d_hat is None else ca.DM(d_hat)
            z_dot = f_z + gV + nn_full + d_h
            z = z + z_dot * self.dt

            # Tube dynamics
            phi = phi + self.dt * (-alpha_sym[j] * phi + d_bar + self.eta)

            # Stage cost
            pos_err = z[:3] - xd_ca[:3]
            vel = z[3:6]
            cost += 10.0 * ca.dot(pos_err, pos_err)
            cost += 2.0 * ca.dot(
                vel, vel
            )  # low penalty — terminal cost handles braking
            cost += 1.0 * ca.dot(Vj, Vj)
            cost += 0.5 * phi**2

            # Obstacle clearance constraints: dist - r - phi >= 0
            for obs_p, obs_r in zip(obs_pos_list, obs_rad_list):
                diff = z[:3] - obs_p
                dist = ca.sqrt(ca.dot(diff, diff) + 1e-6)  # smooth sqrt
                g_sym.append(dist - obs_r - phi)

            # Altitude constraints
            g_sym.append(z[2] - self.z_min)
            g_sym.append(self.z_max - z[2])

        # Terminal cost
        pos_err_f = z[:3] - xd_ca[:3]
        vel_f = z[3:6] - xd_ca[3:6]
        cost += 75.0 * ca.dot(pos_err_f, pos_err_f)
        cost += 50.0 * ca.dot(vel_f, vel_f)

        # Terminal Velocity Hard Constraints (Equality)
        num_ineq = len(g_sym)
        if self.use_terminal_vel:
            g_sym.append(z[3] - xd_ca[3])
            g_sym.append(z[4] - xd_ca[4])
            g_sym.append(z[5] - xd_ca[5])

        g_ca = ca.vertcat(*g_sym)

        # Bounds
        lbx = []
        ubx = []
        for _ in range(self.H):
            lbx += [-5.0, -5.0, 0.0]
            ubx += [5.0, 5.0, 30.0]
        lbx += [self.alpha_min] * self.H
        ubx += [self.alpha_max] * self.H

        lbg = [0.0] * g_ca.shape[0]
        # Inequality constraints are (>= 0), so ubg = inf
        # Terminal velocity constraints are (== 0), so ubg = 0
        ubg = [ca.inf] * num_ineq
        if self.use_terminal_vel:
            ubg += [0.0] * 3

        # Warm-start
        if self._prev_opt is not None:
            OPT_init = np.zeros(n_opt)
            prev_V = self._prev_opt[:n_V].reshape(self.H, 3)
            OPT_init[: 3 * (self.H - 1)] = prev_V[1:].flatten()
            OPT_init[3 * (self.H - 1) : n_V] = prev_V[-1]
            prev_a = self._prev_opt[n_V:]
            OPT_init[n_V : n_opt - 1] = prev_a[1:]
            OPT_init[n_opt - 1] = prev_a[-1]
        else:
            OPT_init = np.zeros(n_opt)
            for j in range(self.H):
                OPT_init[j * 3 + 2] = 9.81 * self.plant.m
            OPT_init[n_V:] = 1.0

        nlp = {"x": OPT, "f": cost, "g": g_ca}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-3,
            "ipopt.constr_viol_tol": 1e-3,
            "ipopt.acceptable_tol": 5e-2,
            "ipopt.acceptable_constr_viol_tol": 5e-2,
            "ipopt.acceptable_iter": 5,  # accept after 5 consecutive acceptable iters
            "ipopt.mu_strategy": "adaptive",  # better near non-convex constraint boundaries
            "ipopt.nlp_scaling_method": "gradient-based",
            "print_time": 0,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        def _cold_init():
            """Default hover warm-start (no previous solution)."""
            x0 = np.zeros(n_opt)
            for j in range(self.H):
                x0[j * 3 + 2] = 9.81 * self.plant.m
            x0[n_V:] = 1.0
            return x0

        sol = solver(x0=OPT_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        stats = solver.stats()

        # On Restoration_Failed the warm-start led to a bad iterate; retry cold.
        if stats.get("return_status") == "Restoration_Failed":
            sol = solver(x0=_cold_init(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            stats = solver.stats()

        success = stats["success"] or stats["return_status"] in (
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        )

        # Verify no hard constraint violation even if "acceptable"
        sol_x = np.array(sol["x"]).flatten()
        # ── end CasADi NLP ─────────────────────────────────────────────────────

        if success:
            # Double-check obstacle constraints with numpy rollout
            def check_constraints(opt_flat):
                V = opt_flat[:n_V].reshape(self.H, 3)
                al = opt_flat[n_V:]
                z_ = np.copy(x)
                phi_ = self.Phi
                cons = []
                for j in range(self.H):
                    for obs, r in zip(self.obstacles, obs_rad_list):
                        cons.append(np.linalg.norm(z_[:3] - obs["pos"]) - r - phi_)
                    cons += [z_[2] - self.z_min, self.z_max - z_[2]]
                    phi_val = phi_z = z_[6]
                    theta_val = z_[7]
                    gV_ = np.zeros(8)
                    gV_[3] = np.sin(theta_val) / m * V[j, 2]
                    gV_[4] = -np.cos(theta_val) * np.sin(phi_val) / m * V[j, 2]
                    gV_[5] = np.cos(theta_val) * np.cos(phi_val) / m * V[j, 2]
                    gV_[6] = V[j, 0]
                    gV_[7] = V[j, 1]
                    f_ = self.plant.f(z_)
                    nn_ = np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)])
                    d_h_np = np.zeros(8) if d_hat is None else d_hat
                    z_ = z_ + (f_ + gV_ + nn_ + d_h_np) * self.dt
                    phi_ = phi_ + self.dt * (
                        -al[j] * phi_ + disturbance_bound + self.eta
                    )
                return np.array(cons)

            if not stats["success"]:
                test_cons = check_constraints(sol_x)
                if np.any(test_cons < -0.05):
                    return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False

        if not success:
            print(
                f"Solver failed (IPOPT status: {stats.get('return_status', 'unknown')})"
            )
            return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False

        self._prev_opt = sol_x

        V_opt = sol_x[:n_V].reshape(self.H, 3)
        alphas_opt = sol_x[n_V:]

        # Apply first control (virtual)
        u_out = V_opt[0].copy()

        # ── Ancillary correction: u = v* + K(z_prev - x) ──────────────────
        if self._z_prev is not None:
            e = self._z_prev - x
            du_T = self.K_pos_z * e[2] + self.K_vel_z * e[5]
            du_p = self.K_pos_xy * e[1] + self.K_vel_xy * e[4]
            du_q = self.K_pos_xy * e[0] + self.K_vel_xy * e[3]
            u_out += np.array([du_p, du_q, du_T])
        # ──────────────────────────────────────────────────────────────────

        # Update Phi internally
        self.Phi = self.Phi + self.dt * (
            -alphas_opt[0] * self.Phi + disturbance_bound + self.eta
        )

        if np.any(np.isnan(u_out)):
            return np.zeros(3), np.zeros((self.H, 8)), np.zeros(self.H), False
        u_norm = np.linalg.norm(u_out)
        if u_norm > self.u_max:
            u_out = u_out / u_norm * self.u_max

        # Compute full prediction horizon & store nominal next state for ancillary
        z_pred = []
        phi_pred = []
        z = np.copy(x)
        phi = self.Phi
        for j in range(self.H):
            d_h_np = np.zeros(8) if d_hat is None else d_hat
            z_dot = (
                self.plant.f(z)
                + self.plant.g_mat(z) @ V_opt[j]
                + np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)])
                + d_h_np
            )
            z = z + z_dot * self.dt
            phi = phi + self.dt * (-alphas_opt[j] * phi + disturbance_bound + self.eta)
            z_pred.append(np.copy(z))
            phi_pred.append(phi)

        # Store z_pred[0] as the predicted nominal state at the NEXT timestep
        self._z_prev = z_pred[0].copy()

        return u_out, np.array(z_pred), np.array(phi_pred), True


QuadcopterSim = Plant
WEIGHTS_PATH = "ssml_weights.pt"
INPUT_DIM = 5
HIDDEN_DIM = 50
OUTPUT_DIM = 3


class SSMLNet(nn.Module):
    """
    SSML-AC neural network that estimates unmodeled acceleration disturbances.
    Takes velocity and attitude as input and outputs additive acceleration corrections.
    """

    def __init__(self):
        super(SSMLNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


def spectral_normalization_clip(model, v_max=10.0):
    """Clips each parameter tensor's spectral norm to v_max for Lipschitz bounding."""
    for param in model.parameters():
        norm = torch.linalg.norm(param)
        if norm > v_max:
            param.data = param.data * (v_max / norm)


def collect_offline_data():
    """
    Simulates the quadcopter under varied controls and records
    (velocity/attitude input, true acceleration mismatch) pairs for SSML pre-training.
    """
    print("Collecting Offline Data...")
    sim = QuadcopterSim()
    dt = 0.02  # 50 Hz
    t_end = 60.0  # 3 loops

    times = np.arange(0, t_end, dt)
    state = np.zeros(8)  # [px, py, pz, vx, vy, vz, phi, theta]

    data_x = []
    data_y = []

    for t in times:
        # Varied controls to explore state space
        p_u = 0.5 * np.sin(t)
        q_u = 0.5 * np.cos(1.5 * t)
        T_u = 9.81 * sim.m + 2.0 * np.sin(0.5 * t)
        u = np.array([p_u, q_u, T_u])

        v = state[3:6]
        angles = state[6:8]
        d_true = sim.unmodeled_dynamics(t, state[0:3], v, angles)

        # Input: [vx, vy, vz, phi, theta]
        x_in = np.concatenate((state[3:6], state[6:8]))

        # Target: additive acceleration mismatch = d_true / m
        y_target = d_true / sim.m

        data_x.append(x_in)
        data_y.append(y_target)

        sol = solve_ivp(sim.dynamics, [t, t + dt], state, args=(u,), method="RK45")
        state = sol.y[:, -1]

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def train_ssml(data_x, data_y):
    """Trains the SSMLNet using First-Order MAML (FOMAML) on offline data."""
    print("Training SSML...")
    model = SSMLNet()

    Ha = 25  # Adaptation horizon
    Ht = 25  # Training horizon
    alpha = 0.05  # Inner-loop step size
    beta = 0.01  # Outer-loop (meta) learning rate
    lambda_dir = 0.5  # Direct prediction loss weight
    lambda_norm = 0.05  # L2 weight regularization
    epochs = 500
    N = len(data_x)

    optimizer = optim.SGD(model.parameters(), lr=beta)

    X_tensor = torch.tensor(data_x, dtype=torch.float32)
    Y_tensor = torch.tensor(data_y, dtype=torch.float32)

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        num_tasks = 10
        for _ in range(num_tasks):
            start_idx = np.random.randint(0, N - Ha - Ht - 1)

            X_a = X_tensor[start_idx : start_idx + Ha]
            Y_a = Y_tensor[start_idx : start_idx + Ha]
            X_t = X_tensor[start_idx + Ha : start_idx + Ha + Ht]
            Y_t = Y_tensor[start_idx + Ha : start_idx + Ha + Ht]

            # Inner loop: FOMAML step
            pred_a = model(X_a)
            loss_a = torch.mean((pred_a - Y_a) ** 2)
            inner_grads = torch.autograd.grad(
                loss_a, model.parameters(), create_graph=False
            )
            fast_weights = [
                p - alpha * g.detach() for p, g in zip(model.parameters(), inner_grads)
            ]

            def inner_forward(x, weights):
                h1 = torch.nn.functional.relu(
                    torch.nn.functional.linear(x, weights[0], weights[1])
                )
                h2 = torch.nn.functional.relu(
                    torch.nn.functional.linear(h1, weights[2], weights[3])
                )
                h3 = torch.nn.functional.relu(
                    torch.nn.functional.linear(h2, weights[4], weights[5])
                )
                return torch.nn.functional.linear(h3, weights[6], weights[7])

            # Outer loop: evaluate adapted model on training set
            pred_t = inner_forward(X_t, fast_weights)
            loss_t_adapt = torch.mean((pred_t - Y_t) ** 2)

            # Direct prediction loss
            pred_t_dir = model(X_t)
            loss_t_dir = torch.mean((pred_t_dir - Y_t) ** 2)

            meta_loss = (loss_t_adapt + lambda_dir * loss_t_dir) / num_tasks
            meta_loss.backward()
            epoch_loss += meta_loss.item()

        # L2 weight regularization
        for p in model.parameters():
            if p.grad is not None:
                p.grad += 2 * lambda_norm * p.data

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        spectral_normalization_clip(model)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

    print("Pre-training Complete.")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")
    return model


def get_or_train_model():
    """Loads pre-trained weights if they exist, otherwise collects data and trains."""
    model = SSMLNet()
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
        print(f"Loaded pre-trained weights from {WEIGHTS_PATH}")
    else:
        data_x, data_y = collect_offline_data()
        model = train_ssml(data_x, data_y)
    return model


def flatten_params(model):
    """Returns all model parameters as a single flat tensor."""
    return torch.cat([p.flatten() for p in model.parameters()])


def assign_params(model, flat_params):
    """Loads a flat parameter vector back into the model."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx : idx + numel].view_as(p))
        idx += numel


def compute_jacobian(model, x_in):
    """
    Computes J = d(f_nn) / d(theta): Jacobian of network output w.r.t. all parameters.

    Args:
        x_in: numpy array of shape (INPUT_DIM,)

    Returns:
        Tensor of shape (OUTPUT_DIM, num_params)
    """
    model.eval()
    x_tensor = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)

    J_list = []
    for i in range(OUTPUT_DIM):
        model.zero_grad()
        out = model(x_tensor)[0, i]
        out.backward(retain_graph=True)
        grad_flat = torch.cat([p.grad.flatten() for p in model.parameters()])
        J_list.append(grad_flat)

    return torch.stack(J_list)


def compute_dist_bound(q_k, L_d, T_p):
    """
    Converts an OCP integral quantile q_k to a disturbance distance bound.

    Uses a piecewise formula derived from the Lipschitz structure of the disturbance:
        - If the implied triangle fits in T_p: d_bar = sqrt(2 * L_d * q_k)
        - Otherwise:                           d_bar = q_k / T_p + 0.5 * L_d * T_p
    """
    thresh = 0.5 * L_d * (T_p**2)
    T_safe = max(T_p, 1e-8)
    return np.where(
        q_k < thresh, q_k / T_safe + 0.5 * L_d * T_p, np.sqrt(2.0 * L_d * q_k)
    )


@dataclass(frozen=True)
class Config:
    seed: int = 1337
    flight_seconds: float = 20.0
    radius: float = 1.0
    cusum_profile: str = "original"
    cusum_residual_accel_max: tuple[float, float, float] = (1.2, 1.2, 0.6)
    cusum_estimate_max: tuple[float, float, float] = (0.6, 0.6, 0.3)
    cusum_angle_limit: float = 0.6
    cusum_horizontal_speed_limit: float = 2.0
    solver_max_iter: int | None = None
    solver_cpu_seconds: float | None = None
    cusum_horizon: int = 10  # same as original adaptation_on scenario
    observer_tau: float = 0.8
    observer_fast_tau: float = 0.12
    fast_seconds: float = 0.6
    cooldown_seconds: float = 0.8
    cusum_scale: float = 0.04
    cusum_slack: float = 0.5
    cusum_threshold: float = 5.0

    @property
    def steps(self):
        return round(self.flight_seconds / DT)


class WeakWindExperiment:
    """Shared environment, NOT part of either controller. Wind speed in m/s.

    Author quadratic aerodynamic force and nonlinear plant are retained.
    Use weak, time-only wind and bounded linearly interpolated force-noise
    replay. No dependence on RNG call order, algorithm, path, or ODE substeps.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.noise_samples = np.random.default_rng(cfg.seed).uniform(
            -1.0, 1.0, (cfg.steps + 3, 3)
        ) * np.array([0.02, 0.02, 0.01])
        self.identifier = hashlib.sha256(self.noise_samples.tobytes()).hexdigest()

    def noise(self, t):
        f = np.clip(t / DT, 0.0, len(self.noise_samples) - 1.0)
        i = min(int(np.floor(f)), len(self.noise_samples) - 2)
        return (1.0 - (f - i)) * self.noise_samples[i] + (f - i) * self.noise_samples[
            i + 1
        ]

    def velocity(self, t):
        c = self.cfg
        wind = 0.25 * np.array(
            [
                2.0 * np.sin(0.5 * t) + np.sin(2.0 * t),
                2.4 * np.cos(0.4 * t) + 1.2 * np.cos(1.8 * t),
                np.sin(0.3 * t),
            ]
        )
        # Smooth, small direction shifts; no instantaneous step in wind speed.
        shift = np.array([0.12, -0.1, 0.03])
        wind += 0.5 * (1.0 + np.tanh((t - 0.4 * c.flight_seconds) / 0.35)) * shift
        wind -= (
            0.5 * (1.0 + np.tanh((t - 0.75 * c.flight_seconds) / 0.35)) * 2.0 * shift
        )
        return wind

    def plant_class(self, base_plant):
        experiment = self

        class WeakPlant(base_plant):
            def __init__(self, spatial_mode=False):
                super().__init__(spatial_mode=False, noise_source=experiment.noise)

            def wind_velocity(self, t, p):
                return experiment.velocity(float(t))

        return WeakPlant


@contextlib.contextmanager
def siocp_solver_budget(cfg, solvers):
    """Apply IPOPT budgets during a baseline step; restore the factory afterward."""
    original = ca.nlpsol

    def create_solver(name, plugin, problem, options=None):
        options = dict(options or {})
        if plugin == "ipopt":
            if cfg.solver_max_iter is not None:
                options["ipopt.max_iter"] = cfg.solver_max_iter
            if cfg.solver_cpu_seconds is not None:
                options["ipopt.max_cpu_time"] = cfg.solver_cpu_seconds
        solver = original(name, plugin, problem, options)
        solvers.append(solver)
        return solver

    ca.nlpsol = create_solver
    try:
        yield
    finally:
        ca.nlpsol = original


class OriginalSIOCP:
    """SI-OCP baseline with its original per-step equations stored as attributes."""

    def __init__(self, cfg, plant_cls, weights_path):
        global WEIGHTS_PATH
        WEIGHTS_PATH = str(weights_path.resolve())
        self.cfg = cfg
        self.plant_cls = plant_cls
        initial = reference(0.0, cfg)[0]
        self.scenario = SimpleNamespace(
            name="Adaptation ON",
            seed=cfg.seed,
            dt_sim=0.05,
            dt_mpc=0.05,
            t_end=cfg.flight_seconds,
            x0=initial,
            x_goal=initial.copy(),
            goal_radius=0.0,
            spatial_wind=False,
            obstacles=[],
            mpc_horizon=10,
            alpha_ocp=0.1,
            eta_ocp=0.5,
            q_init_ocp=0.5,
            ddot_bound=2.0,
            dist_bound_init=2.0,
            gamma_lr=5.0,
            lambd=0.1,
        )
        self._initialize()
        if self.dt_sim != DT or self.sys_controller.H != cfg.cusum_horizon:
            raise ValueError("Baseline sample time/horizon changed")

    @property
    def state(self):
        return self.x.copy()

    def _initialize(self):
        np.random.seed(self.scenario.seed)
        torch.manual_seed(self.scenario.seed)
        self.t = 0.0
        self.dt_sim = self.scenario.dt_sim
        self.dt_mpc = self.scenario.dt_mpc
        self.n_substeps = int(round(self.dt_mpc / self.dt_sim))
        self.t_end = self.scenario.t_end
        self.sys_plant = self.plant_cls(spatial_mode=self.scenario.spatial_wind)
        self.sys_controller = DynamicTubeMPC(
            plant=self.sys_plant,
            obstacles=self.scenario.obstacles,
            H=self.scenario.mpc_horizon,
            dt=self.dt_mpc,
        )
        self.T_window = int(self.scenario.mpc_horizon * self.dt_mpc / self.dt_sim)
        self.ocp_integral = StaggeredDriftScoreOCP(
            alpha=self.scenario.alpha_ocp,
            eta_const=self.scenario.eta_ocp,
            N_threads=self.T_window,
            q_init=self.scenario.q_init_ocp,
        )
        self.T_p = self.T_window * self.dt_sim
        self.model = get_or_train_model()
        self.theta_0_flat = flatten_params(self.model).clone().detach()
        self.theta_flat = flatten_params(self.model).clone().detach()
        self.x = np.copy(self.scenario.x0)
        self.u = np.array([0.0, 0.0, 9.81 * self.sys_plant.m])
        self.u_old = np.copy(self.u)
        self.x_goal = self.scenario.x_goal
        self.goal_radius = self.scenario.goal_radius
        self.x_plotting = [np.copy(self.x)]
        self.t_plotting = [self.t]
        self.z_pred_plotting = [np.tile(self.x, (self.sys_controller.H, 1))]
        self.phi_pred_plotting = [
            np.tile(self.sys_controller.Phi, self.sys_controller.H)
        ]
        self.past_states = collections.deque(maxlen=self.T_window)
        self.past_f_nom = collections.deque(maxlen=self.T_window)
        self.dist_bound_plotting = [self.scenario.dist_bound_init]
        self.xd_plotting = []
        self.tube_plotting = [self.sys_controller.Phi]
        self.theta_plotting = [self.theta_flat.detach().numpy().copy()]
        self.disturbance_norm_plotting = [0.0]
        self.correct_bounds_count = 0
        self.total_steps_count = 0
        print("Running SSML DTMPC 3D Flight Simulation...")
        self.mpc_step_counter = 0
        self.t_sim_block_start = time.time()
        self.xd = self.x_goal

    def _advance(self):
        while True:
            if int(self.t / self.dt_sim + 0.5) % self.n_substeps == 0:
                self.mpc_step_counter += 1
                self.t_solve_start = time.time()
                self.dist_bound_ocp = float(
                    compute_dist_bound(
                        self.ocp_integral.get_quantile(),
                        self.scenario.ddot_bound,
                        self.T_p,
                    )
                )
                self.u, self.z_pred, self.phi_pred, self.success = (
                    self.sys_controller.compute_u(
                        self.x, self.xd, self.dist_bound_ocp, model_nn=self.model
                    )
                )
                self.solve_time = time.time() - self.t_solve_start
                if self.mpc_step_counter % 10 == 0:
                    self.t_now = time.time()
                    self.sim_block_time = self.t_now - self.t_sim_block_start
                    print(
                        f"t: {self.t:.2f}s | MPC Solve: {self.solve_time:.4f}s | Last 50 steps: {self.sim_block_time:.4f}s"
                    )
                    self.t_sim_block_start = self.t_now
                if not self.success:
                    print(f"Solver failed at t={self.t:.2f}s! Ending simulation early.")
                    break
                self.z_pred_plotting.append(self.z_pred)
                self.phi_pred_plotting.append(self.phi_pred)
                self.xd_plotting.append(np.copy(self.xd))
            else:
                self.z_pred_plotting.append(self.z_pred_plotting[-1])
                self.phi_pred_plotting.append(self.phi_pred_plotting[-1])
                self.xd_plotting.append(self.xd_plotting[-1])
            self.x_old = np.copy(self.x)
            self.x = self.sys_plant.step(self.x_old, self.u, self.t, self.dt_sim)
            self.x_in = np.concatenate((self.x_old[3:6], self.x_old[6:8]))
            with torch.no_grad():
                self.f_nn_acc = self.model(
                    torch.tensor(self.x_in, dtype=torch.float32)
                ).numpy()
            self.x_dot_pred = (
                self.sys_plant.f(self.x_old)
                + self.sys_plant.g_mat(self.x_old) @ self.u_old
                + np.concatenate([np.zeros(3), self.f_nn_acc, np.zeros(2)])
            )
            self.x_dot_true = self.sys_plant.dynamics(self.t, self.x_old, self.u_old)
            self.error_acc_true = (self.x_dot_true - self.x_dot_pred)[3:6]
            self.v_dot_est = (self.x[3:6] - self.x_old[3:6]) / self.dt_sim
            self.error_acc = self.v_dot_est - self.x_dot_pred[3:6]
            self.past_states.append(np.copy(self.x_old))
            self.past_f_nom.append(np.copy(self.x_dot_pred))
            self.L = len(self.past_states)
            self.S_I = 0.0
            if self.L >= self.T_window:
                self.x_buffer = np.array(self.past_states)
                self.f_nom_buffer = np.array(self.past_f_nom)
                self.x_k = self.x_buffer[-1]
                self.f_nom_rev = np.flip(self.f_nom_buffer, axis=0)
                self.x_rev = np.flip(self.x_buffer, axis=0)
                self.integrals_rev = np.cumsum(self.f_nom_rev, axis=0) * self.dt_sim
                self.pred_errors = self.x_k - self.x_rev - self.integrals_rev
                self.error_norms = np.linalg.norm(self.pred_errors, axis=1)
                self.S_I = np.max(self.error_norms)
                self.q_I = self.ocp_integral.update(self.S_I)
                self.dist_bound = float(
                    compute_dist_bound(self.q_I, self.scenario.ddot_bound, self.T_p)
                )
            else:
                self.dist_bound = self.scenario.dist_bound_init
            self.J = compute_jacobian(self.model, self.x_in).detach().numpy()
            self.total_steps_count += 1
            if np.linalg.norm(self.error_acc_true) <= self.dist_bound:
                self.correct_bounds_count += 1
            self.theta_dot = self.scenario.gamma_lr * np.dot(
                self.J.T, self.error_acc
            ) - self.scenario.lambd * (
                self.theta_flat.detach().numpy() - self.theta_0_flat.numpy()
            )
            self.theta_flat = (
                self.theta_flat
                + torch.tensor(self.theta_dot, dtype=torch.float32) * self.dt_sim
            )
            assign_params(self.model, self.theta_flat)
            spectral_normalization_clip(self.model)
            self.t += self.dt_sim
            self.x_plotting.append(np.copy(self.x))
            self.t_plotting.append(self.t)
            self.dist_bound_plotting.append(self.dist_bound)
            self.tube_plotting.append(self.sys_controller.Phi)
            self.theta_plotting.append(self.theta_flat.detach().numpy().copy())
            self.disturbance_norm_plotting.append(np.linalg.norm(self.error_acc_true))
            break

    def step(self, target):
        self.xd = target.copy()
        before = float(self.t)
        tic = time.perf_counter()
        solvers = []
        try:
            with siocp_solver_budget(self.cfg, solvers):
                self._advance()
        except RuntimeError as exc:
            status = (
                solvers[-1].stats().get("return_status", "unknown")
                if solvers
                else "unknown"
            )
            raise RuntimeError(f"SI-OCP solver status: {status}; {exc}") from exc
        if self.t <= before or not getattr(self, "success", False):
            status = (
                solvers[-1].stats().get("return_status", "unknown")
                if solvers
                else "unknown"
            )
            raise RuntimeError(
                f"Original SI-OCP DT-MPC did not complete the step; IPOPT status: {status}. See run.log for constraint or controller checks."
            )
        if not np.isfinite(self.x).all():
            raise RuntimeError("Original SI-OCP produced a nonfinite state")
        return {
            "time": float(self.t),
            "solver_seconds": float(self.solve_time),
            "step_seconds": time.perf_counter() - tic,
            "siocp_bound_used": float(self.dist_bound_ocp),
            "integral_score": float(self.S_I),
            "active_thread_after": int(self.ocp_integral.k_step % self.ocp_integral.N),
            "event": 0,
            "fast_update": 0,
            **{
                f"input_{a}": float(self.u[i])
                for i, a in enumerate(("p", "q", "thrust"))
            },
        }


class FixedHoverTube:
    """Fixed RPI enclosure for the sampled linearized 8-state error system.

    Contains a certified ellipsoidal tail of the zonotope sum. The W used
    here is an ASSUMED sampled plant/model mismatch bound, checked in runs;
    it is not a global derivation for all nonlinear flight states/inputs.
    """

    def __init__(self, cfg):
        self.residual_accel_max = np.asarray(cfg.cusum_residual_accel_max, dtype=float)
        self.state_min, self.state_max = STATE_MIN.copy(), STATE_MAX.copy()
        self.state_min[3:5] = -cfg.cusum_horizontal_speed_limit
        self.state_max[3:5] = cfg.cusum_horizontal_speed_limit
        self.state_min[-2:] = -cfg.cusum_angle_limit
        self.state_max[-2:] = cfg.cusum_angle_limit
        ac = np.zeros((8, 8))
        ac[:3, 3:6] = np.eye(3)
        ac[3, 7], ac[4, 6] = 9.81, -9.81
        bc = np.zeros((8, 3))
        bc[6, 0] = bc[7, 1] = bc[5, 2] = 1.0
        self.A, self.B, *_ = cont2discrete((ac, bc, np.eye(8), np.zeros((8, 3))), DT)
        self.D = np.vstack([0.5 * DT**2 * np.eye(3), DT * np.eye(3), np.zeros((2, 3))])
        self.Q = np.diag([30.0, 30.0, 35.0, 5.0, 5.0, 6.0, 3.0, 3.0])
        self.R = np.diag([0.15, 0.15, 0.4])
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = -np.linalg.solve(
            self.R + self.B.T @ self.P @ self.B, self.B.T @ self.P @ self.A
        )
        self.F = self.A + self.B @ self.K
        self.w_max = np.r_[
            0.5 * DT**2 * self.residual_accel_max,
            DT * self.residual_accel_max,
            0.0,
            0.0,
        ]
        # Ignore zero angle-disturbance generators, retain all nonzero channels.
        base = np.diag(self.w_max)[:, :6]
        pn = solve_discrete_lyapunov(self.F.T, np.eye(8))
        self.pn_inv = np.linalg.inv(pn)
        rho = np.sqrt(1.0 - 1.0 / np.linalg.eigvalsh(pn).max())
        power, blocks = np.eye(8), []
        for _ in range(300):
            blocks.append(power @ base)
            power = self.F @ power
        self.generators = np.hstack(blocks)
        import itertools

        corners = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
        tails = corners @ (power @ base).T
        maximum = np.sqrt(np.einsum("bi,ij,bj->b", tails, pn, tails)).max()
        self.tail_radius = maximum / (1.0 - rho)
        self.state_margin = self.support(np.eye(8))
        self.input_margin = self.support(self.K)
        if np.any(
            self.state_min + self.state_margin >= self.state_max - self.state_margin
        ):
            raise ValueError("Fixed tube consumes state region")
        if np.any(U_MIN + self.input_margin >= U_MAX - self.input_margin):
            raise ValueError("Fixed tube consumes input region")

    def support(self, rows):
        return np.abs(rows @ self.generators).sum(axis=1) + self.tail_radius * np.sqrt(
            np.einsum("bi,ij,bj->b", rows, self.pn_inv, rows)
        )


class CUSUMWindObserver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.estimate_max = np.asarray(cfg.cusum_estimate_max, dtype=float)
        self.estimate = np.zeros(3)
        self.plus, self.minus = np.zeros(3), np.zeros(3)
        self.fast_until, self.next_event = -1.0, 0.0

    def update(self, old, new, applied_u, tube, t):
        deviation = applied_u - np.array([0.0, 0.0, 9.81])
        nominal_next = tube.A @ old + tube.B @ deviation
        measured = (new[3:6] - nominal_next[3:6]) / DT
        innovation = measured - self.estimate
        c = self.cfg
        self.plus = np.maximum(
            0.0, self.plus + innovation / c.cusum_scale - c.cusum_slack
        )
        self.minus = np.maximum(
            0.0, self.minus - innovation / c.cusum_scale - c.cusum_slack
        )
        statistic = float(max(self.plus.max(), self.minus.max()))
        event = statistic > c.cusum_threshold and t + 1e-10 >= self.next_event
        if event:
            self.fast_until, self.next_event = (
                t + c.fast_seconds,
                t + c.cooldown_seconds,
            )
            self.plus[:], self.minus[:] = 0.0, 0.0
        fast = t < self.fast_until
        tau = c.observer_fast_tau if fast else c.observer_tau
        self.estimate = np.clip(
            self.estimate - np.expm1(-DT / tau) * innovation,
            -self.estimate_max,
            self.estimate_max,
        )
        return bool(event), bool(fast), statistic


class FixedTubeMPC:
    def __init__(self, tube, cfg):
        self.tube, self.cfg = tube, cfg
        n = cfg.cusum_horizon
        op = ca.Opti()
        z, v = op.variable(8, n + 1), op.variable(3, n)
        self.center, self.estimate = op.parameter(8), op.parameter(3)
        self.ref = op.parameter(8, n + 1)
        self.input_ref = op.parameter(3, n)  # Deviation from hover, not absolute input.
        op.subject_to(z[:, 0] == self.center)
        self.hover_input = np.array([0.0, 0.0, 9.81])
        cost = 0.0
        for j in range(n):
            op.subject_to(
                z[:, j + 1]
                == tube.A @ z[:, j] + tube.B @ v[:, j] + tube.D @ self.estimate
            )
            op.subject_to(
                op.bounded(
                    U_MIN + tube.input_margin - self.hover_input,
                    v[:, j],
                    U_MAX - tube.input_margin - self.hover_input,
                )
            )
            op.subject_to(
                ca.sumsqr(v[:, j] + self.hover_input)
                <= (30.0 - np.linalg.norm(tube.input_margin)) ** 2
            )
            error = z[:, j] - self.ref[:, j]
            input_error = v[:, j] - self.input_ref[:, j]
            cost += ca.mtimes([error.T, tube.Q, error]) + ca.mtimes(
                [input_error.T, tube.R, input_error]
            )
        for j in range(n + 1):
            op.subject_to(
                op.bounded(
                    tube.state_min + tube.state_margin,
                    z[:, j],
                    tube.state_max - tube.state_margin,
                )
            )
        error = z[:, n] - self.ref[:, n]
        op.minimize(cost + ca.mtimes([error.T, tube.P, error]))
        options = {
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-8,
            "max_iter": cfg.solver_max_iter if cfg.solver_max_iter is not None else 200,
        }
        if cfg.solver_cpu_seconds is not None:
            options["max_cpu_time"] = cfg.solver_cpu_seconds
        op.solver("ipopt", {"expand": True, "print_time": False}, options)
        self.op, self.z, self.v, self.previous = op, z, v, None

    def reference_state(self, t):
        """CUSUM-only attitude target from the near-hover nominal model.

        State order: [px, py, pz, vx, vy, vz, roll, pitch].
        ax = g*pitch and ay = -g*roll; angles are in radians.
        No disturbance compensation or input feedforward is added here.
        """
        state, acceleration = reference(t, self.cfg)
        target = state.copy()
        target[6] = -acceleration[1] / 9.81
        target[7] = acceleration[0] / 9.81
        return target

    def reference_input(self, t):
        """Nominal feedforward input [roll rate, pitch rate, thrust].

        The sampled near-hover model has roll_next=roll+DT*roll_rate
        and pitch_next=pitch+DT*pitch_rate. Use interval-average rates
        to connect the attitude references at t and t+DT exactly.
        Its vertical model is az=thrust-g (unit-mass plant in this file).
        Keep the observer estimate in the dynamics; do not compensate it twice.
        """
        start = self.reference_state(t)
        end = self.reference_state(t + DT)
        _, acceleration = reference(t, self.cfg)
        return np.array([
            (end[6] - start[6]) / DT,
            (end[7] - start[7]) / DT,
            9.81 + acceleration[2],
        ], dtype=float)

    def solve(self, center, actual, estimate, t):
        # Future reference preview for CUSUM only; SI-OCP remains unchanged.
        refs = [
            self.reference_state(t + j * DT)
            for j in range(self.cfg.cusum_horizon + 1)
        ]
        self.op.set_value(self.center, center)
        self.op.set_value(self.estimate, estimate)
        self.op.set_value(self.ref, np.array(refs).T)
        # The decision v is relative to hover: v_ref = u_ref - u_hover.
        input_refs = [
            self.reference_input(t + j * DT) - self.hover_input
            for j in range(self.cfg.cusum_horizon)
        ]
        self.op.set_value(self.input_ref, np.array(input_refs).T)
        if self.previous is not None:
            z, v = self.previous
            self.op.set_initial(self.z, np.column_stack([z[:, 1:], z[:, -1]]))
            self.op.set_initial(self.v, np.column_stack([v[:, 1:], v[:, -1]]))
        tic = time.perf_counter()
        try:
            solution = self.op.solve()
        except RuntimeError as exc:
            status = self.op.stats().get("return_status", "unknown")
            raise RuntimeError(f"CUSUM IPOPT status: {status}; {exc}") from exc
        elapsed = time.perf_counter() - tic
        self.previous = solution.value(self.z), solution.value(self.v)
        virtual = self.previous[1][:, 0]
        u = self.hover_input + virtual + self.tube.K @ (actual - center)
        nominal_next = (
            self.tube.A @ center + self.tube.B @ virtual + self.tube.D @ estimate
        )
        return u, nominal_next, elapsed


def csv_write(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_run(kind, cfg, destination, states, rows, completed, failure, extra):
    destination.mkdir(parents=True, exist_ok=True)
    x = np.array(states)
    t = np.arange(len(x)) * DT
    refs = np.array([reference(ti, cfg)[0] for ti in t])
    error = np.linalg.norm(x[:, :3] - refs[:, :3], axis=1)
    flight = (t > 0.0) & (t <= cfg.flight_seconds + 1e-9)
    summary = {
        "algorithm": kind,
        "seed": cfg.seed,
        "completed": completed,
        "steps": len(rows),
        "expected_steps": cfg.steps,
        "circle_rmse_m": (
            float(np.sqrt(np.mean(error[flight] ** 2))) if flight.any() else None
        ),
        "max_position_error_m": float(error.max()),
        "mean_solver_seconds": (
            float(np.mean([r["solver_seconds"] for r in rows])) if rows else None
        ),
        "events": sum(r["event"] for r in rows),
        "failure": failure,
        **extra,
    }
    np.savez(
        destination / "tracking_data.npz",
        time=t,
        actual=x,
        reference=refs,
        position_error=error,
    )
    csv_write(destination / "diagnostics.csv", rows)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_siocp(cfg, experiment, destination, weights):
    plant_cls = experiment.plant_class(Plant)
    baseline = OriginalSIOCP(cfg, plant_cls, weights)
    states, rows, completed, failure = [baseline.state], [], True, None
    for k in range(cfg.steps):
        try:
            row = baseline.step(reference(k * DT, cfg)[0])
            rows.append(row)
            states.append(baseline.state)
        except (RuntimeError, FloatingPointError) as exc:
            completed, failure = False, str(exc)
            print(f"SI-OCP failed at {k*DT:.2f}s: {failure}", flush=True)
            break
    return save_run(
        "siocp",
        cfg,
        destination,
        states,
        rows,
        completed,
        failure,
        {
            "author_commit": PINNED_COMMIT,
            "environment_id": experiment.identifier,
            "ssml_checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
    )


def run_cusum(cfg, experiment, destination):
    tube, observer = FixedHoverTube(cfg), CUSUMWindObserver(cfg)
    controller = FixedTubeMPC(tube, cfg)
    plant = experiment.plant_class(Plant)()
    x = reference(0.0, cfg)[0]
    z = x.copy()
    states, rows, completed, failure = [x.copy()], [], True, None
    max_w_ratio = 0.0
    for k in range(cfg.steps):
        t = k * DT
        estimate = observer.estimate.copy()
        try:
            u, nominal_next, elapsed = controller.solve(z, x, estimate, t)
            if (
                np.any(u < U_MIN - 1e-6)
                or np.any(u > U_MAX + 1e-6)
                or np.linalg.norm(u) > 30.0 + 1e-6
            ):
                raise RuntimeError(
                    "CUSUM actuator constraints violated; no clipping/fallback"
                )
            old = x.copy()
            new = plant.step(old, u, t, DT)
            if not np.isfinite(new).all():
                raise RuntimeError("Nonfinite plant state")
            expected = (
                tube.A @ old + tube.B @ (u - controller.hover_input) + tube.D @ estimate
            )
            mismatch = new - expected
            ratios = np.abs(mismatch[:6]) / tube.w_max[:6]
            max_w_ratio = max(max_w_ratio, float(ratios.max()))
            if np.any(np.abs(mismatch) > tube.w_max + 1e-7):
                raise RuntimeError(
                    f"Fixed sampled mismatch assumption violated: ratio={ratios.max():.3f}"
                )
            e = new - nominal_next
            if np.any(np.abs(e) > tube.state_margin + 1e-6):
                raise RuntimeError("Fixed tube coordinate enclosure exceeded")
            if np.any(new < tube.state_min - 1e-6) or np.any(
                new > tube.state_max + 1e-6
            ):
                raise RuntimeError("CUSUM physical state constraints violated")
            event, fast, statistic = observer.update(old, new, u, tube, t + DT)
            row = {
                "time": t + DT,
                "solver_seconds": elapsed,
                "event": int(event),
                "fast_update": int(fast),
                "cusum_statistic": statistic,
                "sampled_mismatch_ratio": float(ratios.max()),
                "position_tube_ratio": float(
                    np.max(np.abs(e[:3]) / tube.state_margin[:3])
                ),
            }
            for i, axis in enumerate("xyz"):
                row[f"estimate_before_{axis}"] = float(estimate[i])
                row[f"estimate_after_{axis}"] = float(observer.estimate[i])
            for i, axis in enumerate(("p", "q", "thrust")):
                row[f"input_{axis}"] = float(u[i])
            rows.append(row)
            x, z = new, nominal_next
            states.append(x.copy())
        except (RuntimeError, FloatingPointError) as exc:
            completed, failure = False, str(exc)
            print(f"CUSUM failed at {t:.2f}s: {failure}", flush=True)
            break
    return save_run(
        "cusum",
        cfg,
        destination,
        states,
        rows,
        completed,
        failure,
        {
            "author_commit": PINNED_COMMIT,
            "environment_id": experiment.identifier,
            "ssml_checkpoint_sha256": None,
            "max_sampled_mismatch_ratio": max_w_ratio,
            "cusum_profile": cfg.cusum_profile,
            "residual_accel_max": tube.residual_accel_max.tolist(),
            "observer_estimate_max": observer.estimate_max.tolist(),
            "state_min": tube.state_min.tolist(),
            "state_max": tube.state_max.tolist(),
            "fixed_state_margin": tube.state_margin.tolist(),
            "fixed_input_margin": tube.input_margin.tolist(),
            "fixed_sampled_w_max": tube.w_max.tolist(),
        },
    )


SHAPES = ("hover", "line", "sine", "circle", "infinity", "ellipse")
WINDS = ("no_wind", "weak", "medium", "strong", "mixed", "gust")

# Mở rộng vùng XY cho line dài 6 m và ellipse.
# Không sửa controller SI-OCP gốc của tác giả.
STATE_MIN = np.array([-20.0, -20.0, 0.8, -2.0, -2.0, -1.0, -0.6, -0.6])
STATE_MAX = np.array([20.0, 20.0, 1.2, 2.0, 2.0, 1.0, 0.6, 0.6])


@dataclass(frozen=True)
class SuiteConfig(Config):
    trajectory: str = "circle"
    wind: str = "weak"
    line_length: float = 6.0
    sine_amplitude: float = 0.5
    ellipse_a: float = 2.0
    ellipse_b: float = 1.0


def reference(t, cfg):
    if cfg.trajectory == "hover":
        state = np.zeros(8)
        state[2] = 1.0  # giữ vị trí (0, 0, 1) m
        return state, np.zeros(3)

    a = np.clip(t / cfg.flight_seconds, 0.0, 1.0)
    s = 10 * a**3 - 15 * a**4 + 6 * a**5
    ds = (30 * a**2 - 60 * a**3 + 30 * a**4) / cfg.flight_seconds
    dds = (60 * a - 180 * a**2 + 120 * a**3) / cfg.flight_seconds**2

    if t >= cfg.flight_seconds:
        ds = dds = 0.0

    th, w, wd = 2 * np.pi * s, 2 * np.pi * ds, 2 * np.pi * dds
    x, acc = np.zeros(8), np.zeros(3)
    x[2] = 1.0

    if cfg.trajectory in ("line", "sine"):
        x[0] = cfg.line_length * s
        x[3] = cfg.line_length * ds
        acc[0] = cfg.line_length * dds

        if cfg.trajectory == "sine":
            r = cfg.sine_amplitude
            x[1] = r * np.sin(th)
            x[4] = r * np.cos(th) * w
            acc[1] = r * (-np.sin(th) * w * w + np.cos(th) * wd)

    elif cfg.trajectory == "infinity":
        x[0], x[1] = 2 * np.sin(th), 0.75 * np.sin(2 * th)
        x[3], x[4] = 2 * np.cos(th) * w, 1.5 * np.cos(2 * th) * w
        acc[0] = 2 * (-np.sin(th) * w * w + np.cos(th) * wd)
        acc[1] = 1.5 * (-2 * np.sin(2 * th) * w * w + np.cos(2 * th) * wd)

    else:
        rx, ry = (
            (cfg.radius, cfg.radius)
            if cfg.trajectory == "circle"
            else (cfg.ellipse_a, cfg.ellipse_b)
        )
        x[0], x[1] = rx * (1 - np.cos(th)), ry * np.sin(th)
        x[3], x[4] = rx * np.sin(th) * w, ry * np.cos(th) * w
        acc[0] = rx * (np.cos(th) * w * w + np.sin(th) * wd)
        acc[1] = ry * (-np.sin(th) * w * w + np.cos(th) * wd)

    return x, acc


class SuiteWind(WeakWindExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)
        description = (
            f"suite-v1/{cfg.wind}/" f"{cfg.flight_seconds}/flight-only/{DT}"
        )
        self.identifier = hashlib.sha256(
            description.encode() + self.noise_samples.tobytes()
        ).hexdigest()

    def velocity(self, t):
        c = self.cfg

        # No wind: retain the same force-noise replay and nonlinear plant.
        if c.wind == "no_wind":
            return np.zeros(3, dtype=float)

        # Mixed: chuyển dần weak -> medium -> strong.
        if c.wind == "mixed":
            g1 = 0.5 * (1 + np.tanh((t - 0.33 * c.flight_seconds) / 0.5))
            g2 = 0.5 * (1 + np.tanh((t - 0.67 * c.flight_seconds) / 0.5))
            scale = 0.25 + 0.25 * g1 + 0.5 * g2
        else:
            scale = {
                "weak": 0.25,
                "medium": 0.5,
                "strong": 1.0,
                "gust": 0.5, #0.25
            }[c.wind]

        v = scale * np.array(
            [
                2 * np.sin(0.5 * t) + np.sin(2 * t),
                2.4 * np.cos(0.4 * t) + 1.2 * np.cos(1.8 * t),
                np.sin(0.3 * t),
            ]
        )

        shift = (scale / 0.25) * np.array([0.12, -0.1, 0.03])
        v += 0.5 * (1 + np.tanh((t - 0.4 * c.flight_seconds) / 0.35)) * shift
        v -= (1 + np.tanh((t - 0.75 * c.flight_seconds) / 0.35)) * shift

        # Gust: gió nền scale=0.5 cộng ba Gaussian pulse không đổi.
        if c.wind == "gust":
            pulses = (
                (0.35, [1.2, -1.0, 0.3]),
                (0.65, [-1.0, 1.5, -0.2]),
                (0.85, [0.8, 0.9, 0.25]),
            )
            for fraction, direction in pulses:
                width = 0.025 * c.flight_seconds + 0.1
                pulse = np.exp(-0.5 * ((t - fraction * c.flight_seconds) / width) ** 2)
                v += pulse * np.array(direction)

        return v


def run_safe(function, kind, cfg, experiment, destination, weights):
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                if kind == "siocp":
                    summary = function(cfg, experiment, destination, weights)
                else:
                    summary = function(cfg, experiment, destination)
            except Exception as exc:
                traceback.print_exc()
                summary = save_run(
                    kind,
                    cfg,
                    destination,
                    [reference(0.0, cfg)[0]],
                    [],
                    False,
                    str(exc),
                    {"environment_id": experiment.identifier},
                )

    summary["flight_rmse_m"] = summary.pop(
        "circle_rmse_m", summary.get("flight_rmse_m")
    )
    summary.update(
        wind=cfg.wind,
        trajectory=cfg.trajectory,
        environment_id=experiment.identifier,
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def case_plot(case, cfg, summaries, make_gif):
    data = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    curve = np.array(
        [reference(t, cfg)[0] for t in np.linspace(0, cfg.flight_seconds, 400)]
    )
    axes[0].plot(curve[:, 0], curve[:, 1], "k--", label="Reference")

    for kind in LABELS:
        with np.load(case / kind / "tracking_data.npz") as source:
            data[kind] = {k: source[k].copy() for k in source.files}

        d, summary = data[kind], summaries[kind]
        suffix = (
            ""
            if summary["completed"]
            else f" [STOP {summary['steps']}/{summary['expected_steps']}]"
        )
        label = kind.upper() + suffix

        axes[0].plot(
            d["actual"][:, 0],
            d["actual"][:, 1],
            color=COLORS[kind],
            label=label,
        )
        if not summary["completed"]:
            axes[0].plot(
                d["actual"][-1, 0],
                d["actual"][-1, 1],
                marker="x",
                color=COLORS[kind],
                markersize=8,
            )
            axes[1].plot(
                d["time"][-1],
                d["position_error"][-1],
                marker="x",
                color=COLORS[kind],
                markersize=8,
            )
        axes[1].plot(
            d["time"],
            d["position_error"],
            color=COLORS[kind],
            label=label,
        )

    axes[0].set(
        xlabel="X (m)",
        ylabel="Y (m)",
        title=f"{cfg.wind} / {cfg.trajectory}",
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(
        xlabel="Time (s)",
        ylabel="3D position error (m)",
        xlim=(0, cfg.flight_seconds),
    )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(case / "comparison.png", dpi=140)
    plt.close(fig)


def suite_gif(root, winds, shapes):
    """One GIF for all selected winds/shapes, with both pipelines per panel."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    cases = {}
    for shape in shapes:
        for wind in winds:
            folder = root / wind / shape
            cfg = SuiteConfig(**json.loads((folder / "config.json").read_text()))
            data, summaries = {}, {}
            for kind in LABELS:
                with np.load(folder / kind / "tracking_data.npz") as source:
                    data[kind] = {key: source[key].copy() for key in source.files}
                summaries[kind] = json.loads((folder / kind / "summary.json").read_text())
            cases[wind, shape] = cfg, data, summaries

    fig, axes = plt.subplots(
        2 * len(shapes), len(winds), squeeze=False,
        figsize=(3.4 * len(winds), 5.8 * len(shapes)),
    )
    artists = []
    total_time = max(cfg.flight_seconds for cfg, _, _ in cases.values())
    for row, shape in enumerate(shapes):
        # Identical scales across wind cases for the same trajectory.
        positions = np.vstack([
            d["actual"][:, :2]
            for wind in winds for d in cases[wind, shape][1].values()
        ] + [np.array([reference(t, cases[wind, shape][0])[0][:2]
                          for t in np.linspace(0, cases[wind, shape][0].flight_seconds, 200)])
             for wind in winds])
        low, high = positions.min(axis=0), positions.max(axis=0)
        padding = np.maximum(0.05, 0.10 * (high - low))
        max_error = max(float(d["position_error"].max())
                        for wind in winds for d in cases[wind, shape][1].values())
        for col, wind in enumerate(winds):
            cfg, data, summaries = cases[wind, shape]
            top, bottom = axes[2 * row, col], axes[2 * row + 1, col]
            curve = np.array([reference(t, cfg)[0] for t in np.linspace(0, cfg.flight_seconds, 300)])
            if np.allclose(curve[:, :2], curve[0, :2]):
                top.plot(curve[0, 0], curve[0, 1], "k*", ms=9, label="Reference")
            else:
                top.plot(curve[:, 0], curve[:, 1], "k--", lw=1, label="Reference")
            top.set(title=f"{wind} / {shape}", xlabel="X (m)", ylabel="Y (m)",
                    xlim=(low[0]-padding[0], high[0]+padding[0]),
                    ylim=(low[1]-padding[1], high[1]+padding[1]))
            top.set_aspect("equal", adjustable="box")
            bottom.set(xlabel="Time (s)", ylabel="3D error (m)",
                       xlim=(0, total_time), ylim=(0, max(0.01, max_error * 1.15)))
            for index, kind in enumerate(LABELS):
                line, = top.plot([], [], color=COLORS[kind], label=kind.upper())
                dot, = top.plot([], [], "o", color=COLORS[kind], ms=4)
                error, = bottom.plot([], [], color=COLORS[kind])
                label = top.text(0.02, 0.98-index*0.10, "", transform=top.transAxes,
                                 va="top", fontsize=7, color=COLORS[kind])
                artists.append((data[kind], summaries[kind], kind, line, dot, error, label))
            top.legend(loc="lower left", fontsize=6)
            top.grid(alpha=0.25)
            bottom.grid(alpha=0.25)

    title = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(t):
        for d, summary, kind, line, dot, error, label in artists:
            n = max(1, int(np.searchsorted(d["time"], t + 1e-9, side="right")))
            n = min(n, len(d["time"]))
            x = d["actual"][:n]
            line.set_data(x[:, 0], x[:, 1])
            dot.set_data([x[-1, 0]], [x[-1, 1]])
            error.set_data(d["time"][:n], d["position_error"][:n])
            stopped = not summary["completed"] and t + 1e-9 >= d["time"][-1]
            status = (f"STOP {d['time'][-1]:.2f}s" if stopped
                      else f"{d['position_error'][n-1]:.3f} m")
            label.set_text(f"{kind.upper()}: {status}")
        title.set_text(f"SI-OCP vs CUSUM | t = {t:.1f} / {total_time:g} s | playback 2x")
        return []

    frames = np.arange(0.0, total_time, 0.2).tolist() + [total_time]
    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    destination = root / "all_cases_comparison.gif"
    try:
        animation.save(destination, writer=PillowWriter(fps=10), dpi=80)
    finally:
        plt.close(fig)
    print(f"Combined GIF saved to: {destination}", flush=True)


def suite_figures(root, records, winds, shapes, total_time):
    lookup = {(r["wind"], r["trajectory"], r["algorithm"]): r for r in records}

    for mode, filename in (
        ("path", "all_trajectories.png"),
        ("error", "all_tracking_errors.png"),
    ):
        fig, axes = plt.subplots(
            len(winds),
            len(shapes),
            squeeze=False,
            figsize=(4 * len(shapes), 3.4 * len(winds)),
        )

        for i, wind in enumerate(winds):
            for j, shape in enumerate(shapes):
                ax = axes[i, j]

                for index, kind in enumerate(LABELS):
                    row = lookup[(wind, shape, kind)]
                    with np.load(root / wind / shape / kind / "tracking_data.npz") as d:
                        suffix = "" if row["completed"] else f" STOP {row['steps']}"
                        if mode == "path":
                            if index == 0:
                                cfg = SuiteConfig(
                                    **json.loads(
                                        (
                                            root / wind / shape / "config.json"
                                        ).read_text()
                                    )
                                )
                                curve = np.array(
                                    [
                                        reference(t, cfg)[0]
                                        for t in np.linspace(0, cfg.flight_seconds, 300)
                                    ]
                                )
                                ax.plot(
                                    curve[:, 0],
                                    curve[:, 1],
                                    "k--",
                                    label="Reference",
                                )
                            ax.plot(
                                d["actual"][:, 0],
                                d["actual"][:, 1],
                                color=COLORS[kind],
                                label=kind.upper() + suffix,
                            )
                        else:
                            ax.plot(
                                d["time"],
                                d["position_error"],
                                color=COLORS[kind],
                                label=kind.upper() + suffix,
                            )

                if mode == "path":
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.set(xlabel="X (m)", ylabel="Y (m)")
                else:
                    ax.set(
                        xlabel="Time (s)",
                        ylabel="Error (m)",
                        xlim=(0, total_time),
                    )
                ax.set_title(f"{wind} / {shape}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6)

        fig.tight_layout()
        fig.savefig(root / filename, dpi=140)
        plt.close(fig)

    for metric in ("flight_rmse_m",):
        valid = [r[metric] for r in records if r["completed"] and r[metric] is not None]
        top = max(valid, default=1.0) or 1.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad("lightgrey")

        for ax, kind in zip(axes, LABELS):
            matrix = np.full((len(winds), len(shapes)), np.nan)
            for i, wind in enumerate(winds):
                for j, shape in enumerate(shapes):
                    row = lookup[(wind, shape, kind)]
                    if row["completed"] and row[metric] is not None:
                        matrix[i, j] = row[metric]

            image = ax.imshow(
                matrix,
                cmap=cmap,
                vmin=0.0,
                vmax=top,
                aspect="auto",
            )
            for i in range(len(winds)):
                for j in range(len(shapes)):
                    value = matrix[i, j]
                    text = f"{value:.3f}" if np.isfinite(value) else "STOP"
                    color = (
                        "white"
                        if np.isfinite(value) and value > 0.65 * top
                        else "black"
                    )
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
            ax.set_xticks(
                range(len(shapes)),
                shapes,
                rotation=30,
                fontsize=9,
            )
            ax.set_yticks(range(len(winds)), winds, fontsize=9)
            ax.set_title(kind.upper() + " / " + metric)

        fig.colorbar(
            image,
            ax=axes.tolist(),
            label="RMSE (m)",
            shrink=0.8,
        )
        fig.savefig(
            root / (metric + ".png"),
            dpi=140,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_saved_cases(root, winds, shapes, make_gif):
    """Use each case's stored configuration and preserve its STOP status."""
    plotted = 0
    available = set()
    for wind in winds:
        for shape in shapes:
            case = root / wind / shape
            required = [case / "config.json"] + [
                case / kind / name
                for kind in LABELS
                for name in ("summary.json", "tracking_data.npz")
            ]
            missing = [path for path in required if not path.is_file()]
            if missing:
                print(f"Skipped {wind}/{shape}: missing {missing[0]}", flush=True)
                continue
            cfg = SuiteConfig(**json.loads((case / "config.json").read_text()))
            summaries = {
                kind: json.loads((case / kind / "summary.json").read_text())
                for kind in LABELS
            }
            case_plot(case, cfg, summaries, make_gif)
            plotted += 1
            available.add((wind, shape))
            for kind, summary in summaries.items():
                if not summary["completed"]:
                    print(
                        f"{wind}/{shape} {kind}: partial data only; "
                        f"{summary['steps']}/{summary['expected_steps']} steps; "
                        f"reason: {summary.get('failure')}",
                        flush=True,
                    )
    if not plotted:
        raise SystemExit(f"No saved cases to plot under {root}")
    if make_gif:
        missing = [(w, s) for w in winds for s in shapes if (w, s) not in available]
        if missing:
            raise SystemExit(f"Combined GIF needs all selected cases; missing: {missing}")
        suite_gif(root, winds, shapes)
    print(f"Regenerated plots for {plotted} cases under {root}")


def cusum_profile_settings(wind, profile):
    """Experimental simulation tuning; keep the original profile available."""
    if profile == "strong-mixed-tuned" and wind in {"strong", "mixed"}:
        return {
            "cusum_profile": "strong-mixed-tuned",
            "cusum_residual_accel_max": (4.0, 4.0, 1.2),
            "cusum_estimate_max": (4.0, 4.0, 1.2),
            "cusum_angle_limit": 1.2,
            "cusum_horizontal_speed_limit": 3.0,
        }
    return {"cusum_profile": "original"}


def main():
    parser = argparse.ArgumentParser(
        description="6 wind cases (including no_wind); exactly 2 pipelines per case"
    )
    parser.add_argument("--wind", choices=("all", *WINDS), default="all")
    parser.add_argument(
        "--trajectory",
        choices=("all", *SHAPES, "straight", "figure8"),
        default="all",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=6886)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--line-length", type=float, default=6.0)
    parser.add_argument("--sine-amplitude", type=float, default=0.5)
    parser.add_argument("--ellipse-a", type=float, default=2.0)
    parser.add_argument("--ellipse-b", type=float, default=1.0)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--cusum-profile",
        choices=("original", "strong-mixed-tuned"),
        default="strong-mixed-tuned",
        help="Simulation-only CUSUM limits for strong/mixed; other winds retain original limits",
    )
    parser.add_argument(
        "--solver-max-iter",
        type=int,
        default=None,
        help="IPOPT iteration budget per solve for both pipelines (defaults: SI-OCP 300, CUSUM 200)",
    )
    parser.add_argument(
        "--solver-cpu-seconds",
        type=float,
        default=None,
        help="Optional IPOPT CPU-time budget per solve; omitted means no added CPU-time cap",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from saved data, including partial STOP runs; no simulation or training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/all_pipeline_tests_flight_only"),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gif", action="store_true")
    group.add_argument("--no-gif", action="store_false", dest="gif")
    parser.set_defaults(gif=False)
    args = parser.parse_args()

    if args.solver_max_iter is not None and args.solver_max_iter < 1:
        parser.error("--solver-max-iter must be positive")
    if args.solver_cpu_seconds is not None and (
        not np.isfinite(args.solver_cpu_seconds) or args.solver_cpu_seconds <= 0
    ):
        parser.error("--solver-cpu-seconds must be positive and finite")

    values = (
        args.duration,
        args.radius,
        args.line_length,
        args.sine_amplitude,
        args.ellipse_a,
        args.ellipse_b,
    )
    if not all(np.isfinite(v) and v > 0 for v in values) or args.seed < 0:
        parser.error(
            "Durations/dimensions must be positive and finite; "
            "seed must be nonnegative"
        )
    if not all(
        np.isclose(v / DT, round(v / DT)) for v in (args.duration,)
    ):
        parser.error("Durations must be multiples of 0.05 seconds")

    shape = {
        "straight": "line",
        "figure8": "infinity",
    }.get(args.trajectory, args.trajectory)

    winds = WINDS if args.wind == "all" else (args.wind,)
    shapes = SHAPES if shape == "all" else (shape,)

    base = args.output_dir.resolve()
    root = base / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_saved_cases(root, winds, shapes, args.gif)
        return

    weights = args.weights.resolve() if args.weights else base / "ssml_weights.pt"
    if args.weights and not weights.is_file():
        parser.error(f"Checkpoint missing: {weights}")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    records = []
    metadata = {
        "arguments": vars(args)
        | {
            "weights": str(weights),
            "output_dir": str(base),
        },
        "winds": winds,
        "shapes": shapes,
        "pipelines": list(LABELS),
        "author_commit": PINNED_COMMIT,
        "baseline_source": "SI-OCP definitions included directly in this main.py",
        "cusum_profile": args.cusum_profile,
        "cusum_limits_note": (
            "strong-mixed-tuned changes CUSUM residual/observer bounds, angle and horizontal speed limits "
            "only for strong and mixed; see each config.json. This is an empirically tuned "
            "simulation configuration, not a certified disturbance bound."
        ),
        "solver_defaults": {"siocp_max_iter": 300, "cusum_max_iter": 200},
        "wind_definitions": {
            "no_wind": "zero wind velocity; force-noise replay retained",
            "weak": "0.25 x author time field",
            "medium": "0.5 x",
            "strong": "1.0 x",
            "mixed": "smooth weak->medium->strong",
            "gust": "medium (scale=0.5) + 3 smooth pulses",
        },
    }
    (root / "suite.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cases = ((w, s) for w in winds for s in shapes)

    for index, (wind, shape) in enumerate(cases, 1):
        print(
            f"\n[{index}/{len(winds)*len(shapes)}] " f"{wind} / {shape}",
            flush=True,
        )

        cfg = SuiteConfig(
            seed=args.seed,
            flight_seconds=args.duration,
            radius=args.radius,
            trajectory=shape,
            wind=wind,
            line_length=args.line_length,
            sine_amplitude=args.sine_amplitude,
            ellipse_a=args.ellipse_a,
            ellipse_b=args.ellipse_b,
            **cusum_profile_settings(wind, args.cusum_profile),
            solver_max_iter=args.solver_max_iter,
            solver_cpu_seconds=args.solver_cpu_seconds,
        )
        case = root / wind / shape
        case.mkdir(parents=True, exist_ok=True)
        (case / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2),
            encoding="utf-8",
        )

        experiment = SuiteWind(cfg)
        summaries = {}

        csv_write(
            case / "environment.csv",
            [
                {
                    "time": k * DT,
                    **{
                        f"wind_{a}_mps": float(experiment.velocity(k * DT)[i])
                        for i, a in enumerate("xyz")
                    },
                    **{
                        f"noise_{a}_N": float(experiment.noise(k * DT)[i])
                        for i, a in enumerate("xyz")
                    },
                }
                for k in range(cfg.steps + 1)
            ],
        )

        for kind, function in (
            ("siocp", run_siocp),
            ("cusum", run_cusum),
        ):
            print(f"  Running {kind.upper()}...", flush=True)

            summary = run_safe(
                function,
                kind,
                cfg,
                experiment,
                case / kind,
                weights,
            )
            summaries[kind] = summary

            keys = (
                "algorithm",
                "seed",
                "completed",
                "steps",
                "expected_steps",
                "flight_rmse_m",
                "max_position_error_m",
                "mean_solver_seconds",
                "events",
                "failure",
            )
            row = {
                "wind": wind,
                "trajectory": shape,
                **{k: summary.get(k) for k in keys},
            }
            records.append(row)

            status = "OK" if row["completed"] else "STOP"
            print(
                f"  {kind.upper()}: {status} "
                f"{row['steps']}/{row['expected_steps']}",
                flush=True,
            )
            if not row["completed"]:
                print(f"  Reason: {row['failure']}", flush=True)
            csv_write(root / "all_results.csv", records)

        case_plot(case, cfg, summaries, args.gif)

    suite_figures(
        root,
        records,
        winds,
        shapes,
        args.duration,
    )

    if args.gif:
        suite_gif(root, winds, shapes)

    passed = sum(bool(r["completed"]) for r in records)
    print(
        f"\nSuite finished: {len(records)} runs; "
        f"{passed} complete; {len(records)-passed} stopped."
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()

#uv run python main.py --wind all --trajectory line --duration 20 --seed 6886 --gif --output-dir output/line