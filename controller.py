import numpy as np
from plant import Plant


class DynamicTubeMPC:
    def __init__(
        self,
        plant,
        obstacles: list,
        H: int = 5,
        dt: float = 0.1,
        eta: float = 0.1,
    ):
        self.plant = plant
        self.obstacles = obstacles
        self.H = H
        self.dt = dt

        self.eta = 0.0
        self.z_min = 0.8
        self.z_max = 1.2

        self.u_max = 30.0
        self.alpha_min = 0.5
        self.alpha_max = 10.0
        self.Phi = 0.05

        self.use_terminal_vel = True
        self._prev_opt = None
        self._z_prev = None

        self.K_pos_z = 3.0
        self.K_vel_z = 2.0
        self.K_pos_xy = 2.0
        self.K_vel_xy = 3.5

    def compute_u(
        self,
        x: np.ndarray,
        xd: np.ndarray,
        disturbance_bound: float,
        d_hat: np.ndarray = None,
        model_nn=None,
    ):
        import torch
        import casadi as ca

        def failed_result():
            return (
                np.zeros(3),
                np.zeros((self.H, 8)),
                np.zeros(self.H),
                False,
            )

        if np.any(np.isnan(x)):
            return failed_result()

        self.Phi = 0.05

        if model_nn is not None and x.shape[0] == 8:
            x_in = np.concatenate((x[3:6], x[6:8]))

            with torch.no_grad():
                f_nn_acc = model_nn(
                    torch.tensor(x_in, dtype=torch.float32)
                ).numpy()
        else:
            f_nn_acc = np.zeros(3)

        # Đích cố định: (8,).
        # Quỹ đạo tracking: (H, 8).
        refs = np.asarray(xd, dtype=float)
        tracking = refs.ndim == 2

        if refs.shape == (8,):
            refs = np.tile(refs, (self.H, 1))

        if (
            refs.shape != (self.H, 8)
            or not np.isfinite(refs).all()
        ):
            raise ValueError(
                "xd must be finite with shape (8,) or (H, 8)"
            )

        n_V = 3 * self.H
        n_alpha = self.H
        n_opt = n_V + n_alpha

        OPT = ca.SX.sym("OPT", n_opt)
        V_sym = ca.reshape(OPT[:n_V], 3, self.H).T
        alpha_sym = OPT[n_V:]

        x0_ca = ca.DM(x)
        xd_ca = ca.DM(refs[-1])
        d_bar = ca.DM(disturbance_bound)
        nn_acc = ca.DM(f_nn_acc)

        z = x0_ca
        phi = ca.DM(self.Phi)
        cost = ca.DM(0.0)
        g_sym = []

        m = self.plant.m

        obs_pos_list = [
            ca.DM(o["pos"]) for o in self.obstacles
        ]
        obs_rad_list = [
            o["r"] for o in self.obstacles
        ]

        for j in range(self.H):
            Vj = V_sym[j, :].T

            phi_z = z[6]
            theta_z = z[7]

            f_z = ca.vertcat(
                z[3],
                z[4],
                z[5],
                0.0,
                0.0,
                -9.81,
                0.0,
                0.0,
            )

            sin_t = ca.sin(theta_z)
            cos_t = ca.cos(theta_z)
            sin_p = ca.sin(phi_z)
            cos_p = ca.cos(phi_z)

            gV = ca.vertcat(
                0.0,
                0.0,
                0.0,
                sin_t / m * Vj[2],
                -cos_t * sin_p / m * Vj[2],
                cos_t * cos_p / m * Vj[2],
                Vj[0],
                Vj[1],
            )

            nn_full = ca.vertcat(
                ca.DM([0.0, 0.0, 0.0]),
                nn_acc,
                ca.DM([0.0, 0.0]),
            )

            d_h = (
                ca.DM(np.zeros(8))
                if d_hat is None
                else ca.DM(d_hat)
            )

            z_dot = f_z + gV + nn_full + d_h
            z = z + z_dot * self.dt

            phi = phi + self.dt * (
                -alpha_sym[j] * phi
                + d_bar
                + self.eta
            )

            # Chi phí bám tham chiếu ở bước j.
            target = ca.DM(refs[j])
            pos_err = z[:3] - target[:3]

            if tracking:
                vel_err = z[3:6] - target[3:6]
            else:
                vel_err = z[3:6]

            position_weight = 300.0 if tracking else 10.0
            velocity_weight = 20.0 if tracking else 2.0

            cost += (
                position_weight
                * ca.dot(pos_err, pos_err)
            )
            cost += (
                velocity_weight
                * ca.dot(vel_err, vel_err)
            )
            cost += ca.dot(Vj, Vj)
            cost += 0.5 * phi**2

            for obs_p, obs_r in zip(
                obs_pos_list,
                obs_rad_list,
            ):
                diff = z[:3] - obs_p
                dist = ca.sqrt(
                    ca.dot(diff, diff) + 1e-6
                )
                g_sym.append(dist - obs_r - phi)

            g_sym.append(z[2] - self.z_min)
            g_sym.append(self.z_max - z[2])

        # Chi phí tại cuối khoảng dự đoán.
        pos_err_f = z[:3] - xd_ca[:3]
        vel_err_f = z[3:6] - xd_ca[3:6]

        terminal_position_weight = (
            1000.0 if tracking else 75.0
        )

        cost += (
            terminal_position_weight
            * ca.dot(pos_err_f, pos_err_f)
        )
        cost += 50.0 * ca.dot(vel_err_f, vel_err_f)

        num_ineq = len(g_sym)

        if self.use_terminal_vel:
            g_sym.append(z[3] - xd_ca[3])
            g_sym.append(z[4] - xd_ca[4])
            g_sym.append(z[5] - xd_ca[5])

        g_ca = ca.vertcat(*g_sym)

        lbx = []
        ubx = []

        for _ in range(self.H):
            lbx += [-5.0, -5.0, 0.0]
            ubx += [5.0, 5.0, 30.0]

        lbx += [self.alpha_min] * self.H
        ubx += [self.alpha_max] * self.H

        lbg = [0.0] * g_ca.shape[0]
        ubg = [ca.inf] * num_ineq

        if self.use_terminal_vel:
            ubg += [0.0] * 3

        def cold_init():
            initial = np.zeros(n_opt)

            for j in range(self.H):
                initial[j*3 + 2] = 9.81 * m

            initial[n_V:] = 1.0
            return initial

        if self._prev_opt is not None:
            OPT_init = np.zeros(n_opt)

            prev_V = self._prev_opt[:n_V].reshape(
                self.H, 3
            )
            OPT_init[:3*(self.H - 1)] = (
                prev_V[1:].flatten()
            )
            OPT_init[3*(self.H - 1):n_V] = prev_V[-1]

            prev_a = self._prev_opt[n_V:]
            OPT_init[n_V:n_opt - 1] = prev_a[1:]
            OPT_init[n_opt - 1] = prev_a[-1]
        else:
            OPT_init = cold_init()

        nlp = {
            "x": OPT,
            "f": cost,
            "g": g_ca,
        }

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-3,
            "ipopt.constr_viol_tol": 1e-3,
            "ipopt.acceptable_tol": 5e-2,
            "ipopt.acceptable_constr_viol_tol": 5e-2,
            "ipopt.acceptable_iter": 5,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
            "print_time": 0,
        }

        solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            opts,
        )

        sol = solver(
            x0=OPT_init,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
        )
        stats = solver.stats()

        if stats.get("return_status") == "Restoration_Failed":
            sol = solver(
                x0=cold_init(),
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
            )
            stats = solver.stats()

        success = (
            stats["success"]
            or stats["return_status"] in (
                "Solve_Succeeded",
                "Solved_To_Acceptable_Level",
                "Maximum_Iterations_Exceeded",
            )
        )

        sol_x = np.array(sol["x"]).flatten()

        def check_constraints(opt_flat):
            V = opt_flat[:n_V].reshape(self.H, 3)
            al = opt_flat[n_V:]

            z_check = np.copy(x)
            phi_check = self.Phi
            cons = []

            for j in range(self.H):
                for obs, radius in zip(
                    self.obstacles,
                    obs_rad_list,
                ):
                    cons.append(
                        np.linalg.norm(
                            z_check[:3] - obs["pos"]
                        )
                        - radius
                        - phi_check
                    )

                cons += [
                    z_check[2] - self.z_min,
                    self.z_max - z_check[2],
                ]

                roll = z_check[6]
                pitch = z_check[7]

                gV_check = np.zeros(8)
                gV_check[3] = (
                    np.sin(pitch) / m * V[j, 2]
                )
                gV_check[4] = (
                    -np.cos(pitch)
                    * np.sin(roll)
                    / m
                    * V[j, 2]
                )
                gV_check[5] = (
                    np.cos(pitch)
                    * np.cos(roll)
                    / m
                    * V[j, 2]
                )
                gV_check[6] = V[j, 0]
                gV_check[7] = V[j, 1]

                nn_check = np.concatenate([
                    np.zeros(3),
                    f_nn_acc,
                    np.zeros(2),
                ])
                d_check = (
                    np.zeros(8)
                    if d_hat is None
                    else d_hat
                )

                z_check = z_check + (
                    self.plant.f(z_check)
                    + gV_check
                    + nn_check
                    + d_check
                ) * self.dt

                phi_check += self.dt * (
                    -al[j] * phi_check
                    + disturbance_bound
                    + self.eta
                )

            return np.array(cons)

        if success and not stats["success"]:
            test_cons = check_constraints(sol_x)

            if np.any(test_cons < -0.05):
                return failed_result()

        if not success:
            print(
                "Solver failed: "
                f"{stats.get('return_status', 'unknown')}"
            )
            return failed_result()

        self._prev_opt = sol_x

        V_opt = sol_x[:n_V].reshape(self.H, 3)
        alphas_opt = sol_x[n_V:]

        u_out = V_opt[0].copy()

        # Phản hồi hiệu chỉnh sai lệch giữa dự đoán và thực tế.
        if self._z_prev is not None:
            e = self._z_prev - x

            du_T = (
                self.K_pos_z * e[2]
                + self.K_vel_z * e[5]
            )

            # Roll dương gây gia tốc Y âm.
            roll_sign = -1.0 if tracking else 1.0
            du_p = roll_sign * (
                self.K_pos_xy * e[1]
                + self.K_vel_xy * e[4]
            )
            du_q = (
                self.K_pos_xy * e[0]
                + self.K_vel_xy * e[3]
            )

            u_out += np.array([du_p, du_q, du_T])

        self.Phi += self.dt * (
            -alphas_opt[0] * self.Phi
            + disturbance_bound
            + self.eta
        )

        if np.any(np.isnan(u_out)):
            return failed_result()

        u_norm = np.linalg.norm(u_out)

        if u_norm > self.u_max:
            u_out = u_out / u_norm * self.u_max

        # Dự đoán trạng thái cho toàn bộ khoảng MPC.
        z_pred = []
        phi_pred = []

        z = np.copy(x)
        phi = self.Phi

        for j in range(self.H):
            d_h_np = (
                np.zeros(8)
                if d_hat is None
                else d_hat
            )

            nn_full_np = np.concatenate([
                np.zeros(3),
                f_nn_acc,
                np.zeros(2),
            ])

            z_dot = (
                self.plant.f(z)
                + self.plant.g_mat(z) @ V_opt[j]
                + nn_full_np
                + d_h_np
            )

            z = z + z_dot * self.dt

            phi += self.dt * (
                -alphas_opt[j] * phi
                + disturbance_bound
                + self.eta
            )

            z_pred.append(np.copy(z))
            phi_pred.append(phi)

        self._z_prev = z_pred[0].copy()

        return (
            u_out,
            np.array(z_pred),
            np.array(phi_pred),
            True,
        )