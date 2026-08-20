import numpy as np
import scipy as sp
import tomllib

rng = np.random.default_rng()


class ICINGModel:
    def __init__(
        self,
        params,
        dt,
        initial_state,
        u_funcs,
        process_noise_vars,
        measurement_noise_var,
        SI_augment=True,
        ekf = True,
    ):
        self.params = (
            params
            if params
            else {
                "pG": 0.006,
                "alphaG": 1.0 / 65.0,
                "VG": 13.3,
                "EGP": 1.16,
                "CNS": 0.3,
                "d1": -np.log(0.5) / 20.0,
                "d2": -np.log(0.5) / 100.0,
                "Pmax": 6.11,
                "xL": 0.67,
                "nL": 0.1578,
                "alphaI": 0.0017,
                "nK": 0.0542,
                "nC": 0.0075,
                "nI": 0.0075,
                "k1": 45.7,
                "k2": 1.5,
                "k3": 1000,
                "umin": 16.7,
                "umax": 266.7,
                "VI": 4.0,
            }
        )

        self.dt = dt
        self.initial_state = initial_state

        self.process_noise_var = process_noise_vars
        self.measurement_noise_var = measurement_noise_var

        self.ufuncs = u_funcs

        self.SI_augment = SI_augment
        self.ekf = ekf

        with open("../config.toml", "rb") as f:
            cfg = tomllib.load(f)
        root_module = "global"
        self.init_cov = cfg[root_module]["init_cov"]
        self.u_en_max = cfg[root_module]["u_en_max"]
        self.SI_params = cfg[root_module]["SI_params"]
        self.init_cov_scale = cfg[root_module]["init_cov_scale"]

    def get_inputs(self, t):
        Uex = self.ufuncs["Uex"](t)
        D = self.ufuncs["D"](t)
        PN = self.ufuncs["PN"](t)
        return [Uex, D, PN]

    def state_update(self, x, u, t, curr_SI=None):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        if self.SI_augment:
            # x = [G, Q, I, P1, P2, Uen, SI]
            G, Q, I, P1, P2, Uen, SI = x[:, 0]  # noqa: E741
        else:
            # x = [G, Q, I, P1, P2, Uen]
            G, Q, I, P1, P2, Uen = x[:, 0]  # noqa: E741
            SI = curr_SI

        # u = [Uex, D, PN]
        Uex, D, PN = u
        x_next = np.zeros_like(x)
        # Calculate P
        P = np.minimum(self.params["d2"] * P2, self.params["Pmax"]) + PN
        # P = PN
        # Update of G
        x_next[0, 0] = G + self.dt * (
            -self.params["pG"] * G
            - SI * G * Q / (1 + self.params["alphaG"] * Q)
            + (P + self.params["EGP"] - self.params["CNS"]) / self.params["VG"]
        )
        # Update of Q
        x_next[1, 0] = Q + self.dt * (
            self.params["nI"] * (I - Q)
            - self.params["nC"] * Q / (1 + self.params["alphaG"] * Q)
        )
        # Update of I
        x_next[2, 0] = I + self.dt * (
            -self.params["nK"] * I
            - self.params["nL"] * I / (1 + self.params["alphaI"] * I)
            - self.params["nI"] * (I - Q)
            + Uex / self.params["VI"]
            + (1 - self.params["xL"]) * Uen / self.params["VI"]
        )
        # # Update of P1
        x_next[3, 0] = P1 + self.dt * (-self.params["d1"] * P1 + D)
        # # Update P2
        x_next[4, 0] = P2 + self.dt * (
            -np.minimum(self.params["d2"] * P2, self.params["Pmax"])
            + self.params["d1"] * P1
        )
        # Update Uen
        x_next[5, 0] = self.params["k1"] * np.exp(
            -I * self.params["k2"] / self.params["k3"]
        )
        if self.SI_augment:
            # Update SI
            x_next[6, 0] = SI

        if self.SI_augment is True and self.ekf is False:
            noise = np.random.randn() * np.sqrt(self.process_noise_var[6])
            x_next[6, 0] = x_next[6,0] * (1 + noise)
        # # else:
        #     x_next_noise = x_next + w
        return x_next

    def observation(self, x, t):
        return x[0] + rng.normal(0.0, np.sqrt(self.measurement_noise_var))

    def measurement_log_likelihood(self, y, x, t):
        return sp.stats.norm.logpdf(
            y, loc=x[0], scale=np.sqrt(self.measurement_noise_var)
        )

    def simulate(self, t_eval, SI_func):
        num_states = len(self.initial_state)
        num_samples = len(t_eval)
        x = np.zeros((num_states, num_samples), dtype=np.float64)
        x[:,0] = self.initial_state
        x_next = self.initial_state
        x_next = np.expand_dims(x_next, axis=1)
        for ti in range(1, len(t_eval)):
            u = self.get_inputs(ti) 
            if self.SI_augment is True:
                # curr_SI = x_next[-1,0]
                curr_SI = SI_func(ti)
            else:
                curr_SI = SI_func(ti)
            x_next = self.state_update(x_next, u, ti, curr_SI=curr_SI)
            for s in range(num_states):
                if s < 6:
                    x_next[s,0] += np.random.randn() * np.sqrt(self.process_noise_var[s])
                else:
                    # noise = np.random.randn() * np.sqrt(self.process_noise_var[s])
                    # x_next[s,0] = x_next[s,0] * (1 + noise)
                    x_next[s,0] *= 1
            x[:,ti] = x_next[:,0]

        return x

