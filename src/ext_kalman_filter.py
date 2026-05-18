import matplotlib.pyplot as plt
import numpy as np
import tomllib


class ExtendedKalmanFilter:
    def __init__(self, num_states, model, ts_meas):
        # Parameters
        self.num_states = num_states
        self.model = model
        self.ts_meas = ts_meas
        self.SI_augment = model.SI_augment

        with open("../config.toml", "rb") as f:
            cfg = tomllib.load(f)
        root_module = "global"
        self.init_cov = cfg[root_module]["init_cov"]
        self.u_en_max = cfg[root_module]["u_en_max"]
        self.SI_max = cfg[root_module]["SI_max"]
 
        self.process_noise_factor = cfg[root_module]["process_noise_factor"]
        self.init_cov_scale = cfg[root_module]["init_cov_scale"]

    def initialize_filter(self):
        state = self.model.initial_state.copy()
        if self.SI_augment:
            noise_vars = self.init_cov_scale * np.diag(
                np.append(self.init_cov, self.SI_max)
            )
            self.Q = np.diag(
                np.append(
                    self.model.process_noise_var[:-1],
                    (self.process_noise_factor * self.SI_max) ** 2,
                )
            )
        else:
            noise_vars = self.init_cov_scale * np.diag(self.init_cov)
            self.Q = np.diag(self.model.process_noise_var)
        self.h_model_func = self.calc_h_model_func()
        self.R = np.array([[self.model.measurement_noise_var]])
        return state, noise_vars**2

    def calc_f_model_func(self, m, curr_SI):
        # Variable Initialization
        f_model_func = np.zeros((self.num_states, self.num_states))
        dt = self.model.dt
        pG = self.model.params["pG"]
        alphaG = self.model.params["alphaG"]
        nI = self.model.params["nI"]
        nC = self.model.params["nC"]
        nK = self.model.params["nK"]
        nL = self.model.params["nL"]
        alphaI = self.model.params["alphaI"]
        VI = self.model.params["VI"]
        xL = self.model.params["xL"]
        d1 = self.model.params["d1"]
        d2 = self.model.params["d2"]
        Pmax = self.model.params["Pmax"]
        k1 = self.model.params["k1"]
        k2 = self.model.params["k2"]
        k3 = self.model.params["k3"]
        G = m[0, 0]
        Q = m[1, 0]
        I = m[2, 0]  # noqa: E741
        P2 = m[4, 0]
        if self.SI_augment:
            SI = m[6, 0]
        else:
            SI = curr_SI

        # Jacobian Matrix Calculation
        # BG equation
        f_model_func[0, 0] = 1 - dt * pG - dt * SI * Q / (1 + alphaG * Q)
        f_model_func[0, 1] = -dt * SI * G / (1 + alphaG * Q) ** 2
        if self.SI_augment:
            f_model_func[0, 6] = -dt * G * Q / (1 + alphaG * Q)

        # Q equation
        f_model_func[1, 1] = 1 - dt * nI - dt * nC / (1 + alphaG * Q) ** 2
        f_model_func[1, 2] = dt * nI

        # I equation
        f_model_func[2, 1] = dt * nI
        f_model_func[2, 2] = 1 - dt * nK - dt * nL / (1 + alphaI * I) ** 2 - dt * nI
        f_model_func[2, 5] = dt * (1 - xL) / VI

        # P1 equation
        f_model_func[3, 3] = 1 - dt * d1

        # P2 equation
        f_model_func[4, 3] = dt * d1
        if d2 * P2 < Pmax:
            f_model_func[4, 4] = 1 - dt * d2

        # Uen Equation
        f_model_func[5, 2] = (-k1 * k2 / k3) * np.exp(-I * k2 / k3)

        if self.SI_augment:
            # SI equation
            f_model_func[6, 6] = 1

        return f_model_func

    def calc_h_model_func(self):
        calc_h_model_func = np.zeros((1, self.num_states))
        calc_h_model_func[0] = 1
        return calc_h_model_func

    def filter_predict(self, state_update, noise_var_update, t, curr_SI):
        u_vec = self.model.get_inputs(t)
        f_model_func = self.calc_f_model_func(state_update, curr_SI)
        state_predict = self.model.state_update(state_update, u_vec, t, curr_SI)
        noise_var_predict = (
            np.matmul(f_model_func, np.matmul(noise_var_update, f_model_func.T))
            + self.Q
        )
        return state_predict, noise_var_predict

    def filter_update(self, state_predict, noise_var_predict, y, output_params=False):
        S = (
            np.matmul(
                self.h_model_func, np.matmul(noise_var_predict, self.h_model_func.T)
            )
            + self.R
        )
        K = np.matmul(
            noise_var_predict, np.matmul(self.h_model_func.T, np.linalg.inv(S))
        )
        v = np.array([y - state_predict[0]])
        state_update = state_predict + np.matmul(K, v)
        noise_var_update = noise_var_predict - np.matmul(K, np.matmul(S, K.T))
        if not output_params:
            return state_update, noise_var_update
        else:
            return state_update, noise_var_update, S, v

    def predict_to_time(self, state, noise_var, t, curr_SI=None):
        # Run prediction from t_last_measurement to t
        predictions = np.zeros((int(t - self.t_last_measurement), self.num_states))
        step = 0

        for ti in range(self.t_last_measurement, t):
            state_next, noise_var_next = self.filter_predict(
                state, noise_var, ti, curr_SI
            )
            predictions[step] = state_next[:, 0]
            state = state_next
            noise_var = noise_var_next
            step += 1
        return predictions

    def filter_iteration(self, state, noise_var, t, yk=None, curr_SI=None):
        # Make prediction
        state_next, noise_var_next = self.filter_predict(state, noise_var, t, curr_SI)

        # if there is a measurement, update filter
        if yk is not None:
            self.t_last_measurement = t
            return self.filter_update(state_next, noise_var_next, yk)

        else:
            return state_next, noise_var_next

    class simulate:
        def __init__(self, t, t_meas, G_meas, filter, SI_fixed=None):
            self.t = t
            self.t_meas = t_meas
            self.G_meas = G_meas
            self.filter = filter
            self.SI_fixed = SI_fixed

        def run(self):
            num_states = self.filter.num_states
            G_est = np.zeros_like(self.t)
            G_pred = np.zeros_like(self.t)
            Q_est = np.zeros_like(self.t)
            I_est = np.zeros_like(self.t)
            P1_est = np.zeros_like(self.t)
            P2_est = np.zeros_like(self.t)
            Uen_est = np.zeros_like(self.t)
            SI_est = np.zeros_like(self.t)
            saved_state = np.zeros((len(self.t), num_states))
            saved_noise_var = np.zeros((len(self.t), num_states, num_states))
            state, noise_var = self.filter.initialize_filter()

            for idx, ti in enumerate(self.t):
                # If a measurement occured at this time
                if ti in self.t_meas:
                    sample_idx = np.argmin(np.abs(self.t_meas - ti))
                    # if sample_idx > 1: break
                    # print(f"Got measurement {sample_idx} at time {int(ti)}")
                    state_next, noise_var_next = self.filter.filter_iteration(
                        state,
                        noise_var,
                        int(ti),
                        self.G_meas[sample_idx],
                        curr_SI=self.SI_fixed,
                    )

                    # # Predict 2 hours ahead (if not last measurement)
                    # if ti != self.t_meas[-1]:
                    #     # print(f"Predicting the next 2 hours...")
                    #     pred = self.filter.predict_to_time(state_next, noise_var_next, t = int(ti+self.filter.ts_meas))
                    #     G_pred[idx:idx+self.filter.ts_meas] = pred[:,0]

                # If no measurement
                else:
                    state_next, noise_var_next = self.filter.filter_iteration(
                        state, noise_var, ti, curr_SI=self.SI_fixed
                    )

                saved_state[idx], saved_noise_var[idx] = (
                    state_next[:, 0],
                    noise_var_next[:, 0],
                )
                G_est[idx] = state_next[0, 0]
                Q_est[idx] = state_next[1, 0]
                I_est[idx] = state_next[2, 0]
                P1_est[idx] = state_next[3, 0]
                P2_est[idx] = state_next[4, 0]
                Uen_est[idx] = state_next[5, 0]
                if self.SI_fixed is None:
                    SI_est[idx] = state_next[6, 0]
                state = state_next
                noise_var = noise_var_next

            # print("Done")

            # UNEXPECTED BEHAVIOUR THAT ESTIMATES ARE AN INTEGER. FOR NOW ONY USE SAVED_STATE AND SAVED_NOISE_VAR FOR STATE CALCULATIONS
            return (
                saved_state,
                saved_noise_var,
                G_est,
                Q_est,
                I_est,
                P1_est,
                P2_est,
                Uen_est,
                SI_est,
                G_pred,
            )

        def plot(
            self,
            saved_state,
            saved_noise_var,
            G_est,
            Q_est,
            I_est,
            P1_est,
            P2_est,
            Uen_est,
            SI_est,
            G_pred,
            I_true,
            model,
        ):

            fig, ax = model.plot_sim()
            ax[0].plot(
                self.t_meas,
                self.G_meas,
                marker="o",
                label="BG measurements",
                color="k",
                linestyle="--",
            )
            ax[0].plot(self.t, G_est, label="Fitted BG", color="blue", linestyle="--")
            # ax[0].plot(self.t[1:], G_pred[:-1], label="2 hour ahead prediction", color="green")
            uen_true = model.params["k1"] * np.exp(
                I_true * model.params["k2"] / model.params["k3"]
            )
            ax[1].plot(self.t, I_est, label="Fitted I", color="blue", linestyle="-")
            ax[1].plot(
                self.t,
                uen_true,
                label="Endogenous Insulin",
                color="orange",
                linestyle="-.",
            )
            ax[1].plot(
                self.t, Uen_est, label="Fitted Uen", color="blue", linestyle="-."
            )
            ax[2].plot(self.t, Q_est, label="Fitted Q", color="blue", linestyle="-")
            # ax[3].plot(t_meas, Q_est, label='Fitted Q', color='blue', linestyle='--')
            ax[3].plot(self.t, P1_est, label="Fitted P1", color="blue", linestyle="-")
            ax[3].plot(self.t, P2_est, label="Fitted P2", color="blue", linestyle="-.")
            ax[4].plot(self.t, SI_est, label="Fitted SI", color="red", linestyle="--")
            err_ax = ax[4].twinx()
            err_ax.spines["right"]  # .set_position(("outward", 60))
            # err_ax.yaxis.set_ticks_position("left")
            # err_ax.yaxis.set_label_position("left")
            err_ax.plot(
                self.t,
                np.abs(
                    (SI_est - model.last_simulation["SI"])
                    * 100
                    / model.last_simulation["SI"]
                ),
                color="purple",
                label="Relative error [%]",
            )
            err_ax.set_ylabel("Relative abs. error [%]")
            err_ax.legend()
            for a in ax:
                a.legend()
            plt.show()
            fig.tight_layout()
