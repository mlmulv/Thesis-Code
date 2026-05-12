from copy import deepcopy

import numpy as np
import scipy as sp

from ext_kalman_filter import ExtendedKalmanFilter


class MCMCKalman:
    def __init__(self, mu_SI, sigma_SI, prop_std):
        self.mu_SI = mu_SI
        self.sigma_SI = sigma_SI
        self.prop_std = prop_std

    def prior(self, candidate):
        return sp.stats.lognorm.logpdf(candidate, s=self.sigma_SI, scale=self.mu_SI)

    def prop_candidate(self, candidate_curr):
        return np.exp(np.log(candidate_curr) + np.random.normal(scale=self.prop_std))

    def energy_func(self, S, v):
        return 0.5 * (np.log(2.0 * np.pi * S) + (v * v) / S)

    def acceptance(self, energy_curr, energy_prop):
        return np.random.uniform() <= min(1.0, np.exp(energy_curr - energy_prop))

    class simulate:
        def __init__(
            self, SI_init, Estimator, num_states, Model, t, t_meas, ts_meas, G_meas
        ):
            self.SI_init = SI_init
            self.Estimator = Estimator
            self.num_states = num_states
            self.Model = Model
            self.t = t
            self.t_meas = t_meas
            self.ts_meas = ts_meas
            self.G_meas = G_meas

        def EKF_iteration(self, candidate, state_init, var_init, meas_time, EKF):
            state_predict = deepcopy(state_init)
            var_predict = deepcopy(var_init)
            prior = -self.Estimator.prior(candidate)
            energy = deepcopy(prior)
            t_K = self.t[self.t <= meas_time]

            for ti in t_K:
                if ti in self.t_meas:
                    sample_idx = np.argmin(np.abs(self.t_meas - ti))
                    state_predict, var_predict = EKF.filter_predict(
                        state_predict, var_predict, int(ti), candidate
                    )
                    state_update, var_update, S, v = EKF.filter_update(
                        state_predict,
                        var_predict,
                        self.G_meas[sample_idx],
                        output_params=True,
                    )
                    energy += self.Estimator.energy_func(S[0][0], v[0][0])
                    state_predict = deepcopy(state_update)
                    var_predict = deepcopy(var_update)
                else:
                    state_predict, var_predict = EKF.filter_predict(
                        state_predict, var_predict, int(ti), candidate
                    )

            return energy

        def run(self):
            candidate = np.exp(self.SI_init)
            EKF = ExtendedKalmanFilter(self.num_states, self.Model, self.ts_meas)
            state_init, var_init = EKF.initialize_filter()

            SI_est = []
            num_acceptance = 0
            candidate_curr = candidate

            for meas_time in self.t_meas:
                candidate_prop = self.Estimator.prop_candidate(candidate_curr)
                energy_curr = self.EKF_iteration(
                    candidate_curr, state_init, var_init, meas_time, EKF
                )
                energy_prop = self.EKF_iteration(
                    candidate_prop, state_init, var_init, meas_time, EKF
                )

                if self.Estimator.acceptance(energy_curr, energy_prop):
                    candidate_curr = candidate_prop
                    num_acceptance += 1

                SI_est.append(candidate_curr)

            acceptance_rate = num_acceptance / len(self.t_meas)
            return np.asarray(SI_est), acceptance_rate
