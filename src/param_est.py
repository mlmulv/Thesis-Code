from ext_kalman_filter import ExtendedKalmanFilter
import numpy as np
import scipy as sp

class MCMCKalman():
    def __init__(self, prop_std):
        self.prop_std = prop_std

    def update_candidate(self, prev_candidate):
        return np.exp(np.log(prev_candidate) + np.random.normal(scale=self.prop_std))

    def energy_func(self, prev_energy, S, v): 
        return prev_energy + 0.5 * (np.log(2.0 * np.pi * S) + (v * v) / S)

    def acceptance(self, prev_energy, energy):
        threshold = min(1.0, np.exp(prev_energy - energy))
        u = np.random.uniform(low=0.0, high=1.0)
        if u <= threshold:
            return True
        else:
            return False

    class simulate():
        def __init__(self, SI_init, sigma_SI, Estimator, num_states, Model, t, t_meas, ts_meas, G_meas):
            self.SI_init = SI_init
            self.sigma_SI = sigma_SI
            self.Estimator = Estimator
            self.num_states = num_states
            self.Model = Model
            self.t = t
            self.t_meas = t_meas
            self.ts_meas = ts_meas
            self.G_meas = G_meas

        def run(self):
            candidate = np.exp(self.SI_init)
            prev_energy = - sp.stats.lognorm.logpdf(candidate, s=self.sigma_SI, scale=candidate)
            EKF = ExtendedKalmanFilter(self.num_states, self.Model, self.ts_meas)
            state, var = EKF.initialize_filter()
            SI_est = np.zeros((len(self.t_meas),))
            candidate_energy = np.zeros_like(SI_est)
            num_acceptance = 0

            for ti in self.t: 
                if ti in self.t_meas:
                    sample_idx = np.argmin(np.abs(self.t_meas - ti))
                    potential_candidate = self.Estimator.update_candidate(candidate)
                    state, var = EKF.filter_predict(state, var, int(ti), potential_candidate)
                    state, var, S, v = EKF.filter_update(state, var, self.G_meas[sample_idx], output_params=True)
                    energy = self.Estimator.energy_func(prev_energy, S[0][0], v[0][0])
                    if self.Estimator.acceptance(prev_energy, energy):
                        candidate = potential_candidate
                        prev_energy = energy
                        candidate_energy[sample_idx] = energy
                        num_acceptance += 1
                    else:
                        candidate = candidate
                        candidate_energy[sample_idx] = prev_energy
                    SI_est[sample_idx] = candidate
                else:
                    state, var = EKF.filter_predict(state, var, int(ti), candidate)

            return SI_est, candidate_energy, num_acceptance / len(self.t_meas)

                






