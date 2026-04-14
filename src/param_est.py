from ext_kalman_filter import ExtendedKalmanFilter
import numpy as np
import scipy as sp

class MCMC():
    def __init__(self, prop_var):
        self.prop_var = prop_var

    def update_candidate(self, prev_candidate):
        return np.exp(np.random.normal(loc=np.log(prev_candidate), scale=self.prop_var))

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
        def __init__(self, init_SI, sigma_SI, Estimator, num_states, Model, state_augment, t_meas, ts_meas, G_meas):
            self.init_SI = init_SI
            self.sigma_SI = sigma_SI
            self.Estimator = Estimator
            self.num_states = num_states
            self.Model = Model
            self.state_augment = state_augment
            self.t_meas = t_meas
            self.ts_meas = ts_meas
            self.G_meas = G_meas

        def run(self):
            candidate = np.exp(self.init_SI)
            prev_energy = - np.log(sp.stats.lognorm.pdf(candidate, s=self.sigma_SI, scale=candidate))
            EKF = ExtendedKalmanFilter(self.num_states, self.Model, self.ts_meas, state_augment=False)
            state, var = EKF.initialize_filter()
            SI_est = np.zeros((len(self.t_meas),))
            candidate_energy = np.zeros_like(SI_est)
           
            for idx, y in enumerate(self.G_meas):
                potential_candidate = self.Estimator.update_candidate(candidate)
                state, var = EKF.filter_predict(state, var, self.t_meas[idx], potential_candidate)
                state, var, S, v = EKF.filter_update(state, var, y)
                energy = self.Estimator.energy_func(prev_energy, S[0][0], v[0][0])
                if self.Estimator.acceptance(prev_energy, energy):
                    candidate = potential_candidate
                    prev_energy = energy
                    candidate_energy[idx] = energy
                else:
                    candidate = candidate
                    candidate_energy[idx] = prev_energy
                SI_est[idx] = candidate
            return SI_est, candidate_energy
                






