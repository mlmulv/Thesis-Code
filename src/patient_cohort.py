from icing_true import ICINGTrue
import numpy as np
rng = np.random.default_rng(seed=42)

class patient_cohort:
    def __init__(self, num_patients, sim_hours, dt, dtmeas, BG_params):
        # Inputs
        self.num_patients = num_patients
        self.sim_hours = sim_hours
        self.dt = dt # simulation resolution in hours
        self.dtmeas = dtmeas # measurement resolution in hours
        self.BG_params = BG_params # mean and std for BG norm dist

        # Assigned variables
        self.t = np.arange(0, sim_hours*60+1, dt)
        self.t_meas = np.arange(0, sim_hours*60+1, dtmeas)
        self.initial_Q = 15
        self.initial_I = 15
        self.initial_P1 = 0
        self.initial_P2 = 0

    def gen_initial_BG(self):
        mu = np.log(self.BG_params[0]**2 / np.sqrt(self.BG_params[1]**2 + self.BG_params[0]**2))
        sigma = np.sqrt(np.log(1 + (self.BG_params[1]**2 / self.BG_params[0]**2)))
        return rng.lognormal(mean=mu, sigma=sigma, size=self.num_patients)

    def initial_states(self):
        initial_BG = self.gen_initial_BG()
        return np.asarray([[BG, self.initial_Q, self.initial_I, self.initial_P1, self.initial_P2] for BG in initial_BG]).reshape(-1,5)

    def patient_data(self, uex_func, PN_func, D_func, SI_func):
        init_states = self.initial_states()
        icing_model = ICINGTrue()
        t_start = self.t[0]
        t_end = self.t[-1]
        data = []
        for i in range(self.num_patients):
            y0 = init_states[i,:]
            BG, Q, I, P1, P2, P = icing_model.simulate(y0, t_start, t_end, self.t, uex_func, PN_func, D_func, SI_func)
            patient_info = {
                'patient_id': f'Patient_{i}',
                'y0': y0,
                'BG': BG,
                'Q': Q,
                'I': I,
                'P1': P1,
                'P2': P2,
                'P': P
            }
            data.append(patient_info)
        return data

