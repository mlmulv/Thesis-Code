import icing_true
import numpy as np
rng = np.random.default_rng()
import scipy as sp
import utils

class PatientCohort:
    def __init__(self, num_patients, sim_hours, dt, dtmeas, meas_noise_std, uex_bounds, D_bounds, PN_bounds, SI_params, y0, SI_piecewise=False):
        # Inputs
        self.num_patients = num_patients
        self.sim_hours = sim_hours
        self.dt = dt # simulation resolution in hours
        self.dtmeas = dtmeas # measurement resolution in hours
        self.meas_noise_std = meas_noise_std
        self.uex_bounds = uex_bounds # [low, high]
        self.D_bounds = D_bounds # [low, high]
        self.PN_bounds = PN_bounds # [low, high]
        self.SI_params = SI_params # [low, mean, std]
        self.y0 = y0

        # Assigned variables
        self.t = np.arange(0, sim_hours*60+1, dt)
        self.t_meas = np.arange(0, sim_hours*60+1, dtmeas)
        self.sample_indices = [np.argmin(np.abs(self.t - st)) for st in self.t_meas]

    def draw_inputs(self):
        uex_const = rng.uniform(low=self.uex_bounds[0], high=self.uex_bounds[1])
        D_const = rng.uniform(low=self.D_bounds[0], high=self.D_bounds[1])
        PN_const = rng.uniform(low=self.PN_bounds[0], high=self.PN_bounds[1])
        a = (self.SI_params[0] - self.SI_params[1]) / self.SI_params[2]
        b = float('inf')
        SI_const = sp.stats.truncnorm.rvs(a,b,loc=self.SI_params[1], scale=self.SI_params[2])
        return uex_const, D_const, PN_const, SI_const

    def initial_states(self):
        uex_const, D_const, PN_const, SI_const = self.draw_inputs()
        uex_func = utils.gen_uex_func(uex_const)
        D_func = utils.gen_D_func(D_const)
        PN_func = utils.gen_PN_func(PN_const)
        SI_func = utils.gen_SI_func(SI_const)
        ICINGTrue = icing_true.ICINGTrue()        
        t = 0
        x0 = sp.optimize.fsolve(ICINGTrue.icing_odes, self.y0, args=(t, uex_func, D_func, PN_func, SI_func))
        return x0, [uex_const, D_const, PN_const, SI_const]

    def patient_data(self, uex_func, PN_func, D_func, SI_func):
        init_states = self.initial_states()
        # icing_model = icing_true.ICINGTrue()
        # t_start = self.t[0]
        # t_end = self.t[-1]
        # data = []
        # for i in range(self.num_patients):
        #     y0 = init_states[i,:]
        #     BG, Q, I, P1, P2, P = icing_model.simulate(y0, t_start, t_end, self.t, uex_func, PN_func, D_func, SI_func)
        #     BG_meas = BG[self.sample_indices] + rng.normal(0.0, self.meas_noise_std, size=len(self.sample_indices))
        #     patient_info = {
        #         'patient_id': f'Patient_{i}',
        #         'y0': y0,
        #         'BG': BG,
        #         'BG_meas' : BG_meas,
        #         'Q': Q,
        #         'I': I,
        #         'P1': P1,
        #         'P2': P2,
        #         'P': P
        #     }
        #     data.append(patient_info)
        # return data

