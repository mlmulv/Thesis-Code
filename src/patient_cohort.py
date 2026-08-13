import os
import sys
from pathlib import Path

import numpy as np
import scipy as sp

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import icing_true
import utils

rng = np.random.default_rng()
base_dir = Path(__file__).resolve().parent

class PatientCohort:
    def __init__(
        self,
        num_patients,
        sim_hours,
        dt,
        dtmeas,
        meas_noise_std,
        uex_bounds,
        D_bounds,
        PN_bounds,
        SI_params,
        y0,
        exp_uex=False,
        SI_piecewise_changes=None,
        SI_scale=None,
    ):
        # Inputs
        self.num_patients = num_patients
        self.sim_hours = sim_hours
        self.dt = dt  # simulation resolution in hours
        self.dtmeas = dtmeas  # measurement resolution in hours
        self.meas_noise_std = meas_noise_std
        self.uex_bounds = uex_bounds  # [low, mean, std]
        self.D_bounds = D_bounds  # [low, high]
        self.PN_bounds = PN_bounds  # [low, high]
        self.SI_params = SI_params  # [mean, std] in log domain
        self.y0 = y0
        self.SI_piecewise_changes = SI_piecewise_changes  # number of changes
        self.SI_scale = SI_scale

        # Assigned variables
        self.t = np.arange(0, sim_hours * 60 + 1, dt)
        self.t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
        self.sample_indices = [np.argmin(np.abs(self.t - st)) for st in self.t_meas]

        try:
            self.insulin_histogram = np.load(base_dir / "insulin_histogram.npz")
            self.feeding_histogram = np.load(base_dir / "feeding_histogram.npz")
        except Exception:
            print("GLUC-ICU input histograms not in saved_variables directory")

    # def draw_inputs(self):
    #     uex_const = rng.uniform(low=self.uex_bounds[0], high=self.uex_bounds[1])
    #     D_const = rng.uniform(low=self.D_bounds[0], high=self.D_bounds[1])
    #     PN_const = rng.uniform(low=self.PN_bounds[0], high=self.PN_bounds[1])
    #     SI_const = np.exp(
    #         sp.stats.norm.rvs(loc=self.SI_params[0], scale=self.SI_params[1])
    #     )
    #     return uex_const, D_const, PN_const, SI_const

    def draw_KI_inputs(self):
        insulin_height = self.insulin_histogram["insulin_heights"]
        insulin_edges = self.insulin_histogram["insulin_edges"]
        feeding_height = self.feeding_histogram["feeding_heights"]
        feeding_edges = self.feeding_histogram["feeding_edges"]
        uex_const = utils.sample_hist(
            heights=insulin_height, edges=insulin_edges, num_samples=1, ndims=1
        )
        D_const, PN_const = utils.sample_hist(
            heights=feeding_height, edges=feeding_edges, num_samples=1, ndims=2
        )
        SI_const = np.exp(
            sp.stats.norm.rvs(loc=self.SI_params[0], scale=self.SI_params[1])
        )
        return uex_const, D_const, PN_const, SI_const

    def initial_states(self):
        uex_const, D_const, PN_const, SI_const = self.draw_KI_inputs()
        uex_func = utils.gen_uex_func(uex_const, )
        D_func = utils.gen_D_func(D_const)
        PN_func = utils.gen_PN_func(PN_const)
        SI_func = utils.gen_SI_func(SI_const)
        ICINGTrue = icing_true.ICINGTrue()
        t = 0

        def f(x):
            return ICINGTrue.icing_odes(t, x, uex_func, PN_func, D_func, SI_func)

        x0 = sp.optimize.fsolve(f, self.y0)

        return x0, [uex_const, D_const, PN_const, SI_const]

    def patient_data(self):
        t_start = self.t[0]
        t_end = self.t[-1]
        data = []
        for i in range(self.num_patients):
            x0, input_consts = self.initial_states()
            uex_const = input_consts[0]
            PN_const = input_consts[1]
            D_const = input_consts[2]
            SI_const = input_consts[3]
            if self.SI_piecewise_changes is not None:
                SI_next_const = SI_const.copy()
                Uniform = sp.stats.Uniform(a=self.SI_scale[0], b=self.SI_scale[1])
                Uniform_sign = sp.stats.Uniform(a=0, b=1)
                for i in range(self.SI_piecewise_changes):
                    SI_curr_const = SI_next_const.copy()
                    sign = 1 if Uniform_sign.sample(shape=(1,)) <= 0.5 else -1
                    SI_next_const = SI_curr_const * (
                        1 + sign * Uniform.sample(shape=(1,))
                    )
                    # clip SI
                    SI_next_const = np.clip(SI_next_const, 2.7e-5, 2.5e-3)
                    SI_const = np.append(SI_const, SI_next_const)

                shift = int(len(self.t) / self.SI_piecewise_changes)
                SI_values = np.ones_like(self.t)

                for i in range(self.SI_piecewise_changes):
                    next_idx = i + 1
                    if i != self.SI_piecewise_changes - 1:
                        SI_values[i * shift : (next_idx * shift)] = SI_const[i]
                    else:
                        SI_values[i * shift : (next_idx * shift) + 1] = SI_const[i]

                SI_func = utils.piecewise_constant_to_callable(SI_values, self.t)
            else:
                SI_func = utils.gen_SI_func(SI_const=SI_const)

            uex_func = utils.gen_uex_func(uex_const=uex_const)
            D_func = utils.gen_D_func(D_const=D_const)
            PN_func = utils.gen_PN_func(PN_const=PN_const)
            ICINGTrue = icing_true.ICINGTrue()
            BG, I, Q, P1, P2, P = ICINGTrue.simulate(  # noqa: E741
                x0, t_start, t_end, self.t, uex_func, PN_func, D_func, SI_func
            )
            BG_meas = BG[self.sample_indices] + rng.normal(
                0.0, self.meas_noise_std, size=len(self.sample_indices)
            )
            uen = ICINGTrue.params["k1"] * np.exp(
                -(ICINGTrue.params["k2"] / ICINGTrue.params["k3"]) * I
            )
            patient_info = {
                "patient_id": f"Patient_{i}",
                "x0": x0,
                "BG": BG,
                "BG_meas": BG_meas,
                "Q": Q,
                "I": I,
                "P1": P1,
                "P2": P2,
                "P": P,
                "uen": uen,
                "sim_hours": self.sim_hours,
                "t": self.t,
                "t_meas": self.t_meas,
                "meas_noise_std": self.meas_noise_std,
                "uex": np.asarray([uex_func(t) for t in self.t]),
                "D": np.asarray([D_func(t) for t in self.t]),
                "PN": np.asarray([PN_func(t) for t in self.t]),
                "SI": np.asarray([SI_func(t) for t in self.t]),
            }
            data.append(patient_info)
        return data
