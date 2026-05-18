import os
import sys

import numpy as np
import tomllib

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import patient_cohort
import icing_true


def main():
    try:
        x0 = np.load("variables/x0_00.npy")
        inputs = np.load("variables/inputs_00.npy")
        print("Previous run variables exist")
    except Exception:
        with open("../config.toml", "rb") as f:
            cfg = tomllib.load(f)

        root_module = "global"
        sim_hours = cfg[root_module]["sim_hours"]
        dt = cfg[root_module]["dt"]
        dtmeas = cfg[root_module]["dtmeas"]
        meas_noise_std = cfg[root_module]["meas_noise_std"]
        uex_bounds = cfg[root_module]["uex_bounds"]
        D_bounds = cfg[root_module]["D_bounds"]
        PN_bounds = cfg[root_module]["PN_bounds"]
        SI_params = cfg[root_module]["SI_params"]
        y0 = cfg[root_module]["y0"]

        curr_module = "00"
        num_patients = cfg[curr_module]["num_patients"]
        num_simulations = cfg[curr_module]["num_simulations"]

        PatientCohort = patient_cohort.PatientCohort(
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
        )

        ICINGTrue = icing_true.ICINGTrue()

        x0 = np.zeros((len(y0)+1, num_simulations))
        inputs = np.zeros((4, num_simulations))  # uex, D, PN, SI

        for i in range(num_simulations):
            x0[:-1, i], inputs[:, i] = PatientCohort.initial_states()
            x0[-1, i] = ICINGTrue.params["k1"] * np.exp(-(ICINGTrue.params["k2"] / ICINGTrue.params["k3"]) * x0[1,i])

        np.save("variables/x0_00", x0)
        np.save("variables/inputs_00", inputs)


if __name__ == "__main__":
    main()
