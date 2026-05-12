import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import patient_cohort


def main():
    try:
        x0 = np.load("variables/x0_00.npy")
        inputs = np.load("variables/inputs_00.npy")

    except Exception:
        num_patients = None
        sim_hours = 24
        dt = 1.0
        dtmeas = 120
        meas_noise_std = None
        uex_bounds = [0, 166]
        D_bounds = [0.2, 0.4]
        PN_bounds = [0.2, 0.4]
        SI_params = np.asarray([1e-5, 0.0001, 0.00045])
        y0 = [7.5, 150.0, 75.0, 6, 40]
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
        num_simulations = 100000
        x0 = np.zeros((len(y0), num_simulations))
        inputs = np.zeros((4, num_simulations))  # uex, D, PN, SI

        for i in range(num_simulations):
            x0[:, i], inputs[:, i] = PatientCohort.initial_states()

        np.save("variables/x0_00", x0)
        np.save("variables/inputs_00", inputs)


if __name__ == "__main__":
    main()
