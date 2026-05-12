import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import patient_cohort


def main():
    try:
        np.load("variables/patient_data_00")
    except Exception:
        num_patients = 250
        sim_hours = 60
        dt = 1
        dtmeas = 120
        meas_noise_std = 0.25
        uex_bounds = [0, 0.6]
        D_bounds = [0.1, 0.3]
        PN_bounds = [0.1, 0.3]
        SI_params = [0, 0.5e-4, 6e-4]
        y0 = [7.5, 15.0, 15.0, 0.5, 0.5]

        PatientCohort = patient_cohort.PatientCohort(
            num_patients=num_patients,
            sim_hours=sim_hours,
            dt=dt,
            dtmeas=dtmeas,
            meas_noise_std=meas_noise_std,
            uex_bounds=uex_bounds,
            D_bounds=D_bounds,
            PN_bounds=PN_bounds,
            SI_params=SI_params,
            y0=y0,
        )

        patient_data = PatientCohort.patient_data()
        np.save("variables/patient_data_00", patient_data)


if __name__ == "__main__":
    main()
