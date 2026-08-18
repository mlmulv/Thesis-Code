import os
import pickle as pkl
import sys

import tomllib

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import numpy as np

import patient_cohort

def main():
    try:
        with open("variables/patient_data_00.pkl", "rb") as f:
            patient_data = pkl.load(f)
        print("patient_data_00 variable already exists")

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
        process_noise_factor = cfg[root_module]["process_noise_factor"]
        process_noises = np.asarray(
            [
                (0.008) * process_noise_factor,
                (0.027) * process_noise_factor,
                (0.086) * process_noise_factor,
                (0.001) * process_noise_factor,
                (0.034) * process_noise_factor,
                (1e-8) * process_noise_factor,
            ]
        )

        curr_module = "00"
        num_patients = cfg[curr_module]["num_patients"]

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
            process_noise_vars=process_noises,
            y0=y0,
        )

        patient_data = PatientCohort.patient_data()
        pkl.dump(patient_data, open("variables/patient_data_00.pkl", "wb"))


if __name__ == "__main__":
    main()
