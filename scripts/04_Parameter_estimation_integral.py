import os
import sys

import numpy as np
import tomllib

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import patient_cohort
import utils


def main():
    try:
        SI_ests = np.load("variables/SI_ests_integral_static_04.npy")
        SI_errs = np.load("variables/SI_errs_integral_static_04.npy")
        SI_true_arr = np.load("variables/SI_true_arr_integral_static_04.npy")

        print("Loaded variables from previous run")
    except Exception:
        with open("../config.toml", "rb") as f:
            cfg = tomllib.load(f)
            root_module = "global"
            dt = cfg[root_module]["dt"]
            dtmeas = cfg[root_module]["dtmeas"]
            meas_noise_std = cfg[root_module]["meas_noise_std"]
            uex_bounds = cfg[root_module]["uex_bounds"]
            D_bounds = cfg[root_module]["D_bounds"]
            PN_bounds = cfg[root_module]["PN_bounds"]
            SI_params = cfg[root_module]["SI_params"]
            y0 = cfg[root_module]["y0"]

            curr_module = "04"
            sim_hours = cfg[curr_module]["sim_hours"]
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
                y0=y0,
            )

            patient_data = PatientCohort.patient_data()

            t = np.arange(0, sim_hours * 60 + 1, dt)
            t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
            t_hourly = np.arange(0, sim_hours * 60, 60)
            SI_ests = np.zeros((num_patients, len(t_hourly)))
            SI_errs = np.zeros_like(SI_ests)
            SI_true_arr = np.zeros_like(SI_ests)

            for i in range(num_patients):
                G_meas = patient_data[i]["BG_meas"]
                P_true = patient_data[i]["P"]
                Q_true = patient_data[i]["Q"]
                SI_true = patient_data[i]["SI"][:-1:60]
                SI_true_arr[i,:] = SI_true
                SI_ests[i,:] = utils.integral_approximate_SI_seq(t, t_meas, dtmeas, G_meas, Q_true, P_true)
                SI_errs[i,:] = np.abs(SI_true_arr[i,:] - SI_ests[i,:]) / SI_true_arr[i,:]

            print("Done")
            np.save("variables/SI_ests_integral_static_04", SI_ests)
            np.save("variables/SI_errs_integral_static_04", SI_errs)
            np.save("variables/SI_true_arr_integral_static_04", SI_true_arr)

if __name__ == "__main__":
    main()
