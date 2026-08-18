import os
import sys

import numpy as np
import tomllib

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import patient_cohort
import utils


def main():
    try:
        SI_ests = np.load("variables/SI_ests_02.npy")
        SI_est_error = np.load("variables/SI_est_error_02.npy")
        meas_noise_std = np.load("variables/meas_noise_std_02.npy")
        print("Previous run variables already exist")

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
        curr_module = "02"
        num_patients = cfg[curr_module]["num_patients"]
        sim_hours = cfg[curr_module]["sim_hours"]
        max_std = cfg[curr_module]["max_std"]
        num_std = cfg[curr_module]["num_std"]
        t = np.arange(0, sim_hours * 60 + 1, dt)
        t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
        t_hourly = np.arange(0, sim_hours * 60, 60)
        meas_noise_std = np.linspace(0.0, max_std, num_std)
        num_noise_std = len(meas_noise_std)
        SI_est_error = np.zeros((num_noise_std, num_patients, len(t_hourly)))
        SI_ests = np.zeros_like(SI_est_error)
        change_SI = cfg[curr_module]["change_SI"]
        SI_scale = cfg[curr_module]["SI_scale"]

        for i in range(num_noise_std):
            PatientCohort = patient_cohort.PatientCohort(
                num_patients=num_patients,
                sim_hours=sim_hours,
                dt=dt,
                dtmeas=dtmeas,
                meas_noise_std=meas_noise_std[i],
                uex_bounds=uex_bounds,
                D_bounds=D_bounds,
                PN_bounds=PN_bounds,
                SI_params=SI_params,
                y0=y0,
                process_noise_vars = process_noises,
                SI_piecewise_changes=change_SI,
                SI_scale=SI_scale,
            )
            patient_data = PatientCohort.patient_data()
            for j in range(num_patients):
                G_meas = patient_data[j]["BG_meas"]
                Q = patient_data[j]["Q"]
                P = patient_data[j]["P"]
                SI_true = patient_data[j]["SI"][:-1:60]
                SI_est = utils.integral_approximate_SI_seq(
                    t=t,
                    t_meas=t_meas,
                    Ts_meas=dtmeas,
                    G_meas=G_meas,
                    Q_true=Q,
                    P_true=P,
                )
                SI_ests[i, j, :] = SI_est
                SI_est_error[i, j, :] = (SI_true - SI_est)**2

        print("Done")
        np.save("variables/SI_ests_02", SI_ests)
        np.save("variables/SI_est_error_02", SI_est_error)
        np.save("variables/meas_noise_std_02", meas_noise_std)


if __name__ == "__main__":
    main()
