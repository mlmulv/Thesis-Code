import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import tomllib

import gramian_obs
import patient_cohort
import utils


def main():
    try:
        diag_values = np.load("variables/diag_values_01.npy")
        eig_values = np.load("variables/eig_values_01.npy")
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
        x_max = cfg[root_module]["x_max"]
        process_noise_factor = cfg[root_module]["process_noise_factor"]
        process_noises = np.asarray(
            [
                (0.008), 
                (0.146),
                (0.314),
                (0.007),
                (0.174),
                (1e-8),
            ]
        )


        curr_module = "01"
        n_pert = cfg[curr_module]["n_pert"]
        c = cfg[curr_module]["c"]
        threshold = cfg[curr_module]["threshold"]
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
            SI_piecewise_changes=None,
            process_noise_vars = process_noises
        )

        patient_data = PatientCohort.patient_data()

        t = np.arange(
            0, sim_hours * 60 + 1, dt
        )  # 0 to 60 hours with minute-level resolution
        eig_values = np.zeros((num_patients,6))
        diag_values = np.zeros_like(eig_values)

        time_start = time.time()
        for i in range(num_patients):
            if (i + 1) % 1000 == 0:
                print(f"Patient {i + 1} / {num_patients}")
                print(f"{(time.time() - time_start) / 60:.3f} mins have elapsed")

            x0 = patient_data[i]["x0"]
            uex_const = patient_data[i]["uex"][0]
            D_const = patient_data[i]["D"][0]
            PN_const = patient_data[i]["PN"][0]
            SI_const = patient_data[i]["SI"][0]
            uex_func = utils.gen_uex_func(uex_const=uex_const)
            PN_func = utils.gen_PN_func(PN_const=PN_const)
            D_func = utils.gen_D_func(D_const=D_const)
            SI_func = utils.gen_SI_func(SI_const=SI_const)

            Gramian = gramian_obs.EmperialGramianMatrix(
                x0=x0,
                n_pert=n_pert,
                c=c,
                x_max=x_max,
                t=t,
                dt=dt,
                uex_func=uex_func,
                PN_func=PN_func,
                D_func=D_func,
                SI_func=SI_func,
                threshold=threshold,
                process_noise_vars=process_noises,
                measurement_noise_var=meas_noise_std**2,
                SI_est=None,
            )
            W_O, eig_vals, eig_vectors = Gramian.gramian()
            eig_values[i,:] = eig_vals
            diag_values[i,:] = np.diag(W_O)

        print("Done!")
        np.save("variables/diag_values_01", diag_values)
        np.save("variables/eig_values_01", eig_values)


if __name__ == "__main__":
    main()
