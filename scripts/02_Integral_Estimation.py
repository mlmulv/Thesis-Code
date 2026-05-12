import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import patient_cohort
import utils


def main():
    try:
        SI_ests = np.load("variables/SI_ests_01.npy")
        SI_est_error = np.load("variables/SI_est_error_01.npy")
        meas_noise_std = np.load("variables/meas_noise_std_01.npy")
        print("Previous run variables already exist")

    except Exception:
        num_patients = 250
        sim_hours = 24
        dt = 1
        dtmeas = 120
        t = np.arange(0, sim_hours * 60 + 1, dt)
        t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
        t_hourly = np.arange(0, sim_hours * 60, 60)
        meas_noise_std = np.linspace(0.0, 1.0, 5)
        num_noise_std = len(meas_noise_std)
        uex_bounds = [0, 166]
        D_bounds = [0.2, 0.4]
        PN_bounds = [0.2, 0.4]
        SI_params = np.asarray([1e-5, 0.0001, 0.00045])
        y0 = [7.5, 150.0, 75.0, 6, 40]
        SI_est_error = np.zeros((num_noise_std, num_patients, len(t_hourly)))
        SI_ests = np.zeros_like(SI_est_error)

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
                SI_est_error[i, j, :] = np.abs((SI_est - SI_true) * 100 / SI_true)

        print("Done")
        np.save("variables/SI_ests_01", SI_ests)
        np.save("variables/SI_est_error_01", SI_est_error)
        np.save("variables/meas_noise_std_01", meas_noise_std)


if __name__ == "__main__":
    main()
