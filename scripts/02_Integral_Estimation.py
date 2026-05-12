import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import icing_true
import patient_cohort
import utils


def main():
    try:
        SI_ests = np.load("variables/SI_ests_01.npy")
        SI_est_error = np.load("variables/SI_est_error_01.npy")
        meas_noise_std = np.load("variables/meas_noise_std_01.npy")
        SI_true = np.load("variables/SI_true_01.npy")
        print("Previous run variables already exist")

    except Exception:
        BG_logmu = 7.6
        BG_logsigma = 1.3
        num_patients = 100
        sim_hours = 60
        dt = 1
        dtmeas = 120
        t = np.arange(0, sim_hours * 60 + 1, dt)
        t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
        t_hourly = np.arange(0, sim_hours * 60, 60)
        meas_noise_std = np.linspace(0.0, 1.0, 5)
        num_noise_std = len(meas_noise_std)
        uex_func = utils.gen_uex_func()
        PN_func = utils.gen_PN_func()
        D_func = utils.gen_D_func()
        SI_func = utils.gen_SI_func()
        SI_true = np.asarray([SI_func(ts) for ts in t_hourly])
        pG = icing_true.ICINGTrue().params["pG"]
        alphaG = icing_true.ICINGTrue().params["alphaG"]
        EGP = icing_true.ICINGTrue().params["EGP"]
        CNS = icing_true.ICINGTrue().params["CNS"]
        VG = icing_true.ICINGTrue().params["VG"]

        SI_est_error = np.zeros((num_noise_std, num_patients, len(t_hourly)))
        SI_ests = np.zeros_like(SI_est_error)

        for i in range(num_noise_std):
            PatientCohort = patient_cohort.PatientCohort(
                num_patients=num_patients,
                sim_hours=sim_hours,
                dt=dt,
                dtmeas=dtmeas,
                meas_noise_std=meas_noise_std[i],
                BG_params=[BG_logmu, BG_logsigma],
            )
            patient_data = PatientCohort.patient_data(
                uex_func=uex_func, PN_func=PN_func, D_func=D_func, SI_func=SI_func
            )

            for j in range(num_patients):
                G_meas = patient_data[j]["BG_meas"]
                Q = patient_data[j]["Q"]
                P = patient_data[j]["P"]
                SI_est = utils.integral_approximate_SI_seq(
                    pG=pG,
                    alphaG=alphaG,
                    EGP=EGP,
                    CNS=CNS,
                    VG=VG,
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
        np.save("variables/SI_true_01", SI_true)


if __name__ == "__main__":
    main()
