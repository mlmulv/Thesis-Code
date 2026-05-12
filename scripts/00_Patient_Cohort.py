import os
import pickle as pkl
import sys

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import patient_cohort


def main():
    try:
        patient_data = pkl.load(open("variables/patient_data_00.pkl", "rb"))
        print("patient_data_00 variable already exists")

    except Exception:
        num_patients = 5
        sim_hours = 24
        dt = 1.0
        dtmeas = 120
        meas_noise_std = 0.25
        uex_bounds = [0, 166]
        D_bounds = [0.2, 0.4]
        PN_bounds = [0.2, 0.4]
        SI_params = [1e-5, 0.0001, 0.00045]
        y0 = [7.5, 150.0, 75.0, 6, 40]

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
        pkl.dump(patient_data, open("variables/patient_data_00.pkl", "wb"))


if __name__ == "__main__":
    main()
