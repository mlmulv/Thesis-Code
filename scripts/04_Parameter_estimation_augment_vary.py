import os
import sys
import time

import numpy as np
import tomllib

sys.path.append(os.path.abspath(os.path.join("..", "src")))

import patient_cohort
import utils


def main():

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
        x_max = cfg[root_module]["x_max"]
        u_en_max = cfg[root_module]["u_en_max"]
        log_sigma_SI = cfg[root_module]["log_sigma_SI"]

        curr_module = "04"
        sim_hours = 2 * cfg[curr_module]["sim_hours"]
        num_patients = cfg[curr_module]["num_patients"]
        process_noise_factor = cfg[curr_module]["process_noise_factor"]
        num_particles = np.asarray(cfg[curr_module]["num_particles"])
        deviation = cfg[curr_module]["deviation"]
        change_SI = cfg[curr_module]["change_SI"]

        t = np.arange(0, sim_hours * 60 + 1, dt)
        t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)

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
            SI_piecewise_changes=change_SI,
        )

        patient_data = PatientCohort.patient_data()


if __name__ == "__main__":
    main()
