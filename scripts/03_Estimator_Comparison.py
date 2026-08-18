import os
import sys
import time
import tomllib

import numpy as np

sys.path.append(os.path.abspath(os.path.join("..", "src")))
import warnings
from multiprocessing import Pool, cpu_count

import ext_kalman_filter
import icing_model
import icing_true
import particle_filter
import patient_cohort
import utils

warnings.filterwarnings("error", category=RuntimeWarning)


def worker(task):
    (
        filter_type,
        num_particles,
        num_states,
        dt,
        dtmeas,
        t,
        t_meas,
        G_meas,
        initial_state,
        process_noise_vars,
        meas_noise_std,
        uex_const,
        PN_const,
        D_const,
        SI_const,
    ) = task
    uex_func = utils.gen_uex_func(uex_const)
    PN_func = utils.gen_PN_func(PN_const)
    D_func = utils.gen_D_func(D_const)
    input_functions = {"D": D_func, "PN": PN_func, "Uex": uex_func}
    model = icing_model.ICINGModel(
        params=None,
        dt=dt,
        initial_state=initial_state,
        process_noise_vars=process_noise_vars,
        measurement_noise_var=meas_noise_std**2,
        u_funcs=input_functions,
        SI_augment=False,
    )

    if filter_type == "PF":
        PF_filter = particle_filter.BootstrapParticleFilter(
            num_particles=num_particles,
            num_states=num_states,
            model=model,
            Ts_meas=dtmeas,
        )
        sim = PF_filter.simulate(t, t_meas, G_meas, PF_filter, SI_fixed=SI_const)
        timeStart = time.time()
        _, _, saved_particles, saved_weights, _, _, _, _, _, _, _, _ = sim.run()
        timeEnd = time.time()
        G_est = np.average(saved_particles[:, :, 0], weights=saved_weights, axis=1)
        weights = np.repeat(saved_weights[:, :, np.newaxis], 6, axis=2) # 6 is num_states
        states = np.average(saved_particles, weights=weights, axis=1)
        difference = saved_particles - states[:, np.newaxis, :]
        outer = difference[:, :, np.newaxis, :] * difference[:, :, :, np.newaxis]
        weights = np.repeat(weights[:, :, :, np.newaxis], 6, axis=3)
        all_vars = np.average(
            (outer) ** 2,
            weights=weights,
            axis=1,
        )
        time_elapsed = timeEnd - timeStart
    else:
        EKF_filter = ext_kalman_filter.ExtendedKalmanFilter(
            num_states=num_states, model=model, ts_meas=dtmeas
        )
        sim = EKF_filter.simulate(t, t_meas, G_meas, EKF_filter, SI_fixed=SI_const)
        timeStart = time.time()
        saved_state, saved_noise_var, _, _, _, _, _, _, _, _ = sim.run()
        timeEnd = time.time()
        G_est = saved_state[:, 0]
        states = saved_state
        all_vars = saved_noise_var
        time_elapsed = timeEnd - timeStart

    return time_elapsed, G_est, states, all_vars


def main():
    try:
        error_est = np.load("variables/error_arr_03_temp.npy")
        time_train = np.load("variables/time_arr_03_temp.npy")
        G_ests = np.load("variables/G_ests_03_temp.npy")
        G_trues = np.load("variables/G_trues_03_temp.npy")
        states = np.load("variables/states_03_temp.npy")
        all_vars = np.load("variables/vars_03_temp.npy")
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
        curr_module = "03"
        num_patients = cfg[curr_module]["num_patients"]
        sim_hours = cfg[curr_module]["sim_hours"]
        process_noise_factor = cfg[curr_module]["process_noise_factor"]
        num_particles = np.asarray(cfg[curr_module]["num_particles"])

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
            process_noise_vars = process_noises
        )

        patient_data = PatientCohort.patient_data()

        ICINGTrue = icing_true.ICINGTrue()

        num_filters = (
            len(num_particles) + 1
        )  # 1 EKF filter and the number of variations of PF with the assigned number of particles
        t = np.arange(0, sim_hours * 60 + 1, dt)
        t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
        time_train = np.zeros(shape=(num_filters, num_patients))
        error_est = np.zeros(shape=(num_filters, num_patients, len(t)))
        G_ests = np.zeros_like(error_est)
        G_trues = np.zeros((num_patients, len(t)))
        states = np.zeros((num_filters, num_patients, len(t), 6))
        all_vars = np.zeros(num_filters, num_patients, len(t), 6, 6)

        time_start = time.time()
        for i in range(num_patients):
            print(f"Patient {i + 1} / {num_patients}")
            G_true = patient_data[i]["BG"]
            G_trues[i,:] = G_true
            G_meas = patient_data[i]["BG_meas"]
            uex_const = patient_data[i]["uex"][0]
            SI_const = patient_data[i]["SI"][0]
            D_const = patient_data[i]["D"][0]
            PN_const = patient_data[i]["PN"][0]
            G_0 = patient_data[i]["BG"][0]
            Q_0 = patient_data[i]["Q"][0]
            I_0 = patient_data[i]["I"][0]
            P1_0 = patient_data[i]["P1"][0]
            P2_0 = patient_data[i]["P2"][0]
            y0 = [G_0, I_0, Q_0, P1_0, P2_0]
            initial_state = np.append(
                y0,
                [
                    ICINGTrue.params["k1"]
                    * np.exp(
                        -y0[2] * ICINGTrue.params["k2"] / ICINGTrue.params["k3"]
                    )
                ],
            )
            initial_state = np.expand_dims(initial_state, axis=1)
            num_states = len(initial_state)

            tasks = []
            for j in range(num_filters):
                if j != 0:  # PF
                    tasks.append(
                        (
                            "PF",
                            num_particles[j - 1],
                            num_states,
                            dt,
                            dtmeas,
                            t,
                            t_meas,
                            G_meas,
                            initial_state,
                            process_noises,
                            meas_noise_std,
                            uex_const,
                            PN_const,
                            D_const,
                            SI_const,
                        )
                    )
                else:  # EKF
                    tasks.append(
                        (
                            "EKF",
                            0,
                            num_states,
                            dt,
                            dtmeas,
                            t,
                            t_meas,
                            G_meas,
                            initial_state,
                            process_noises,
                            meas_noise_std,
                            uex_const,
                            PN_const,
                            D_const,
                            SI_const,
                        )
                    )

            n_workers = min(len(tasks), cpu_count())

            with Pool(processes=n_workers) as p:
                results = p.map(worker, tasks)

            for k, (elapsed, G_est, all_state, all_var) in enumerate(results):
                time_train[k, i] = elapsed
                G_ests[k, i, :] = G_est
                error_est[k, i, :] = (G_true - G_est) ** 2
                states[k, i, :, :] = all_state
                all_vars[k, i, :, :, :] = all_var

            print(f"{(time.time() - time_start) / 60:.3f} mins have elapsed")

        print("Done")
        np.save("variables/time_arr_03_temp", time_train)
        np.save("variables/error_arr_03_temp", error_est)
        np.save("variables/G_ests_03_temp", G_ests)
        np.save("variables/G_trues_03_temp", G_trues)
        np.save("variables/states_03_temp", states)
        np.save("variables/vars_03_temp", all_vars)

if __name__ == "__main__":
    main()
