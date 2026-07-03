import os
import sys
import time

import numpy as np
import tomllib

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

    if filter_type == "PF":
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
        )
        PF_filter = particle_filter.BootstrapParticleFilter(
            num_particles=num_particles,
            num_states=num_states,
            model=model,
            Ts_meas=dtmeas,
        )
        sim = PF_filter.simulate(t, t_meas, G_meas, PF_filter)
        _, _, saved_particles, saved_weights, _, _, _, _, _, _, _, _ = sim.run()
        SI_est = np.average(saved_particles[:, :, -1], weights=saved_weights, axis=1)
        SI_var = np.average(
            (saved_particles[:, :, -1] - SI_est[:, np.newaxis]) ** 2,
            weights=saved_weights,
            axis=1,
        )
    else:
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
        )

        EKF_filter = ext_kalman_filter.ExtendedKalmanFilter(
            num_states=num_states, model=model, ts_meas=dtmeas
        )
        try:
            sim = EKF_filter.simulate(t, t_meas, G_meas, EKF_filter)
            saved_state, saved_noise_var, _, _, _, _, _, _, _, _ = sim.run()
            SI_est = saved_state[:, -1]
            SI_var = saved_noise_var[:, -1, -1]
        except Exception as e:
            SI_est = np.nan
            SI_var = np.nan

    return SI_est, SI_var


def main():
    try:
        SI_ests = np.load("variables/SI_ests_augment_vary_04.npy")
        SI_vars = np.load("variables/SI_vars_augment_vary_04.npy")
        SI_errs = np.load("variables/SI_errs_augment_vary_04.npy")
        SI_true_arr = np.load("variables/SI_true_arr_augment_vary_04.npy")
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
        sim_hours = cfg[curr_module]["sim_hours_vary"]
        num_patients = cfg[curr_module]["num_patients"]
        process_noise_factor = cfg[curr_module]["process_noise_factor"]
        SI_process_noise = cfg[root_module]["SI_process_noise"]
        num_particles = np.asarray(cfg[curr_module]["num_particles"])
        deviations = cfg[curr_module]["deviations"]
        change_SI = cfg[curr_module]["change_SI"]
        SI_scale = cfg[curr_module]["SI_scale"]

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
            SI_scale=SI_scale,
        )

        patient_data = PatientCohort.patient_data()

        ICINGTrue = icing_true.ICINGTrue()

        num_filters = (
            len(num_particles) + 1
        )  # 1 EKF filter and the number of variations of PF with the assigned number of particles
        num_deviations = len(deviations)
        t = np.arange(0, sim_hours * 60 + 1, dt)
        t_meas = np.arange(0, sim_hours * 60 + 1, dtmeas)
        SI_ests = np.full(
            (num_deviations, num_filters, num_patients, len(t)), np.nan, dtype=float
        )
        SI_vars = np.full_like(SI_ests, np.nan)
        SI_errs = np.full_like(SI_ests, np.nan)
        SI_true_arr = np.zeros((num_patients, len(t)))

        process_noises = np.asarray(
            [
                (0.005) * process_noise_factor,
                (1) * process_noise_factor,
                (1) * process_noise_factor,
                (0.005) * process_noise_factor,
                (0.025) * process_noise_factor,
                (1e-8) * process_noise_factor,
                (SI_process_noise) * process_noise_factor,
            ]
        )

        time_start = time.time()
        for i in range(num_patients):
            print(f"Patient {i + 1} / {num_patients}")
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
            SI_true = patient_data[i]["SI"]
            SI_true_arr[i, :] = SI_true
            SI_0 = patient_data[i]["SI"][0]
            y0 = [G_0, I_0, Q_0, P1_0, P2_0]
            for k in range(num_deviations):
                deviation = deviations[k]
                initial_state = deviation * np.append(
                    y0,
                    [
                        ICINGTrue.params["k1"]
                        * np.exp(
                            -y0[2] * ICINGTrue.params["k2"] / ICINGTrue.params["k3"]
                        ),
                        SI_0,
                    ],
                )
                initial_state = np.expand_dims(initial_state, axis=1)
                num_states = len(initial_state)

                tasks = []
                for j in range(num_filters):
                    if j != 0:  # PF
                        noOp = True
                        # tasks.append(
                        #     (
                        #         "PF",
                        #         num_particles[j - 1],
                        #         num_states,
                        #         dt,
                        #         dtmeas,
                        #         t,
                        #         t_meas,
                        #         G_meas,
                        #         initial_state,
                        #         process_noises,
                        #         meas_noise_std,
                        #         uex_const,
                        #         PN_const,
                        #         D_const,
                        #         SI_const,
                        #     )
                        # )

                    else:  # EKF
                        # noOp = True
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

                for j, (SI_est, SI_var) in enumerate(results):
                    if np.isfinite(SI_est).all():
                        SI_ests[k, j, i, :] = SI_est
                        SI_vars[k, j, i, :] = SI_var
                        SI_errs[k, j, i, :] = (
                            100 * (np.abs((SI_est - SI_true))) / SI_true
                        )
            print(f"{(time.time() - time_start) / 60:.3f} mins have elapsed")

        print("Done")
        nans = np.isfinite(SI_ests)
        filter_patient_nan = ~nans.all(axis=(3))
        broadcast = filter_patient_nan[:,:,:,None]
        SI_ests[broadcast] = np.nan
        SI_vars[broadcast] = np.nan
        SI_errs[broadcast] = np.nan
        np.save("variables/SI_ests_augment_vary_04", SI_ests)
        np.save("variables/SI_vars_augment_vary_04", SI_vars)
        np.save("variables/SI_errs_augment_vary_04", SI_errs)
        np.save("variables/SI_true_arr_augment_vary_04", SI_true_arr)


if __name__ == "__main__":
    main()
