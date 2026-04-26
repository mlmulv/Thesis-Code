import numpy as np
import matplotlib.pyplot as plt
import scipy

import time
import pickle
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import ext_kalman_filter
import particle_filter
import icing_model
import icing_true
import patient_cohort
import utils

def worker(task):
    uex_func = utils.gen_uex_func()
    PN_func = utils.gen_PN_func()
    D_func = utils.gen_D_func()
    #SI_func = utils.gen_SI_func()
    input_functions = {"D": D_func, "PN": PN_func, "Uex": uex_func}
    filter_type, num_particles, num_states, dt, dtmeas, t, t_meas, G_meas, initial_state, process_noise_vars, meas_noise_std = task
    model = icing_model.ICINGModel(params=None,dt=dt, initial_state=initial_state, process_noise_vars=process_noise_vars, measurement_noise_var=meas_noise_std**2, u_funcs=input_functions)

    if filter_type == "PF":
        PF_filter = particle_filter.BootstrapParticleFilter(num_particles=num_particles, num_states=num_states, model=model, Ts_meas=dtmeas)
        sim = PF_filter.simulate(t, t_meas, G_meas, PF_filter)
        time_filter_start = time.time()
        _, _, saved_particles, saved_weights, _, _, _, _, _, _, _, _ = sim.run()
        time_filter_end = time.time()
        G_est = np.average(saved_particles[:,:,0], weights=saved_weights, axis=1)
        time_run = time_filter_end - time_filter_start

    else:
        EKF_filter = ext_kalman_filter.ExtendedKalmanFilter(num_states=num_states, model=model, ts_meas=dtmeas)
        sim = EKF_filter.simulate(t, t_meas, G_meas, EKF_filter)
        time_filter_start = time.time()
        saved_state, _, _, _,_, _, _, _, _,_ = sim.run()
        time_filter_end = time.time()
        G_est = saved_state[:,0]
        time_run = time_filter_end - time_filter_start

    return G_est, time_run

def main():
    try:
        with open('variables/G_errors.pkl', 'rb') as f:
            G_errors = pickle.load(f) 
        with open('variables/G_ests.pkl', 'rb') as f:
            G_ests = pickle.load(f)
        with open('variables/Cohorts.pkl', 'rb') as f:
            Cohorts = pickle.load(f)
        with open('variables/filters_time_run.pkl', 'rb') as f:
            filters_time_run = pickle.load(f) 
        print("Previous run variables already exist")

    except Exception:
        BG_logmu = 7.6
        BG_logsigma = 0.8
        y0 = [BG_logmu, 15.0, 15.0, 0, 0]

        num_patients = 10
        uex_func = utils.gen_uex_func()
        PN_func = utils.gen_PN_func()
        D_func = utils.gen_D_func()
        SI_func = utils.gen_SI_func()
        input_functions = {"D": D_func, "PN": PN_func, "Uex": uex_func}

        sim_hours = 60
        dtmeas = 120
        t_meas = np.arange(0, sim_hours*60+1, dtmeas)

        ICINGTrue = icing_true.ICINGTrue()
        initial_state = np.append(y0, [ICINGTrue.params['k1']*np.exp(-y0[2]*ICINGTrue.params['k2']/ICINGTrue.params["k3"]), 2.1e-4])
        initial_state = np.expand_dims(initial_state, axis=1)
        num_states = len(initial_state)
        process_noises = 1e-4/(dtmeas)*np.ones((len(initial_state),))
        process_noises[-1] = (1e-6)**2/(dtmeas)
        dts = np.asarray([1, 5, 10])
        meas_noise_std = 0.25

        num_particles = np.asarray([100, 200, 300, 400, 500, 1000, 1500, 3000, 5000])
        num_filters = len(num_particles) + 1

        G_ests = []
        G_errors = []
        filters_time_run = []
        Cohorts = []

        time_start = time.time()
        for idx, dt in enumerate(dts):
            print(f"dt = {dt} run")
            t = np.arange(0, sim_hours*60+1, dt)
            PatientCohort = patient_cohort.PatientCohort(num_patients=num_patients, sim_hours=sim_hours, dt=dt, dtmeas=dtmeas, meas_noise_std=meas_noise_std, BG_params=[BG_logmu, BG_logsigma])

            patient_data = PatientCohort.patient_data(uex_func=uex_func, PN_func=PN_func, D_func=D_func, SI_func=SI_func)
            Cohorts.append(patient_data)
            G_error = np.zeros((num_patients, num_filters, len(t)))
            G_est = np.zeros_like(G_error)
            filter_time_run = np.zeros((num_patients, num_filters))

            for i in range(num_patients):
                print(f"Patient {i+1} / {num_patients}")
                G_true = patient_data[i]['BG']
                G_meas = patient_data[i]['BG_meas']
                tasks = []
                for j in range(num_filters):
                    if j != 0:
                         tasks.append(("PF", num_particles[j-1], num_states,dt, dtmeas, t, t_meas, G_meas, initial_state, process_noises, meas_noise_std))
                    else:
                        tasks.append(("EKF", 0, num_states, dt, dtmeas, t, t_meas, G_meas, initial_state, process_noises, meas_noise_std)) 
               
                n_workers = min(len(tasks), cpu_count())
                with Pool(processes=n_workers) as p:        
                    results = p.map(worker, tasks)

                for j, (G_est_filter, time_run) in enumerate(results):        
                    G_error[i,j,:] = (G_est_filter - G_true)**2
                    print(np.mean(G_error[i,j,:]))
                    G_est[i,j,:] = G_est_filter
                    filter_time_run[i,j] = times_run

            G_ests.append(G_est)
            G_errors.append(G_error)
            print(filter_time_run)
            filters_time_run.append(filter_time_run)

            print(f"{(time.time() - time_start)/60:.3f} mins have elapsed") 

        with open('variables/G_errors.pkl', 'wb') as f:
            pickle.dump(G_errors, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('variables/G_ests.pkl', 'wb') as f:
            pickle.dump(G_ests, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open('variables/Cohorts.pkl', 'wb') as f:
            pickle.dump(Cohorts, f, protocol=pickle.HIGHEST_PROTOCOL) 

        with open('variables/filters_time_run.pkl', 'wb') as f:
            pickle.dump(filters_time_run, f, protocol=pickle.HIGHEST_PROTOCOL) 


if __name__ == "__main__":
    main()
