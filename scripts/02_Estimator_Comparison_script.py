import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy
import os
import sys
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import ext_kalman_filter
import icing_model
import icing_true
import particle_filter
import patient_cohort
import utils

def worker(task):
    uex_func = utils.gen_uex_func()
    PN_func = utils.gen_PN_func()
    D_func = utils.gen_D_func()
    SI_func = utils.gen_SI_func()
    input_functions = {"D": D_func, "PN": PN_func, "Uex": uex_func}
    filter_type, num_particles, num_states, dt, dtmeas, t, t_meas, G_meas, initial_state, process_noise_vars, meas_noise_std = task
    model = icing_model.ICINGModel(params=None,dt=dt, initial_state=initial_state, process_noise_vars=process_noise_vars, measurement_noise_var=meas_noise_std**2, u_funcs=input_functions)

    if filter_type == "PF":
        PF_filter = particle_filter.BootstrapParticleFilter(num_particles=num_particles, num_states=num_states, model=model, Ts_meas=dtmeas)
        sim = PF_filter.simulate(t, t_meas, G_meas, PF_filter)
        timeStart = time.time()
        _, _, _, _, G_est, _, _, _, _, _, _, _ = sim.run()
        timeEnd = time.time()
        time_elapsed = timeEnd - timeStart
    else:
        EKF_filter = ext_kalman_filter.ExtendedKalmanFilter(num_states=num_states, model=model, ts_meas=dtmeas)
        sim = EKF_filter.simulate(t, t_meas, G_meas, EKF_filter)
        timeStart = time.time()
        _, _, G_est,_,_, _, _, _, _,_ = sim.run()
        timeEnd = time.time()
        timeEnd = time.time()
        time_elapsed = timeEnd - timeStart
    return time_elapsed, G_est

def main():
    BG_logmu = 7.6
    BG_logsigma = 1.3
    num_patients = 100
    sim_hours = 60
    dt = 1
    dtmeas = 120
    meas_noise_std = 0.25
    PatientCohort = patient_cohort.PatientCohort(num_patients=num_patients, sim_hours=sim_hours, dt=dt, dtmeas=dtmeas, meas_noise_std=meas_noise_std, BG_params=[BG_logmu, BG_logsigma])
    uex_func = utils.gen_uex_func()
    PN_func = utils.gen_PN_func()
    D_func = utils.gen_D_func()
    SI_func = utils.gen_SI_func()
    patient_data = PatientCohort.patient_data(uex_func=uex_func, PN_func=PN_func, D_func=D_func, SI_func=SI_func)

    y0 = [BG_logmu, 15, 15, 0, 0]
    ICINGTrue = icing_true.ICINGTrue()
    initial_state = np.append(y0, [ICINGTrue.params['k1']*np.exp(-y0[2]*ICINGTrue.params['k2']/ICINGTrue.params["k3"]), 2.1e-4])
    initial_state = np.expand_dims(initial_state, axis=1)
    num_states = len(initial_state)
    process_noises = 1e-4/(dtmeas)*np.ones((len(initial_state),))
    process_noises[-1] = (1e-6)**2/(dtmeas)


    try:
        error_est = np.load("variables/error_arr_02.npy")
        time_train = np.load("variables/time_arr_02.npy")
        print("Loaded variables from previous run")
    except Exception:
        num_particles = np.asarray([100, 200, 300, 400, 500, 1000, 1500, 3000, 5000]) 
        num_filters = len(num_particles) + 1 # 1 EKF filter and the number of variations of PF with the assigned number of particles
        t = np.arange(0, sim_hours*60+1, dt)
        t_meas = np.arange(0, sim_hours*60+1, dtmeas)
        time_train = np.zeros(shape=(num_filters, num_patients))
        error_est = np.zeros(shape=(num_filters, num_patients, len(t))) 

        for i in range(num_patients):
            print(f"Patient {i+1} / {num_patients}")
            G_true = patient_data[i]['BG']
            G_meas = patient_data[i]['BG_meas']

            tasks = []
            for j in range(num_filters):
                if j != 0: # PF
                    tasks.append(("PF", num_particles[j-1], num_states,dt, dtmeas, t, t_meas, G_meas, initial_state, process_noises, meas_noise_std))
                else: # EKF 
                    tasks.append(("EKF", 0, num_states, dt, dtmeas, t, t_meas, G_meas, initial_state, process_noises, meas_noise_std)) 
            
            n_workers = min(len(tasks), cpu_count())

            with Pool(processes=n_workers) as p:        
                results = p.map(worker, tasks)

            for j, (elapsed, G_est) in enumerate(results):        
                time_train[j, i] = elapsed        
                error_est[j, i, :] = G_true - G_est
            
        print("Done")
        np.save("variables/time_arr_02", time_train)
        np.save("variables/error_arr_02", error_est)  

if __name__ == "__main__":
    main() 
