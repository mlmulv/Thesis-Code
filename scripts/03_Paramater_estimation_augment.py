import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import ext_kalman_filter
import particle_filter
import icing_true
import icing_model
import patient_cohort
import utils
import param_est

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
        _, _, saved_particles, saved_weights, _, _, _, _, _, _, _, _ = sim.run()
        SI_est = np.average(saved_particles[:,:,-1], weights=saved_weights, axis=1)
    else:
        EKF_filter = ext_kalman_filter.ExtendedKalmanFilter(num_states=num_states, model=model, ts_meas=dtmeas)
        sim = EKF_filter.simulate(t, t_meas, G_meas, EKF_filter)
        saved_state, _, _, _,_, _, _, _, _,_ = sim.run()
        SI_est = saved_state[:,-1]

    return SI_est


def main():
    try:
        SI_ests = np.load("variables/SI_ests_augment_static_03.npy")
        SI_errs = np.load("variables/SI_errs_augment_static_03.npy")
        print("Loaded variables from previous run")
    except Exception:
        BG_logmu = 7.6
        BG_logsigma = 1.3
        num_patients = 1
        sim_hours = 60
        dt = 1.0
        t = np.arange(0, sim_hours*60+1, dt)
        dtmeas = 60
        t_meas = np.arange(0, sim_hours*60+1, dtmeas)
        meas_noise_std = 0.25
        SI_const = 2.0e-4
        SI_init = 2.2e-4
        y0 = [BG_logmu, 1.1*15.0, 1.1*15.0, 0.0, 0.0]

        PatientCohort = patient_cohort.PatientCohort(num_patients=num_patients, sim_hours=sim_hours, dt=dt, dtmeas=dtmeas, meas_noise_std=meas_noise_std, BG_params=[BG_logmu, BG_logsigma])
        uex_func = utils.gen_uex_func()
        PN_func = utils.gen_PN_func()
        D_func = utils.gen_D_func()
        SI_func = utils.gen_SI_func(SI_const=SI_const)
        SI_true = SI_const * np.ones_like(t)
        patient_data = PatientCohort.patient_data(uex_func=uex_func, PN_func=PN_func, D_func=D_func, SI_func=SI_func)
        y0 = [BG_logmu, 1.1*15, 1.1*15, 0, 0]
        ICINGTrue = icing_true.ICINGTrue()
        initial_state = np.append(y0, [ICINGTrue.params['k1']*np.exp(-y0[2]*ICINGTrue.params['k2']/ICINGTrue.params["k3"]), SI_init])
        initial_state = np.expand_dims(initial_state, axis=1)
        num_states = len(initial_state)
        process_noises = np.asarray([(0.00001*11)**2, (0.00005*100)**2, (0.00005*175)**2, (0.0005*1.5)**2, (0.0005*1.5)**2, (0.00005*150)**2, (1e-5/120)**2])
        num_particles = np.asarray([250, 500, 750, 1000]) 
        num_filters = len(num_particles) + 1 # 1 EKF filter and the number of variations of PF with the assigned number of particles

        SI_ests = np.zeros((num_filters, num_patients, len(t)))
        SI_errs = np.zeros_like(SI_ests)

        time_start = time.time()
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

            for j, SI_est in enumerate(results):
                SI_ests[j,i,:] = SI_est
                SI_errs[j,i,:] = 100 * (np.abs((SI_est - SI_true)))/SI_true

            print(f"{(time.time() - time_start)/60:.3f} mins have elapsed")

        print("Done")
        np.save("variables/SI_ests_augment_static_03", SI_ests)
        np.save("variables/SI_errs_augment_static_03", SI_errs)
        

if __name__ == "__main__":
    main()