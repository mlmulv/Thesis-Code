#!/usr/bin/env python
# coding: utf-8

# __Markus Mulvihill__
# 
# __Last updated March 2026__

# # Necessary Packages and Modules

# In[2]:


import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy
import os
import sys


# In[3]:


sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import ext_kalman_filter
import icing_model
import icing_true
import particle_filter
import patient_cohort
import utils


# # Introduction
# 
# This file examines the accuracy and time complexity of the extended Kalman filter and the particle filter with various number of particles.

# # Setup
# ## Patient Cohort
# + 20 patients
# + Simulation over 60 hours
# + Initial $BG$ (mmol/L) is chosen from a log normal distribution of $\mu = 7.6$ and $\sigma=1.3$
# + Initial $Q$ and $I$ is 15 mu/L
# + Inital $P1$ and $P2$ is assumed to be 0 mmol/L
# + $\,SI$ for the patients is constant at $2 \cdot 10^{-4}$
# + $u_{ex}(t) = 75 \cdot e^{-\left(\frac{\log(2)}{300}\right) \cdot \left((t + 120) \mod (300)\right)}$ (mU/min)
# + $\,PN(t) = e^{-\left(\frac{\log(2)}{300}\right) \cdot \left(t \mod (300)\right)}$ (mmol/min)
# + $D(t) = 0.24$ (mmol/min)
# + True dynamics are derived from the ICING model with time steps $\,dt=1$ min
# + Blood Glucose measurements occur every 2 hours that contain Gaussian measurement noise $N(0,0.25^2)$

# In[ ]:


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


# ## State Estimators
# + State Variables are $\dot{\,BG}$, $\dot{Q}$, $\dot{I}$, $\dot{\,P1}$, $\dot{\,P2}$, $u_{en}$, $\,SI$ 
# + The initial states for all patients are assigned as $\begin{bmatrix} 7.6 & 15 & 15 & 0 & 0 & k_1 e^{-\frac{k_2}{k_3}15} & 2.1 \cdot 10^{-4} \end{bmatrix}$
# + Process Noise is $\text{diag} \left( \begin{bmatrix} \frac{1e^{-4}}{120} & \frac{1e^{-4}}{120} & \frac{1e^{-4}}{120} & \frac{1e^{-4}}{120} & \frac{1e^{-4}}{120} & 0 & \frac{1e^{-12}}{120} \end{bmatrix} \right)$
# + Initial states for each particle is sampled from a Gaussian distribution with a mean of the initial state and variance of the process noise

# In[5]:


y0 = [BG_logmu, 15, 15, 0, 0]
ICINGTrue = icing_true.ICINGTrue()
initial_state = np.append(y0, [ICINGTrue.params['k1']*np.exp(-y0[2]*ICINGTrue.params['k2']/ICINGTrue.params["k3"]), 2.1e-4])
initial_state = np.expand_dims(initial_state, axis=1)
num_states = len(initial_state)
process_noises = 1e-4/(dtmeas)*np.ones((len(initial_state),))
process_noises[-1] = (1e-6)**2/(dtmeas)
input_functions = {"D": D_func, "PN": PN_func, "Uex": uex_func}


# # Experiment
# + The time to run the state estimator and the accuracy of the $BG$ estimate will be saved
# + Particle Filters of sizes $N = \begin{bmatrix}  500  & 750 & 1000 & 1250 \end{bmatrix}$ will be used

# In[ ]:


try:
    mse_est = np.load("variables/mse_arr_02.npy")
    time_train = np.load("variables/time_arr_02.npy")
    print("Loaded variables from previous run")
except Exception:
    num_particles = np.arange(500, 2250, 250) 
    num_filters = len(num_particles) + 1 # 1 EKF filter and the number of variations of PF with the assigned number of particles
    t = np.arange(0, sim_hours*60+1, dt)
    t_meas = np.arange(0, sim_hours*60+1, dtmeas)
    time_train = np.zeros(shape=(num_filters, num_patients))
    mse_est = np.zeros_like(time_train)

    for i in range(num_patients):
        print(f"Patient {i+1} / {num_patients}")
        model = icing_model.ICINGModel(params=None,dt=dt, initial_state=initial_state, process_noise_vars=process_noises, measurement_noise_var=meas_noise_std**2, u_funcs=input_functions)
        G_true = patient_data[i]['BG']
        G_meas = patient_data[i]['BG_meas']

        for j in range(num_filters):
            if j != 0: # PF
                PF_filter = particle_filter.BootstrapParticleFilter(num_particles=num_particles[j-1], num_states=num_states, model=model, Ts_meas=dtmeas)
                sim = PF_filter.simulate(t, t_meas, G_meas, PF_filter)
                timeStart = time.time()
                _, _, _, _, G_est, _, _, _, _, _, _, _ = sim.run()
                timeEnd = time.time()
                time_train[j,i] = timeEnd-timeStart
                mse_est[j,i] = mean_squared_error(G_true,G_est)
            else: # EKF 
                EKF_filter = ext_kalman_filter.ExtendedKalmanFilter(num_states=num_states, model=model, ts_meas=dtmeas)
                sim = EKF_filter.simulate(t, t_meas, G_meas, EKF_filter)
                timeStart = time.time()
                _, _, G_est,_,_, _, _, _, _,_ = sim.run()
                timeEnd = time.time()
                time_train[j,i] = timeEnd-timeStart
                mse_est[j,i] = mean_squared_error(G_true,G_est)

    print("Done")
    np.save("variables/time_arr_02", time_train)
    np.save("variables/mse_arr_02", mse_est)   


# # Plot Results
# + The median time to run the estimators for each filter for all patients will be plotted along the x-axis
# + The median MSE will be compared for each filter along with the 95% confidence interval

# In[7]:


time_train_med = np.median(time_train, axis=1)
time_train_iqr = np.quantile(time_train, 0.75, axis=1) - np.quantile(time_train, 0.25, axis=1)
mse_est_med = np.median(mse_est, axis=1)
mse_est_iqr = np.quantile(mse_est, 0.75, axis=1) - np.quantile(mse_est, 0.25, axis=1)
labels = ['EKF',  'PF (500 Particles)', 'PF (750 Particles)', 'PF (1000 Particles)', 'PF (1250 Particles)']

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(time_train_med, mse_est_med, linestyle='--', marker='^')
for x,y,label in zip(time_train_med, mse_est_med, labels):
    ax.annotate(label, (x,y), ha='right')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Mean Square Error')
ax.set_title('Time to Run State Estimator vs Estimator Performance ')
ax.grid()
fig.tight_layout()


# In[11]:


print(time_train_med)
print(mse_est_med)
print(time_train_iqr)
print(mse_est_iqr)

