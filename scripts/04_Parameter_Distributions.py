import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import patient_cohort

def main():
    try:
        x0 = np.load("variables/x0_04.npy")
        inputs = np.load("variables/inputs_04.npy")
        states = ['BG (mmol/L)', 'I (mU/L)', 'Q (mu/L)' , 'P1 (mmol)', 'P2 (mmol)']
        input_labels = ['uex (mU/h)', 'D (mmol/min)', 'PN (mmol/min)', 'SI']

        fig, ax = plt.subplots(x0.shape[0],1, figsize=(12,8))
        for i in range(x0.shape[0]):
            sns.histplot(data=x0[i,:], bins=50, stat="density", ax=ax[i])
            ax[i].set_xlabel(states[i])
            ax[i].grid()
        fig.suptitle('Observed Initial States of 10000 Patients')
        fig.tight_layout()

        fig2, ax2 = plt.subplots(inputs.shape[0],1, figsize=(12,8))
        for i in range(inputs.shape[0]):
            sns.histplot(data=inputs[i,:], bins=50, stat="density", ax=ax2[i])
            ax2[i].set_xlabel(input_labels[i]) 
            ax2[i].grid()
        fig2.suptitle('Input and Parameter values for 10000 Patients')
        fig2.tight_layout()

        plt.show()

        print("Mean initial states: ", np.mean(x0, axis=1))
        print("Initial Covariance: ", np.cov(x0, rowvar=True, bias=False))
         

    except Exception:
        num_patients = None
        sim_hours = 60
        dt = 1.0
        dtmeas = 120
        meas_noise_std = None
        uex_bounds = [0, 0.6]
        D_bounds = [0.1, 0.3]
        PN_bounds = [0.1, 0.3]
        SI_params = np.asarray([0, 5e-4, 6e-4])
        y0 = [7.0, 15.0, 15.0, 0.5, 0.5]
        PatientCohort = patient_cohort.PatientCohort(num_patients, sim_hours, dt, dtmeas, meas_noise_std, uex_bounds, D_bounds, PN_bounds, SI_params, y0)
        num_simulations = 10000
        x0 = np.zeros((len(y0), num_simulations))
        inputs = np.zeros((4, num_simulations)) # uex, D, PN, SI

        for i in range(num_simulations):
            x0[:,i], inputs[:,i] = PatientCohort.initial_states()

        np.save("variables/x0_04", x0)
        np.save("variables/inputs_04", inputs)

if __name__ == "__main__":
    main()