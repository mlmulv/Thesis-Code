from utils import gen_PN_func, integral_approximate_SI
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class ExtendedKalmanFilter:
    def __init__(self, num_states, model, Ts_meas):
        # Parameters
        self.num_states = num_states
        self.model = model
        self.Ts_meas = Ts_meas

    def initialize_filter(self):
        state = self.model.initial_state
        noise_vars = np.ones(len(state))
        noise_vars[-1] = (1e-5)**2
        noise_var = np.diag(noise_vars)
        self.Q = np.diag(self.model.process_noise_var)
        self.h_model_func = self.calc_h_model_func()
        self.R = np.array([[self.model.measurement_noise_var]])
        return state, noise_var

    def calc_f_model_func(self, m):
        # Variable Initialization
        f_model_func = np.zeros((self.num_states, self.num_states))
        dt = self.model.dt
        pG = self.model.params["pG"]
        alphaG = self.model.params["alphaG"]
        nI = self.model.params["nI"]
        nC = self.model.params["nC"]
        nK = self.model.params["nK"]
        nL = self.model.params["nL"]
        alphaI = self.model.params["alphaI"]
        VI = self.model.params["VI"]
        xL = self.model.params["xL"]
        k1 = self.model.params["k1"]
        k2 = self.model.params["k2"]
        k3 = self.model.params["k3"]
        G = m[0,0]
        Q = m[1,0]
        I = m[2,0]
        SI = m[4,0]


        # Jacobian Matrix Calculation
        f_model_func[0,0] = 1 - dt*pG - dt*SI*Q/(1 + alphaG*Q)
        f_model_func[0,1] = -dt*SI*G / (1 + alphaG*Q)**2
        f_model_func[0,4] = -dt*G*Q/(1 + alphaG*Q)

        f_model_func[1,1] = 1 -dt*nI - dt*nC/(1+alphaG*Q)**2
        f_model_func[1,2] = dt*nI

        f_model_func[2,1] = dt*nI
        f_model_func[2,2] = 1 - dt*nK - dt*nL/(1+alphaI*I)**2 - dt*nI
        f_model_func[2,3] = dt*(1-xL)/VI

        f_model_func[3,2] = (-k1*k2/k3)*np.exp(-I*k2/k3)

        f_model_func[4,4] = 1

        return f_model_func

    def calc_h_model_func(self):
        calc_h_model_func = np.zeros((1,self.num_states))
        calc_h_model_func[0] = 1
        return calc_h_model_func


    def filter_predict(self, state_update, noise_var_update, t):
        u_vec = self.model.get_inputs(t) 
        f_model_func = self.calc_f_model_func(state_update)
        state_predict = self.model.state_update(state_update, u_vec, t)
        noise_var_predict = np.matmul(f_model_func, np.matmul(noise_var_update, f_model_func.T)) + self.Q
        return state_predict, noise_var_predict

    def filter_update(self, state_predict, noise_var_predict, y):
        S = np.matmul(self.h_model_func, np.matmul(noise_var_predict, self.h_model_func.T)) + self.R
        K = np.matmul(noise_var_predict, np.matmul(self.h_model_func.T, np.linalg.inv(S)))
        v = np.array([y - state_predict[0]])
        state_update = state_predict + np.matmul(K, v)
        noise_var_update = noise_var_predict - np.matmul(K, np.matmul(S,K.T))
        return state_update, noise_var_update

    def predict_to_time(self, state, noise_var, t):
        # Run prediction from t_last_measurement to t
        predictions = np.zeros((int(t-self.t_last_measurement), self.num_states))
        step = 0

        for ti in range(self.t_last_measurement, t):
            state_next, noise_var_next = self.filter_predict(state, noise_var, ti)
            predictions[step] = state_next[:,0]
            state = state_next
            noise_var = noise_var_next
            step += 1
        return predictions

    def filter_iteration(self, state, noise_var, t, yk=None):
        # Make prediction
        state_next, noise_var_next = self.filter_predict(state, noise_var, t)

        # if there is a measurement, update filter
        if yk is not None:
            self.t_last_measurement = t
            return self.filter_update(state_next, noise_var_next,yk)

        else:
            return state_next, noise_var_next
        

    class simulate:
        def __init__(self, t, t_meas, G_meas, filter):
            self.t = t
            self.t_meas = t_meas
            self.G_meas = G_meas
            self.filter = filter

        def run(self):
            num_states = self.filter.num_states
            G_est = np.zeros_like(self.t)
            G_pred = np.zeros_like(self.t)
            Q_est = np.zeros_like(self.t)
            I_est = np.zeros_like(self.t)
            Uen_est = np.zeros_like(self.t)
            SI_est = np.zeros_like(self.t)
            SI_fit = np.zeros_like(self.t)
            saved_state = np.zeros((len(self.t), num_states))
            saved_noise_var = np.zeros((len(self.t), num_states, num_states))
            state, noise_var = self.filter.initialize_filter()
            vars_diag = []
            PN_func = gen_PN_func()
            alphaG = self.filter.model.params["alphaG"]
            pG = self.filter.model.params["pG"]
            VG = self.filter.model.params["VG"]
            EGP = self.filter.model.params["EGP"]
            CNS = self.filter.model.params["CNS"]
            
            for idx, ti in enumerate(self.t):
                # If a measurement occured at this time
                if ti in self.t_meas:
                    sample_idx = np.argmin(np.abs(self.t_meas - ti))
                    # if sample_idx > 1: break
                    # print(f"Got measurement {sample_idx} at time {int(ti)}")
                    state_next, noise_var_next = self.filter.filter_iteration(state, noise_var, int(ti), self.G_meas[sample_idx])

                    if ti != self.t_meas[0]:
                        idxs = np.arange(idx-self.filter.Ts_meas, idx+1)
                        t = self.t[idxs]
                        print(t)
                        G_est_2h = np.append(G_est[idxs], state_next[0,:])
                        Q_est_2h = np.append(Q_est[idxs], state_next[1,:])
                        Q_bar_est_2h = Q_est_2h / (1 + alphaG * Q_est_2h)
                        PN_2h = [PN_func(i) for i in t]
                        SI_fit[idxs] = integral_approximate_SI(G_est_2h, PN_2h, Q_bar_est_2h, t[0], t[-1], 1, pG, VG, EGP, CNS)

                    # Predict 2 hours ahead (if not last measurement)
                    if ti != self.t_meas[-1]:
                        # print(f"Predicting the next 2 hours...")
                        pred = self.filter.predict_to_time(state_next, noise_var_next, t = int(ti+self.filter.Ts_meas))
                        G_pred[idx:idx+self.filter.Ts_meas] = pred[:,0]

                # If no measurement
                else:
                    state_next, noise_var_next = self.filter.filter_iteration(state, noise_var, ti)

                saved_state[idx], saved_noise_var[idx] = state_next[:,0], noise_var_next[:,0]
                G_est[idx] = state_next[0,0]
                Q_est[idx] = state_next[1,0]
                I_est[idx] = state_next[2,0]
                Uen_est[idx] = state_next[3,0]
                SI_est[idx] = state_next[-1,0]
                state = state_next
                noise_var = noise_var_next

            # print("Done")
            return saved_state, saved_noise_var, G_est, Q_est, I_est, Uen_est, SI_est, G_pred, SI_fit

        def plot(self, saved_state, saved_noise_var, G_est, Q_est, I_est, Uen_est, SI_est, G_pred, SI_fit, I_true, model):

            fig, ax = model.plot_sim()
            ax[0].plot(self.t_meas, self.G_meas,  marker='o', label='BG measurements', color='k', linestyle='--')
            ax[0].plot(self.t, G_est, label='Fitted BG', color='blue', linestyle='--')
            ax[0].plot(self.t[1:], G_pred[:-1], label="2 hour ahead prediction", color="green")
            uen_true = model.params["k1"]*np.exp(I_true*model.params["k2"]/model.params["k3"])
            ax[1].plot(self.t, I_est, label='Fitted I', color='blue', linestyle='-')
            ax[1].plot(self.t, uen_true, label="Endogenous Insulin", color="orange", linestyle="-.")
            ax[1].plot(self.t, Uen_est, label="Fitted Uen", color='blue', linestyle='-.')
            ax[2].plot(self.t, Q_est, label='Fitted Q', color='blue', linestyle='-')
            # ax[3].plot(t_meas, Q_est, label='Fitted Q', color='blue', linestyle='--')
            ax[4].plot(self.t, SI_est, label='Fitted SI', color='red', linestyle='--')
            ax[4].plot(self.t, SI_fit, label='Integral Fit', color='blue', linestyle='--')
            # err_ax = ax[4].twinx()
            # err_ax.spines["right"]#.set_position(("outward", 60))
            # err_ax.yaxis.set_ticks_position("left")
            # err_ax.yaxis.set_label_position("left")
            # err_ax.plot(self.t, np.abs((SI_est - model.last_simulation["SI"])*100/model.last_simulation["SI"]), color="purple", label="Relative error [%]")
            # err_ax.set_ylabel("Relative abs. error [%]")
            # err_ax.legend()
            for a in ax:
                a.legend()
            plt.show()
            fig.tight_layout()




