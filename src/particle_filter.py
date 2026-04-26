import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from copy import deepcopy
import utils 
rng = np.random.default_rng()

class BootstrapParticleFilter:
    def __init__(self, num_particles, num_states, model, Ts_meas, resampling_method="systematic"):
        # Copy parameters
        self.num_particles = num_particles
        self.num_states = num_states
        self.model = model
        self.step_counter = None
        self.resampling_method = resampling_method
        self.Ts_meas = Ts_meas

        # Initialise particles and weights
        self.particles = np.zeros((num_particles, num_states))
        self.initialise_particles()
        self.log_weights = np.ones(num_particles) * (-np.log(num_particles))

        # Set some hyperparameters
        self.t_last_measurement = 0
        self.resample_threshold = num_particles / 2  # Resample when effective sample size is less than half the number of particles
        

    def initialise_particles(self):
        for i in range(self.num_particles):
            self.particles[i,:] = self.model.draw_initial_state()
        # self.step_counter = 0

    def get_weights(self):
        return np.exp(self.log_weights)
    
    def get_log_weights(self):
        return self.log_weights

    def get_particles(self):
        return self.particles    
    
    def get_particles_and_weights(self):
        return self.get_particles(), self.get_weights()
    
    def __multinomial_resampling(self):
        indices = rng.choice(self.num_particles, size=self.num_particles, p=self.get_weights())
        return indices
    
    def __systematic_resampling(self):
        u_1 = rng.uniform(0, 1/self.num_particles)
        u = np.concatenate(([u_1], u_1 + np.arange(1, self.num_particles)/self.num_particles))
        cdf = np.cumsum(self.get_weights())
        indices = np.empty((self.num_particles,), dtype=int)

        i = j = 0
        while i < self.num_particles:
            if u[i] < cdf[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def predict_to_time(self, t):
        # Run prediction from t_last_measurement to t
        particles = deepcopy(self.get_particles())
        log_weights = deepcopy(self.get_log_weights())
        predictions = np.zeros((int(t-self.t_last_measurement), self.num_states))
        step=0
        for ti in range(self.t_last_measurement, t):
            u_vec = self.model.get_inputs(ti)
            for i in range(self.num_particles):
                particles[i] = self.model.state_update(particles[i], u_vec, ti)[:,0]
            mean = np.average(particles, weights=np.exp(log_weights), axis=0)
            predictions[step] = mean
            step += 1
        
        return predictions


    def resample_particles(self):
        # Resample with desired method
        if self.resampling_method == "multinomial":
            indices = self.__multinomial_resampling()
        elif self.resampling_method == "systematic":
            indices = self.__systematic_resampling()
        else:
            raise ValueError(f"Unknown resampling method: {self.resampling_method}")
        
        self.particles = self.particles[indices]
        self.log_weights.fill(-np.log(self.num_particles))

    
    
    def update_particles(self, t, yk=None):
        # Propagate particles through model for all times until this sample
        u_vec = self.model.get_inputs(t)
        for i in range(self.num_particles):
            self.particles[i,:] = self.model.state_update(self.particles[i,:], u_vec, t)[:,0]

        if yk: #measurement was included => update weights
            self.t_last_measurement = t

            # Update weights based on likelihood of observation
            # print(f"updating {yk, t}: ")
            for i in range(self.num_particles):
                temp_weight_correction = self.model.measurement_log_likelihood(yk, self.particles[i], t)
                # print(f"i {i} -- est G {self.particles[i,0]} -- yk {yk} -- squared error {(self.particles[i,0]-yk)**2} -- correction {temp_weight_correction} -- measurement likelihood {sp.stats.norm.logpdf(yk, loc=self.particles[i,0], scale=0.25)}")
                self.log_weights[i] += temp_weight_correction
            
            # Normalize weights
            self.log_weights -= sp.special.logsumexp(self.log_weights)

            # print(f"Weight sum at step {self.step_counter}: {np.exp(sp.special.logsumexp(self.log_weights))}")
            
            # # Resample if needed
            # effective_sample_size = np.exp(-sp.special.logsumexp(2 * self.log_weights))
            # if effective_sample_size < self.resample_threshold:
            #     self.resample_particles()
            # if effective_sample_size < self.resample_threshold or self.step_counter < 100 or self.iterations_since_last_resampling > self.min_resampling_frequency:
            #     print(f"Resampling at step {self.step_counter}")
            #     self.iterations_since_last_resampling = 0
            #     self.resample_particles()
            # else:
            #     self.iterations_since_last_resampling += 1

            # Always resample after a measurement
            self.resample_particles()

    def filter(self):
        weights = self.get_weights()
        x_mean = np.average(self.particles, weights=weights, axis=0)
        x_var = np.average((self.particles - x_mean)**2, weights=weights, axis=0)
        return x_mean, np.sqrt(x_var)

    
    class simulate:
        def __init__(self, t, t_meas, G_meas, filter):
            self.t = t
            self.t_meas = t_meas
            self.G_meas = G_meas
            self.filter = filter

        def run(self):
            num_particles = self.filter.num_particles
            num_states = self.filter.num_states
            initial_particles, initial_weights = self.filter.get_particles_and_weights()
            G_est = np.zeros_like(self.t)
            G_pred = np.zeros_like(self.t)
            Q_est = np.zeros_like(self.t)
            I_est = np.zeros_like(self.t)
            P1_est = np.zeros_like(self.t)
            P2_est = np.zeros_like(self.t)
            Uen_est = np.zeros_like(self.t)
            SI_est = np.zeros_like(self.t)
            saved_particles = np.zeros((len(self.t), num_particles, num_states))
            saved_weights = np.zeros((len(self.t), num_particles))

            for idx, ti in enumerate(self.t):
                # Check if a measurement occured at this time
                if ti in self.t_meas:
                    sample_idx = np.argmin(np.abs(self.t_meas - ti))
                    # if sample_idx > 1: break
                    # print(f"Got measurement {sample_idx} at time {int(ti)}")
                    # print(f"Measured {G_meas[sample_idx]}; true value {G_true[idx]}")
                    self.filter.update_particles(int(ti), yk=self.G_meas[sample_idx])
                    # Predict 2 hours ahead (if not last measurement)
                    # if ti != self.t_meas[-1]:
                    #     # print(f"Predicting the next 2 hours...")
                    #     pred = self.filter.predict_to_time(int(ti+self.filter.Ts_meas))
                    #     G_pred[idx:idx+self.filter.Ts_meas] = pred[:,0]

                # else no measurement, so just update particles
                else: 
                    self.filter.update_particles(int(ti))

                # Get stats for the current filter
                mean_i, var_i = self.filter.filter()
                temp_particles, temp_weights = self.filter.get_particles_and_weights()
                saved_particles[idx], saved_weights[idx] = temp_particles, temp_weights
                G_est[idx] = mean_i[0]
                Q_est[idx] = mean_i[1]
                I_est[idx] = mean_i[2]
                P1_est[idx] = mean_i[3]
                P2_est[idx] = mean_i[4]
                Uen_est[idx] = mean_i[5]
                SI_est[idx] = mean_i[-1]
            # print("done")
            
            # UNEXPECTED BEHAVIOUR THAT ESTIMATES ARE AN INTEGER. FOR NOW ONY USE SAVED_PARTICLES AND SAVED_PARTICLES FOR STATE CALCULATIONS
            return initial_particles, initial_weights, saved_particles, saved_weights, G_est, Q_est, I_est, P1_est, P2_est, Uen_est, SI_est, G_pred

        def plot(self, initial_particles, initial_weights, saved_particles, saved_weights, G_est, Q_est, I_est, P1_est, P2_est, Uen_est, SI_est, G_pred, I_true, ts, scale, model):
            
            sci_formatter = FuncFormatter(lambda val, _: f"{val*scale:.2f}")
            # fig, ax = plt.subplots(1,len(ts)+1, figsize=(15, 6))
            fig, ax = plt.subplots(2,len(ts)+1, figsize=(15, 6))
            for axi in np.atleast_2d(ax)[1, :]:
                axi.xaxis.set_major_formatter(sci_formatter)
                axi.xaxis.offsetText.set_visible(False)
                axi.yaxis.set_major_formatter(sci_formatter)
                axi.set_xlabel(r"$\times 10^{-4}$")
            for idx, axi in enumerate(np.atleast_2d(ax)[0, :]):
                axi.yaxis.set_major_formatter(sci_formatter)

            target_kde_std = [0.05, 1e-5]
            for i, ti in enumerate(ts):
                # plot_weighted_kde(ax[0,i], saved_particles[ti,:,0], saved_weights[ti], bw=target_kde_std[0], color="C0")
                ax[0,i].stem(saved_particles[ti,:,0], saved_weights[ti], basefmt="k")
                ax[0,i].set_title(ti)
                ax[0,0].set_ylabel("Glucose")

                utils.plot_weighted_kde(ax[1,i], saved_particles[ti,:,-1], saved_weights[ti], bw=target_kde_std[0], color="C0")
                # ax[1,i].stem(saved_particles[ti,:,-1], saved_weights[ti], basefmt="k")
                ax[1,0].set_ylabel(r"SI")
            utils.plot_weighted_kde(ax[0,-1], initial_particles[:,0], initial_weights, bw=target_kde_std[0], color="C0")
            # ax[0,-1].stem(initial_particles[:,0], initial_weights, basefmt="k")
            ax[0,-1].set_title("Initial")
            utils.plot_weighted_kde(ax[1,-1], initial_particles[:,-1], initial_weights, bw=target_kde_std[0], color="C0")
            # ax[1,-1].stem(initial_particles[:,-1], initial_weights, basefmt="k")
            fig.tight_layout()

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
            ax[3].plot(self.t, P1_est, label='Fitted P1', color='blue', linestyle='-')
            ax[3].plot(self.t, P2_est, label='Fitted P2', color='blue', linestyle='-.')
            ax[4].plot(self.t, SI_est, label='Fitted SI', color='red', linestyle='--')
            err_ax = ax[4].twinx()
            err_ax.spines["right"]#.set_position(("outward", 60))
            # err_ax.yaxis.set_ticks_position("left")
            # err_ax.yaxis.set_label_position("left")
            err_ax.plot(self.t, np.abs((SI_est - model.last_simulation["SI"])*100/model.last_simulation["SI"]), color="purple", label="Relative error [%]")
            err_ax.set_ylabel("Relative abs. error [%]")
            err_ax.legend()
            for a in ax:
                a.legend()
            plt.show()
            fig.tight_layout()