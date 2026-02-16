import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# class ICING model
class ICINGTrue:
    def __init__(self, params = None):
        if params:
            self.params = params
        else: #set default params
            self.params = {
                "pG": 0.006,
                "alphaG": 1.0/65.0,
                "VG": 13.3,
                "EGP": 1.16,
                "CNS": 0.3,
                "d1": -np.log(0.5)/20.0,
                "d2": -np.log(0.5)/100.0,
                "Pmax": 6.11,
                "xL": 0.67,
                "nL": 0.1578, 
                "alphaI": 0.0017,
                "nK": 0.0542,
                "nC": 0.0075,
                "nI": 0.0075,
                "k1": 45.7,
                "k2": 1.5,
                "k3": 1000,
                "umin": 16.7,
                "umax": 266.7,
                "VI": 4.0
            }
         
    def __icing_odes(self, t, y, uex_func, PN_func, D_func, SI_func):
        # Unpack input state
        G, I, Q, P1, P2 = y  

        # Time-varying inputs: assume all are callable
        uex = uex_func(t)
        PN = PN_func(t)
        D = D_func(t)

        # SI at time t
        SI = SI_func(t) 

        # Nutrition input
        P_from_gut = min(self.params["d2"] * P2, self.params["Pmax"])
        P_t = P_from_gut + PN

        # Endogenous insulin secretion
        uen = self.params["k1"] * np.exp(-I*self.params["k2"]/self.params["k3"])

        # BG dynamics
        insulin_effect = SI * G * Q / (1.0 + self.params["alphaG"] * Q)

        dGdt = - self.params["pG"] * G - insulin_effect + (P_t + self.params["EGP"] - self.params["CNS"]) / self.params["VG"]

        dQdt = self.params["nI"] * (I - Q) - self.params["nC"] * Q / (1.0 + self.params["alphaG"] * Q)
        
        dIdt = - self.params["nK"] * I - self.params["nL"] * I / (1.0 + self.params["alphaI"] * I) - self.params["nI"] * (I - Q) \
            + uex / self.params["VI"] + (1.0 - self.params["xL"]) * uen / self.params["VI"]
        
        dP1dt = - self.params["d1"] * P1 + D
        
        dP2dt = - min(self.params["d2"] * P2, self.params["Pmax"]) + self.params["d1"] * P1

        # Return derivatives
        return [dGdt, dIdt, dQdt, dP1dt, dP2dt]

    def simulate(self, y0, t_start, t_end, t_eval, uex_func, PN_func, D_func, SI_func):

        # Simulate the ODEs with high precision using RK45 
        sol = solve_ivp(self.__icing_odes, (t_start, t_end), y0, args=(uex_func, PN_func, D_func, SI_func), t_eval=t_eval, rtol=1e-6, atol=1e-9)
        
        # Save this for later plotting
        self.last_simulation = {"t": sol.t, "G": sol.y[0], "I": sol.y[1], "Q": sol.y[2], "P1": sol.y[3], "P2": sol.y[4]}

        # Also save input functions evaluated at times t_eval
        self.last_simulation["uex"] = np.array([uex_func(ti) for ti in sol.t])
        self.last_simulation["PN"] = np.array([PN_func(ti) for ti in sol.t])
        self.last_simulation["D"] = np.array([D_func(ti) for ti in sol.t])
        self.last_simulation["SI"] = np.array([SI_func(ti) for ti in sol.t])
        P = [min(self.params["d2"]*sol.y[4][idx],self.params["Pmax"]) + PN_func(ti) for idx,ti in enumerate(sol.t)]
        P = np.asarray(P)

        return [sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], P] #G, I, Q, P1, P2, P
    
    def plot_sim(self, simulation=None):
        if not simulation:
            simulation = self.last_simulation
        
        fig, ax = plt.subplots(5,1, sharex=True, figsize=(15,10))

        # Plot G in first subplot
        ax[0].plot(simulation["t"], simulation["G"], label='BG (mmol/L)')
        ax[0].set_ylabel('BG', rotation=0, ha="right")

        # Plot I in second subplot
        ax[1].plot(simulation["t"], simulation["I"], label='Insulin (mU/L)', color='orange')
        ax[1].plot(simulation["t"], simulation["uex"], label='Exogenous insulin (mU/min)', color='orange', linestyle='--')
        ax[1].set_ylabel('Insulin', rotation=0, ha="right")

        # Plot Q in third subplot
        ax[2].plot(simulation["t"], simulation["Q"], label='Interstitial Insulin Q (mU/L)', color='purple')
        effective_interstitial_insulin = simulation["Q"] / (1.0 + self.params["alphaG"] * simulation["Q"])
        ax[2].plot(simulation["t"], effective_interstitial_insulin, label='Effective Interstitial Q/(1+aG*Q)', color='purple', linestyle='--')
        ax[2].set_ylabel('Interstitial Insulin', rotation=0, ha="right")

        # Plot nutrition in fourth subplot
        P_from_gut = np.minimum(self.params["d2"] * simulation["P2"], self.params["Pmax"])
        P_t = P_from_gut + simulation["PN"]
        ax[3].plot(simulation["t"], P_t, label='Total Nutrition Input (mmol/min)', color='green')
        ax[3].plot(simulation["t"], simulation["PN"], label='Parenteral Nutrition (mmol/min)', color='green', linestyle='--')
        ax[3].plot(simulation["t"], simulation["D"], label='Enteral Nutrition D (mmol/min)', color='green', linestyle=':')
        ax[3].set_ylabel('Nutrition Input', rotation=0, ha="right")
        # Plot SI in fifth subplot
        ax[4].plot(simulation["t"], simulation["SI"], label='SI (L/mU/min)', color='red')
        ax[4].set_ylabel('SI', rotation=0, ha="right")
        ax[4].set_xlabel('Time (min)')

        for a in ax:
            a.grid(True)
            #a.legend(loc='best')
        
        # Remove hspace
        # fig.subplots_adjust(hspace=0)

        return fig, ax
