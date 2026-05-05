import numpy as np
import icing_true

class EmperialGramianMatrix:
    def __init__(self, x0, n_pert, c, x_max, t, dt_meas, t_meas, uex_func, PN_func, D_func, SI_func):
        self.x0 = x0
        self.n_pert = n_pert
        self.i_max = 2 ** n_pert
        self.S = np.diag(x_max)
        self.t = t
        self.c = c
        self.dt_meas = dt_meas
        self.t_meas = t_meas
        self.uex_func = uex_func
        self.PN_func = PN_func
        self.D_func = D_func
        self.SI_func = SI_func

    def full_factorial_vectors(self):
        ints = np.arange(self.i_max, dtype=np.uint32)
        bits = ((ints[:, None] >> np.arange(self.n_pert)) & 1).astype(np.int8)
        signs = 2*bits - 1
        return (signs.astype(np.float64) / np.sqrt(self.i_max)).T

    def initial_condition(self, T):
        x0_i = np.zeros((self.x0.shape[0],self.i_max))

        for i in range(self.i_max):
            x0_i[:,i] = self.x0 + self.c * np.matmul(self.S, T[:,i].T)

        return x0_i

    def outputs(self, x0_i):
        y_i = np.zeros((self.t_meas.shape[0], self.i_max))
        ICINGTrue = icing_true.ICINGTrue()
        self.y_0_k =  icing_true.ICINGTrue().simulate(self.x0, self.t[0], self.t[-1], self.t, self.uex_func, self.PN_func, self.D_func, self.SI_func)[0][::self.dt_meas]

        for i in range(self.i_max):
            y_i[:,i] = ICINGTrue.simulate(x0_i[:,i], self.t[0], self.t[-1], self.t, self.uex_func, self.PN_func, self.D_func, self.SI_func)[0][::self.dt_meas]

        return y_i

    def phi_value(self, y_i_k, k):
        phi = np.zeros((self.i_max, self.i_max))

        for i in range(self.i_max):
            j = self.i_max-1-i
            phi[i,j] = (y_i_k[i] - self.y_0_k[k])*(y_i_k[j] - self.y_0_k[k])

        return phi

    def gramian(self):
        T = self.full_factorial_vectors()
        x0_i = self.initial_condition(T)
        y_i = self.outputs(x0_i)
        k_max = len(self.t_meas)

        summation = 0
        for i in range(k_max):
            phi = self.phi_value(y_i[i,:],i)
            summation += np.matmul(T, np.matmul(phi, T.T))

        cs_inv = np.linalg.inv(self.c * self.S)
        W_O = np.matmul(cs_inv, np.matmul(summation, self.dt_meas*cs_inv))

        return W_O


