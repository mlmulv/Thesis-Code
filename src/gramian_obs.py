import numpy as np
import icing_true
import utils

class EmperialGramianMatrix:
    def __init__(self, x0, n_pert, c, x_max, t, dt, uex_func, PN_func, D_func, SI_func, threshold, SI_est = None):
        self.x0 = x0
        self.n_pert = n_pert
        self.i_max = 2 ** n_pert
        self.S = np.diag(x_max)
        self.t = t
        self.dt = dt
        self.c = c
        self.uex_func = uex_func
        self.PN_func = PN_func
        self.D_func = D_func
        self.SI_func = SI_func
        self.threshold = threshold
        self.SI_est = SI_est

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

    def outputs(self, x0_i, x0):
        y_i = np.zeros((self.t.shape[0], self.i_max))
        ICINGTrue = icing_true.ICINGTrue()

        if self.SI_est is None:
            self.y_0_k =  icing_true.ICINGTrue().simulate(x0, self.t[0], self.t[-1], self.t, self.uex_func, self.PN_func, self.D_func, self.SI_func)[0]

            for i in range(self.i_max):
                y_i[:,i] = ICINGTrue.simulate(x0_i[:,i], self.t[0], self.t[-1], self.t, self.uex_func, self.PN_func, self.D_func, self.SI_func)[0]
        else:
            self.y_0_k =  icing_true.ICINGTrue().simulate(x0[:-1], self.t[0], self.t[-1], self.t, self.uex_func, self.PN_func, self.D_func, self.SI_func)[0]

            for i in range(self.i_max):
                SI_func = utils.gen_SI_func(SI_const=x0_i[-1,i])
                y_i[:,i] = ICINGTrue.simulate(x0_i[:-1,i], self.t[0], self.t[-1], self.t, self.uex_func, self.PN_func, self.D_func, SI_func)[0]

        return y_i

    def phi_matrix(self, y_i_k, k):
        phi = np.outer(y_i_k-self.y_0_k[k], y_i_k-self.y_0_k[k])

        return phi

    def gramian(self):
        T = self.full_factorial_vectors()
        x0_i = self.initial_condition(T)
        y_i = self.outputs(x0_i, self.x0)
        k_max = len(self.t)

        summation = 0
        for i in range(k_max):
            phi = self.phi_matrix(y_i[i,:],i)
            summation += np.matmul(T, np.matmul(phi, T.T))

        cs_inv = np.linalg.inv(self.c * self.S)
        W_O = np.matmul(cs_inv, np.matmul(summation, self.dt*cs_inv))

        x_s = np.sqrt(np.diag(W_O))
        T_s = np.linalg.inv(np.diag(x_s))
        T_bar = np.matmul(T_s, T)
        x0_bar = np.matmul(T_s, self.x0)
        S_bar = np.matmul(T_s, self.S)
        cs_inv = np.linalg.inv(self.c * S_bar)

        x_bar_i = np.matmul(T_s, x0_i)
        y_bar_i = self.outputs(x_bar_i, x0_bar)

        summation = 0
        for i in range(k_max):
            phi = self.phi_matrix(y_bar_i[i,:],i)
            summation += np.matmul(T_bar, np.matmul(phi, T_bar.T))

        W_O = np.matmul(cs_inv, np.matmul(summation, self.dt*cs_inv))
        W_O = 0.5 * (W_O + W_O.T)
        eig_vals, eig_vectors = np.linalg.eigh(W_O)
        return W_O, eig_vals, eig_vectors

