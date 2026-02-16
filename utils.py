import numpy as np   
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid

def gen_SI_func(SI_const):
    return lambda t: SI_const

def gen_uex_func():
    return lambda t: 75*np.exp(-(np.log(2)/(5*60))*((t+120) % (5*60)))

def gen_D_func(D_const):
    return lambda t: D_const

def gen_PN_func():
    return lambda t: np.exp(-(np.log(2)/(5*60))*(t % (5*60)))

def plot_weighted_kde(axis, samples, weights, bw, **plot_kwargs):
    avg = np.average(samples, weights=weights)
    var = np.average((samples - avg) ** 2, weights=weights)
    std = np.sqrt(var)
    factor = 1/std * bw

    kde = gaussian_kde(samples, weights=weights, bw_method=factor)
    grid = np.linspace(samples.min(), samples.max(), 200)
    density = kde(grid)
    axis.plot(grid, density, **plot_kwargs)
    axis.fill_between(grid, density, alpha=0.2, **plot_kwargs)

def piecewise_constant_to_callable(values, timestamps):
    # returns a callable function for piecewise constant SI
    def f(t):
        idx = np.searchsorted(timestamps, t, side='right') - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]
    return f

def integral_approximate_SI(G_t1, G_t0, Q, P, t, pG, alphaG, EGP, CNS, VG):
    t_1 = t[-1]
    t_0 = t[0]
    delta_t = t[1] - t[0]
    Qbar = Q/(1 + alphaG*Q)
    m = (G_t1 - G_t0)/(t_1 - t_0)
    G = [m*(ti - t_0) + G_t0 for ti in t]
    SI = (G_t0 - G_t1 - pG*trapezoid(G,dx=delta_t) + (1/VG)*trapezoid(P+(EGP-CNS),dx=delta_t)) / trapezoid(G*Qbar,dx=delta_t)
    return SI