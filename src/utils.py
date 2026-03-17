import numpy as np   
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid

def gen_SI_func():
    return lambda t: 2e-4

def gen_uex_func(type):
    if type == "exp":
        func = lambda t: 75*np.exp(-(np.log(2)/(5*60))*((t+120) % (5*60)))
    elif type == "constant":
        func = lambda t: 65
    return func
         
def gen_D_func():
    return lambda t: 0.24

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
    t_1 = t[-1] + 1
    t_0 = t[0]
    delta_t = t[1] - t[0]
    Qbar = Q/(1 + alphaG*Q)
    m = (G_t1 - G_t0)/(t_1 - t_0) # Linear interpolation slope
    G = [m*(ti - t_0) + G_t0 for ti in t] # Interpolate BG measurements
    num_SI = 2 # Hourly SI with measurements of 120 minutes
    num_lin_eq = num_SI*2 # 4 linear equations
    len_segment = int((t_1-t_0) / num_lin_eq)
    a = np.zeros((num_lin_eq, num_SI))
    b = np.zeros((num_lin_eq))

    a[0,0] = trapezoid(G[0:len_segment]*Qbar[0:len_segment],dx=delta_t)
    a[1,0] = trapezoid(G[len_segment:2*len_segment]*Qbar[len_segment:2*len_segment],dx=delta_t)
    a[2,1] = trapezoid(G[2*len_segment:3*len_segment]*Qbar[2*len_segment:3*len_segment],dx=delta_t)
    a[3,1] = trapezoid(G[3*len_segment:]*Qbar[3*len_segment:],dx=delta_t)

    b[0] = G[0] - G[len_segment] - pG*trapezoid(G[0:len_segment],dx=delta_t) + trapezoid((P[0:len_segment]+EGP-CNS)/VG,dx=delta_t)
    b[1] = G[len_segment] - G[2*len_segment] - pG*trapezoid(G[len_segment:2*len_segment],dx=delta_t) + trapezoid((P[len_segment:2*len_segment]+EGP-CNS)/VG,dx=delta_t)
    b[2] = G[2*len_segment] - G[3*len_segment] - pG*trapezoid(G[2*len_segment:3*len_segment],dx=delta_t) + trapezoid((P[2*len_segment:3*len_segment]+EGP-CNS)/VG,dx=delta_t)
    b[3] = G[3*len_segment] - G[-1] - pG*trapezoid(G[3*len_segment:],dx=delta_t) + trapezoid((P[3*len_segment:]+EGP-CNS)/VG,dx=delta_t)

    SI =  np.linalg.lstsq(a,b)[0] 
    return SI