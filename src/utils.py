import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

import icing_true

rng = np.random.default_rng()


def gen_SI_func(SI_const):
    return lambda t: SI_const


def gen_uex_func(uex_const):
    # if exp is False:
    #     return lambda t: uex_const

    # exponential ramp down
    T = 4 * 60  # repeat insulin intake every 4h
    p = 0.5 # end at half of the start
    k = - np.log(1-p) / T  # rate of min^-1  
    return lambda t: uex_const * np.exp(-k*(t % T))


def gen_D_func(D_const):
    return lambda t: D_const


def gen_PN_func(PN_const):
    # return lambda t: PN_const
    T = 4 * 60  # repeat insulin intake every 4h
    p = 0.5 # end at half of the start
    k = - np.log(1-p) / T  # rate of min^-1  
    return lambda t: PN_const * np.exp(-k*(t % T))    

def plot_weighted_kde(axis, samples, weights, bw, **plot_kwargs):
    avg = np.average(samples, weights=weights)
    var = np.average((samples - avg) ** 2, weights=weights)
    std = np.sqrt(var)
    factor = 1 / std * bw

    kde = gaussian_kde(samples, weights=weights, bw_method=factor)
    grid = np.linspace(samples.min(), samples.max(), 200)
    density = kde(grid)
    axis.plot(grid, density, **plot_kwargs)
    axis.fill_between(grid, density, alpha=0.2, **plot_kwargs)


def piecewise_constant_to_callable(values, timestamps):
    # returns a callable function for piecewise constant SI
    def f(t):
        idx = np.searchsorted(timestamps, t, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return f


def sample_hist(heights, edges, num_samples, ndims):
    if ndims < 1 or ndims > 2:
        assert ValueError("ndims must be either 1 or 2")

    if ndims == 1:
        midpoints = (edges[:-1] + edges[1:]) / 2

        cdf = np.cumsum(heights)
        cdf = cdf / cdf[-1]

        values = rng.random(num_samples)
        value_edges = np.searchsorted(cdf, values)
        samples = midpoints[value_edges][0]

    if ndims == 2:
        x_edges, y_edges = edges
        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2
        y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2

        cdf = np.cumsum(heights.flatten())
        cdf /= cdf[-1]

        values = np.random.rand(num_samples)
        value_edges = np.searchsorted(cdf, values)
        x_idx, y_idx = np.unravel_index(
            value_edges, (len(x_midpoints), len(y_midpoints))
        )
        samples = [x_midpoints[x_idx][0], y_midpoints[y_idx][0]]

    return samples


def integral_approximate_SI(G_t1, G_t0, Q, P, t):
    ICINGTrue = icing_true.ICINGTrue()
    pG = ICINGTrue.params["pG"]
    alphaG = ICINGTrue.params["alphaG"]
    EGP = ICINGTrue.params["EGP"]
    CNS = ICINGTrue.params["CNS"]
    VG = ICINGTrue.params["VG"]
    t_1 = t[-1] + 1
    t_0 = t[0]
    delta_t = t[1] - t[0]
    Qbar = Q / (1 + alphaG * Q)
    m = (G_t1 - G_t0) / (t_1 - t_0)  # Linear interpolation slope
    G = np.asarray([m * (ti - t_0) + G_t0 for ti in t])  # Interpolate BG measurements
    num_SI = 2  # Hourly SI with measurements of 120 minutes
    num_lin_eq = num_SI * 2  # 4 linear equations
    len_segment = int((t_1 - t_0) / num_lin_eq)
    a = np.zeros((num_lin_eq, num_SI))
    b = np.zeros((num_lin_eq))

    a[0, 0] = trapezoid(G[0:len_segment] * Qbar[0:len_segment], dx=delta_t)
    a[1, 0] = trapezoid(
        G[len_segment : 2 * len_segment] * Qbar[len_segment : 2 * len_segment],
        dx=delta_t,
    )
    a[2, 1] = trapezoid(
        G[2 * len_segment : 3 * len_segment] * Qbar[2 * len_segment : 3 * len_segment],
        dx=delta_t,
    )
    a[3, 1] = trapezoid(G[3 * len_segment :] * Qbar[3 * len_segment :], dx=delta_t)

    b[0] = (
        G[0]
        - G[len_segment]
        - pG * trapezoid(G[0:len_segment], dx=delta_t)
        + trapezoid((P[0:len_segment] + EGP - CNS) / VG, dx=delta_t)
    )
    b[1] = (
        G[len_segment]
        - G[2 * len_segment]
        - pG * trapezoid(G[len_segment : 2 * len_segment], dx=delta_t)
        + trapezoid((P[len_segment : 2 * len_segment] + EGP - CNS) / VG, dx=delta_t)
    )
    b[2] = (
        G[2 * len_segment]
        - G[3 * len_segment]
        - pG * trapezoid(G[2 * len_segment : 3 * len_segment], dx=delta_t)
        + trapezoid((P[2 * len_segment : 3 * len_segment] + EGP - CNS) / VG, dx=delta_t)
    )
    b[3] = (
        G[3 * len_segment]
        - G[-1]
        - pG * trapezoid(G[3 * len_segment :], dx=delta_t)
        + trapezoid((P[3 * len_segment :] + EGP - CNS) / VG, dx=delta_t)
    )

    SI = np.linalg.lstsq(a, b)[0]

    return SI


def integral_approximate_SI_seq(t, t_meas, Ts_meas, G_meas, Q_true, P_true):
    SI = []
    for idx, ti in enumerate(t_meas[:-1]):
        t_mask = (t >= t[Ts_meas * idx]) & (t < t[Ts_meas * (idx + 1)])
        times = t[t_mask]
        Q = Q_true[t_mask]
        P = P_true[t_mask]
        G_t1 = G_meas[idx + 1]
        G_t0 = G_meas[idx]
        SI.append(integral_approximate_SI(G_t1, G_t0, Q, P, times))

    SI = np.reshape(SI, shape=-1)
    return SI


def plot_model(t, t_meas, G, G_meas, Q, I, P1, P2, P, uex, PN, D, SI):  # noqa: E741
    ICINGTrue = icing_true.ICINGTrue()
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(15, 10))

    # Plot G in first subplot
    ax[0].plot(t, G, label="BG (mmol/L)")
    ax[0].plot(
        t_meas, G_meas, marker="o", linestyle="--", color="black", label="BG Meas"
    )
    ax[0].set_ylabel("BG", rotation=0, ha="right")

    # Plot I in second subplot
    ax[1].plot(t, I, label="Insulin (mU/L)", color="orange")
    ax[1].plot(
        t,
        uex,
        label="Exogenous insulin (mU/min)",
        color="orange",
        linestyle="--",
    )
    ax[1].set_ylabel("Insulin", rotation=0, ha="right")

    # Plot Q in third subplot
    ax[2].plot(
        t,
        Q,
        label="Interstitial Insulin Q (mU/L)",
        color="purple",
    )
    effective_interstitial_insulin = Q / (1.0 + ICINGTrue.params["alphaG"] * Q)
    ax[2].plot(
        t,
        effective_interstitial_insulin,
        label="Effective Interstitial Q/(1+aG*Q)",
        color="purple",
        linestyle="--",
    )
    ax[2].set_ylabel("Interstitial Insulin", rotation=0, ha="right")

    ax[3].plot(
        t,
        P,
        label="Total Nutrition Input (mmol/min)",
        color="green",
    )
    ax[3].plot(
        t,
        PN,
        label="Parenteral Nutrition (mmol/min)",
        color="green",
        linestyle="--",
    )
    ax[3].plot(
        t,
        D,
        label="Enteral Nutrition D (mmol/min)",
        color="green",
        linestyle=":",
    )
    ax[3].set_ylabel("Nutrition Input", rotation=0, ha="right")
    # Plot SI in fifth subplot
    ax[4].plot(t, SI, label="SI (L/mU/min)", color="red")
    ax[4].set_ylabel("SI", rotation=0, ha="right")
    ax[4].set_xlabel("Time (min)")

    for a in ax:
        a.grid(True)
        a.legend(loc="best")
