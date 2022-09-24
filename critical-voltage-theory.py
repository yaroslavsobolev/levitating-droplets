import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import root
from matplotlib.ticker import FuncFormatter

density_of_luquid = 1.05 #grams per cubic centimeter
sigma = 25e-3 # N/m
cap_length = 0.0015579 #m
mu_air = 1.81e-5 # dynamic viscosity of air in Pa * s
epsilon = 8.85e-12 # vacuum permittivity
cap_num_coeff = mu_air/sigma # gives Capillary number when multiplied by speed in m/s
droplet_volume = 25e-6 * (1e-1) ** 3 # 25 microliters. Volume in m**3
volume_reduced = droplet_volume * 3 / (4 * np.pi * (cap_length ** 3)) # dimensionless volume (see paper)
kappa_b = 2 / cap_length * np.sqrt(1 + volume_reduced ** (-2 / 3) ) # curvature in inverse meters (see paper)
h0 = 1.3e-6 / 0.716 #3e-6 # Critical gap, meters
to_microns = 1e6

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def cap_num(v):
    return cap_num_coeff * v


def gap_for_gap(U, h0, v):
    return (0.102 + 0.538 * np.exp( -0.576 * epsilon * U**2 / (sigma * h0) * (6 * cap_num(v))**(-2/3) )) * \
           (1 / kappa_b) * (6 * cap_num(v))**(2/3)


def plot_illustration_of_stability():
    to_microns = 1e6
    v = 2 # speed, m/s
    Us = np.linspace(0, 50, 6)
    h0s = np.linspace(1e-7, 17e-6, 100)

    fig, ax = plt.subplots(figsize=(5,4.75))
    simpleaxis(ax)
    colors = mpl.cm.viridis(np.linspace(0,1,Us.shape[0]))
    for i,U in enumerate(Us):
        plt.plot(h0s * to_microns, gap_for_gap(U, h0s, v) * to_microns, label=f'U={U}', color=colors[i])

    plt.plot([0, h0s[-1] * to_microns], [0, h0s[-1] * to_microns], '--', color='C1')

    def func(h0, U, v):
        return gap_for_gap(U, h0, v) - h0

    to_microns = 1e6
    roots = [to_microns * root(func, x0=2e-5, args=(U, v)).x[0] for U in Us]
    plt.scatter(roots, roots, color='k')

    plt.xlim(0, h0s[-1] * to_microns)
    plt.ylim(0, h0s[-1] * to_microns)
    plt.xlabel('Left-hand side of equation (22), $\mu$m')
    plt.ylabel('Right-hand side of equation (22), $\mu$m')
    plt.tight_layout()
    fig.savefig('figures/equation_22_illustration.png', dpi=300)
    plt.show()

    ### Plot the colormap
    fig, ax = plt.subplots(figsize=(2.7, 0.9))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.viridis
    bounds = list(Us)
    bounds.append(Us[-1] + (Us[-1] - Us[-2]))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fmt = lambda x, pos: '{:.0f}'.format(x)
    cax = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 format=FuncFormatter(fmt),
                 cax=ax, orientation='horizontal',
                 label="Potential U applied to the droplet, V")
    cax.set_ticks(np.array(bounds) + (Us[-1] - Us[-2])/2)
    cax.set_ticklabels(['{0:.0f}'.format(x) for x in Us])
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.tight_layout()
    fig.savefig('figures/equation_22_illustration_colorbar.png', dpi=300)
    plt.show()


def plot_gaps_for_given_v(v, Umin, Umax, Nsteps):
    Us = np.linspace(Umin, Umax, Nsteps)
    def func(h0, U, v):
        return gap_for_gap(U, h0, v) - h0
    return Us, np.array([root(func, x0=2e-5, args=(U, v)).x[0] for U in Us])

def mingap_for_given_U_and_v(U, v):
    def func(h0, U, v):
        return gap_for_gap(U, h0, v) - h0
    h0_here = root(func, x0=3e-6, args=(U, v)).x[0]
    Ws = epsilon * U ** 2 / (sigma * h0_here) * (6 * cap_num(v)) ** (-2 / 3)
    hmin = h0_here * Hmin(Ws)
    return hmin


def plot_illustration_of_gap_vs_U():
    fig, ax = plt.subplots(figsize=(5, 4.75))
    simpleaxis(ax)
    Us, h0s = plot_gaps_for_given_v(v=2, Umin=0, Umax=50, Nsteps=6)
    plt.scatter(Us, h0s * to_microns, color='k')
    vs = [0.5, 0.8, 1.0, 1.2, 1.5]#, 2.0]
    vs = np.linspace(0.5, 3, 6)
    colors = mpl.cm.plasma(np.linspace(0, 1, vs.shape[0]))
    for i, v in enumerate(vs):
        Us, h0s = plot_gaps_for_given_v(v=v, Umin=0, Umax=50, Nsteps=200)
        plt.plot(Us, h0s * to_microns, color=colors[i])
    plt.ylim(0, 17)
    plt.xlabel('Potential $U$ applied to the droplet, V')
    plt.ylabel('Equilibrium gap $h_{0}$ (root of equatiom (22)), $\mu$m')
    plt.tight_layout()
    fig.savefig('figures/analytical_gap-vs-voltage_illustration_2.png', dpi=300)
    plt.show()


def Hmin(W):
    return 0.304 * np.exp(-0.471 * W) + 0.421


def plot_illustration_of_gap_min_vs_U():
    fig, ax = plt.subplots(figsize=(5, 4.75))
    simpleaxis(ax)
    # Us, h0s = plot_gaps_for_given_v(v=2, Umin=0, Umax=50, Nsteps=6)
    # plt.scatter(Us, h0s * to_microns, color='k')
    # vs = [0.5, 0.8, 1.0, 1.2, 1.5]#, 2.0]
    vs = np.linspace(0.5, 3, 6)
    colors = mpl.cm.plasma(np.linspace(0, 1, vs.shape[0]))
    for i, v in enumerate(vs):
        Us, h0s = plot_gaps_for_given_v(v=v, Umin=0, Umax=50, Nsteps=200)
        Ws = epsilon * Us**2 / (sigma * h0s) * (6 * cap_num(v))**(-2/3)
        hmins = h0s * Hmin(Ws)
        plt.plot(Us, hmins * to_microns, color=colors[i])

    plt.axhline(y=1.33, linestyle='--', color='black')
    plt.ylim(0, 17)
    plt.xlabel('Potential $U$ applied to the droplet, V')
    plt.ylabel('Minimum gap $h_{min}$, $\mu$m')
    plt.tight_layout()
    fig.savefig('figures/analytical-gap-minimum-vs-voltage_illustration_2.png', dpi=300)
    plt.show()

# plot_illustration_of_stability()
# plot_illustration_of_gap_vs_U()
# plot_illustration_of_gap_min_vs_U()

# ### Version where h0 must be below some threshold h_critical
#
# def func(U, h_critical, v):
#     return gap_for_gap(U, h_critical, v) - h_critical
# # v = 1.5 # m/s
# h_critical = 3.3e-6
# vs = np.linspace(0.3, 2.5, 20)
# Us_critical = np.array([root(func, x0=20, args=(h_critical, v)).x[0] for v in vs])
# plt.plot(vs, Us_critical)
# plt.plot(vs, 28 * vs**(2/3))
#
# # h_critical =
# fraction = 0.4
# Us_critical = np.array([root(func, x0=20,
#                              args=(gap_for_gap(U=0, h0=1e-6, v=fraction * v),
#                                                 v)).x[0]
#                         for v in vs])
# plt.plot(vs, Us_critical)
#
# plt.show()

### Version where hmin must be below some critical threshold
def func(U, h_critical, v):
    return mingap_for_given_U_and_v(U, v) - h_critical
# v = 1.5 # m/s
h_critical = 1.35e-6
vs = np.linspace(0.5, 2.5, 20)
Us_critical = np.array([root(func, x0=20, args=(h_critical, v)).x[0] for v in vs])
plt.plot(vs, Us_critical)
# plt.plot(vs, 28 * vs**(2/3))
plt.show()

# plot_illustration_of_stability()

# for i, U in enumerate(Us):
#     h0_solution =
#     print()
# print(U_solution)
    # plt.plot(h0s * to_microns, gap_for_gap(U, h0s, v) * to_microns, label=f'U={U}', color=colors[i])
#
# plt.plot([0, h0s[-1] * to_microns], [0, h0s[-1] * to_microns], '--', color='C1')
#
# plt.xlim(0, h0s[-1] * to_microns)
# plt.ylim(0, h0s[-1] * to_microns)
# plt.xlabel('Left-hand side of equation (22), $\mu$m')
# plt.ylabel('Right-hand side of equation (22), $\mu$m')
# plt.tight_layout()
# fig.savefig('figures/equation_22_illustration.png', dpi=300)
# plt.show()

# v = 1 # speed, m/s
# Us = np.linspace(0, 100, 100)
# for h0 in np.linspace(1e-6, 2e-6, 10):
#     plt.plot(Us, func(Us, h0, v), label=f'h0={h0}')
# plt.legend()
# plt.axhline(y=0, color='black')
# plt.show()
#
# U_solution = root(func, x0=30, args=(h0, v))
# print(U_solution)