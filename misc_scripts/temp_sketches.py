import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import root
from matplotlib.ticker import FuncFormatter

ensity_of_luquid = 1.05 #grams per cubic centimeter
sigma = 25e-3 # N/m
cap_length = 0.0015579 #m
mu_air = 1.81e-5 # dynamic viscosity of air in Pa * s
epsilon = 8.85e-12 # vacuum permittivity
cap_num_coeff = mu_air/sigma # gives Capillary number when multiplied by speed in m/s
droplet_volume = 25e-6 * (1e-1) ** 3 # 25 microliters. Volume in m**3
volume_reduced = droplet_volume * 3 / (4 * np.pi * (cap_length ** 3)) # dimensionless volume (see paper)
kappa_b = 2 / cap_length * np.sqrt(1 + volume_reduced ** (-2 / 3) ) # curvature in inverse meters (see paper)
h0 = 1.3e-6 / 0.716 #3e-6 # Critical gap, meters

def cap_num(v):
    return cap_num_coeff * v

def gap_for_gap(U, h0, v):
    return (0.102 + 0.538 * np.exp( -0.576 * epsilon * U**2 / (sigma * h0) * (6 * cap_num(v))**(-2/3) )) * \
           (1 / kappa_b) * (6 * cap_num(v))**(2/3)



def plot_illustration_of_stability():
    v = 2 # speed, m/s
    Us = np.linspace(0, 50, 6)
    h0s = np.linspace(1e-7, 17e-6, 100)

    fig, ax = plt.subplots(figsize=(5,4.75))
    colors = mpl.cm.viridis(np.linspace(0,1,Us.shape[0]))
    to_microns = 1e6
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

# plot_illustration_of_stability()
to_microns = 1e6
fig, ax = plt.subplots(figsize=(5, 4.75))

def plot_gaps_for_given_v(v, Umin, Umax, Nsteps):
    # v = 2  # speed, m/s
    Us = np.linspace(Umin, Umax, Nsteps)
    # h0s = np.linspace(1e-7, 17e-6, 100)
    # colors = mpl.cm.viridis(np.linspace(0, 1, Us.shape[0]))

    def func(h0, U, v):
        return gap_for_gap(U, h0, v) - h0

    return Us, np.array([root(func, x0=2e-5, args=(U, v)).x[0] for U in Us])

Us, h0s = plot_gaps_for_given_v(v=2, Umin=0, Umax=50, Nsteps=6)
plt.scatter(Us, h0s * to_microns, color='k')

# Us, h0s = plot_gaps_for_given_v(v=2, Umin=0, Umax=50, Nsteps=200)
# plt.plot(Us, h0s * to_microns, color='k')

vs = [0.5, 0.8, 1.0, 1.2, 1.5]#, 2.0]
vs = np.linspace(0.5, 3, 6)
colors = mpl.cm.plasma(np.linspace(0, 1, vs.shape[0]))
for i, v in enumerate(vs):
    Us, h0s = plot_gaps_for_given_v(v=v, Umin=0, Umax=50, Nsteps=200)
    plt.plot(Us, h0s * to_microns, color=colors[i])

# plt.xlim(0, h0s[-1] * to_microns)
plt.ylim(0, 17)
plt.xlabel('Potential $U$ applied to the droplet, V')
plt.ylabel('Equilibrium gap $h_{0}$ (root of equatiom (22)), $\mu$m')
plt.tight_layout()
fig.savefig('figures/analytical_gap-vs-voltage_illustration_2.png', dpi=300)
plt.show()

plt.show()
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