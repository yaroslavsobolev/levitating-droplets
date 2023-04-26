import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

experimental_data = np.loadtxt('misc_data/experimental_stability/Droplet instability_clean.txt', delimiter='\t', skiprows=1)
experimental_data_raw = np.loadtxt('misc_data/experimental_stability/droplet-instability-raw.csv', delimiter=',', skiprows=2)
# Convert from angular velocity in rpm to linear velocity in mm/s
experimental_data_raw[:, 1] *= 1.68 * 3.14 * 35 / 60
experimental_data_raw[:, 2] *= 0.365 * 3.14 * 23 / 60 # for slow rotation, more smooth motion was achieved by a different transmission belt

# Experimental data is done for water-surfactant mixtyre. Density ~1050 kg/m^3, surface tension 25 mN/m, and high viscosity.
density_of_luquid = 1.05 #grams per cubic centimeter
sigma = 25e-3 # surface tension in N/m
capillary_length = 0.0015579 #m
mu_air = 1.81e-5 # Pa*s
cap_num_coeff = 0.000724 # gives Capillary number when multiplied by speed in m/s
density_of_air = 1.225 # kg/m^3

Oh_master = np.sqrt(mu_air ** 2 / (density_of_air * capillary_length * sigma)) # ohnesorge number

experimental_data_dimensionless = np.copy(experimental_data)
experimental_data_dimensionless[:, 0] *= 1e-9 / (4 / 3 * np.pi * capillary_length ** 3) # 1e-9 factor is for converting from uL to m^3
experimental_data_dimensionless[:, 1:] *= 1e-3 * cap_num_coeff # 1e-3 factor is for converting from mm/s to m/s

expeirmental_data_raw_dimensionless = np.copy(experimental_data_raw)
expeirmental_data_raw_dimensionless[:, 0] *= 1e-9 / (4 / 3 * np.pi * capillary_length ** 3) # 1e-9 factor is for converting from uL to m^3
expeirmental_data_raw_dimensionless[:, 1:] *= 1e-3 * cap_num_coeff # 1e-3 factor is for converting from mm/s to m/s

reduced_volume_maximal = np.max(experimental_data_dimensionless[:, 0])
reduced_volume_minimal = np.min(experimental_data_dimensionless[:, 0])

color_of_unstable_zone = 'goldenrod'
fig, ax = plt.subplots(figsize=(3.7, 3.3), dpi=300)
# plt.errorbar(x=experimental_data_dimensionless[:,0], y=experimental_data_dimensionless[:,1], yerr=experimental_data_dimensionless[:,2],
#              capsize=5, linestyle='', marker='o', color='grey')
# plt.errorbar(x=experimental_data_dimensionless[:,0], y=experimental_data_dimensionless[:,3], yerr=experimental_data_dimensionless[:,4],
#              capsize=5, linestyle='', marker='o', color='grey')
plt.fill_between(x=experimental_data_dimensionless[:, 0], y1=np.zeros_like(experimental_data_dimensionless[:, 0]),
                 y2=experimental_data_dimensionless[:, 3], color=color_of_unstable_zone, alpha=0.3)
plt.fill_between(x=experimental_data_dimensionless[:, 0], y1=experimental_data_dimensionless[:, 3], y2=experimental_data_dimensionless[:, 1],
                 color='C0', alpha=0.4)
plt.fill_between(x=experimental_data_dimensionless[:, 0], y1=experimental_data_dimensionless[:, 1],
                 y2=0.08*np.ones_like(experimental_data_dimensionless[:, 0]), color=color_of_unstable_zone, alpha=0.3)
plt.scatter(expeirmental_data_raw_dimensionless[:, 0], expeirmental_data_raw_dimensionless[:, 1], color='black', marker='x', s=20, alpha=0.75, linewidth=0.8)
plt.scatter(expeirmental_data_raw_dimensionless[:, 0], expeirmental_data_raw_dimensionless[:, 2], color='C2', marker='x', s=20, alpha=0.75, linewidth=0.8)
plt.annotate('Stable levitation', color='C0', xy=(2.7, 0.00065), fontsize=16, alpha=0.6, ha='center')

# Limit due to wall climbing by drag
reduced_volumes = np.logspace(np.log10(reduced_volume_minimal), np.log10(reduced_volume_maximal), 100)
def cd_factor(reduced_volumes, CaOhm2):
    """
    Computed drag coefficient C_d for a spherical droplet

    :param reduced_volumes: float
        Reduced volume of the droplet
    :param CaOhm2: float
        Ca * Oh ** (-2), where Ca is the capillary number, and Oh is the Ohnesorge number, which is the
        ratio of viscous forces to surface tension forces and is a function of capillary number
    :return: float
        Drag coefficient
    """
    Re = 2 * CaOhm2 * (reduced_volumes) ** (1 / 3)
    return 1 + 1/6*(Re)**(2/3)

# more rigorous solving for Ca_max
def func(Ca, Oh, vred):
    cd_f = cd_factor(vred, Ca/((Oh)**2))
    return Ca - (1/cd_f)*2.0/9*(vred)**(2/3)
capillary_numbers = []
for v in reduced_volumes:
    r = root(func, x0=1e-2, args=(Oh_master, v))
    capillary_numbers.append(r.x)

plt.plot(reduced_volumes, capillary_numbers, '--', color='black')

#limit due to very small gap under the droplet
G = 4
mingap = 850e-9
k_bs = 1 / capillary_length * np.sqrt(G * (reduced_volumes) ** (-2 / 3) + 4)
F = 0.42

# for large droplet limit
capillary_number_lowlimit_for_large_droplets = (k_bs * mingap / (0.716 * 2.123 * F)) ** (1.5)
plt.plot(reduced_volumes, capillary_number_lowlimit_for_large_droplets, '--', color='C2')

# # for small droplet limit
capillary_number_lowlimit_for_small_droplets = (k_bs * mingap / (0.871 * F)) ** (5 / 4) * reduced_volumes ** (1 / 3)
# plt.plot(reduced_volumes, capillary_number_lowlimit_for_large_droplets, '--', color='C3')

plt.xscale('log')
plt.yscale('log')
plt.xlim(reduced_volume_minimal / 1.1, reduced_volume_maximal * 1.1)
plt.ylim(5e-5, 0.019)
plt.ylabel('Capillary number $\mu v / \sigma$')
plt.xlabel('Droplet volume, unitless $\\hat{V}=3V (4 \pi a^3)^{-1}$')
ax_volume = ax.twiny()
ax_volume.set_xlim(np.min(experimental_data[:, 0]), np.max(experimental_data[:, 0]))
ax_volume.set_xscale('log')
ax_volume.set_xlabel('Droplet volume $V$, μL')

### linear speed on the right axis
ax_linspeed = ax.twinx()
ax_linspeed.set_ylim(5e-5 / (1e-3 * cap_num_coeff) / 1000,
                     0.019 / (1e-3 * cap_num_coeff) / 1000)
ax_linspeed.set_yscale('log')
ax_linspeed.set_ylabel('Linear speed $v$, m/s')

# ### Angular velocity on the right axis
# ax_linspeed = ax.twinx()
# ax_linspeed.set_ylim(5e-5 / (1e-3 * cap_num_coeff) / (2 * np.pi * (35)/2) * 60 /1000,
#                      0.019 / (1e-3 * cap_num_coeff) / (2 * np.pi * (35)/2) * 60 /1000)
# ax_linspeed.set_yscale('log')
# ax_linspeed.set_ylabel('Angular velocity, 1000 r.p.m.')

plt.tight_layout()
# simpleaxis(ax)
fig.savefig('figures/stability-phase-diagram-1.png', dpi=600)
plt.show()