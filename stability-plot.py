import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

exp_data = np.loadtxt('misc_data/experimental_stability/Droplet instability_clean.txt', delimiter='\t', skiprows=1)

# Experimental data is done for water-surfactant mixtyre. Density ~1050 kg/m^3, surface tension 25 mN/m, and high viscosity.
density_of_luquid = 1.05 #grams per cubic centimeter
sigma = 25e-3 # N/m
cap_length = 0.0015579 #m
mu_air = 1.81e-5 # Pa*s
cap_num_coeff = 0.000724 # gives Capillary number when multiplied by speed in m/s
density_of_air = 1.225 # kg/m^3

Oh_master = np.sqrt(mu_air**2/(density_of_air * cap_length * sigma)) # ohnesorge number

exp_data_dimensionless = np.copy(exp_data)
exp_data_dimensionless[:,0] *= 1e-9/(4/3*np.pi*cap_length**3) # 1e-9 factor is for converting from uL to m^3
exp_data_dimensionless[:,1:] *= 1e-3*cap_num_coeff # 1e-3 factor is for converting from mm/s to m/s

vred_max = np.max(exp_data_dimensionless[:,0])
vred_min = np.min(exp_data_dimensionless[:,0])

unstable_color = 'goldenrod'

fig, ax = plt.subplots(figsize=(3.5, 3.3), dpi=300)
plt.errorbar(x=exp_data_dimensionless[:,0], y=exp_data_dimensionless[:,1], yerr=exp_data_dimensionless[:,2],
             capsize=5, linestyle='', marker='o', color='grey')
plt.errorbar(x=exp_data_dimensionless[:,0], y=exp_data_dimensionless[:,3], yerr=exp_data_dimensionless[:,4],
             capsize=5, linestyle='', marker='o', color='grey')
plt.fill_between(x=exp_data_dimensionless[:,0], y1=np.zeros_like(exp_data_dimensionless[:,0]),
                 y2=exp_data_dimensionless[:,3], color=unstable_color, alpha=0.3)
plt.fill_between(x=exp_data_dimensionless[:,0], y1=exp_data_dimensionless[:,3], y2=exp_data_dimensionless[:,1],
                 color='C0', alpha=0.4)
plt.fill_between(x=exp_data_dimensionless[:,0], y1=exp_data_dimensionless[:,1],
                 y2=0.08*np.ones_like(exp_data_dimensionless[:,0]), color=unstable_color, alpha=0.3)
plt.annotate('Stable levitation', color='C0', xy=(2.7, 0.00065), fontsize=20, alpha=0.6, ha='center')

# Limit due to wall climbing by drag
vreds = np.logspace(np.log10(vred_min), np.log10(vred_max), 100)
def cd_factor(vreds, CaOhm2):
    # Re = 1*CaOhm2*(3/4/np.pi*vreds)**(1/3)
    Re = 2 * CaOhm2 * (vreds) ** (1 / 3)
    return 1 + 1/6*(Re)**(2/3)
# cd_factors=1
# ca_lim = (1/cd_factors)*2.0/9*(3/4*vreds/np.pi)**(2/3)
# plt.plot(vreds, ca_lim)
# cd_factors=cd_factor(vreds, CaOhm2=1e-2/(0.0025889525)**2)
# ca_lim = (1/cd_factors)*2.0/9*(3/4*vreds/np.pi)**(2/3)
# ca_lim = (1/cd_factors)*2.0/9*(vreds)**(2/3)
# plt.plot(vreds, ca_lim, '--', color='black')

# more rigorous solving for Ca_max


def func(Ca, Oh, vred):
    cd_f = cd_factor(vred, Ca/((Oh)**2))
    return Ca - (1/cd_f)*2.0/9*(vred)**(2/3)

cas = []
for v in vreds:
    r = root(func, x0=1e-2, args=(Oh_master, v))
    cas.append(r.x)

plt.plot(vreds, cas, '--', color='black')

#limit due to very small gap under the droplet

# ## Version where curvature is found as k_b = sqrt(k_0**2 + 2/caplen**2)
# mingap = 2000e-9
# prefactor = ((mingap/2.123/0.3)/cap_length)**(3/2) * 1
# ca_lowlim = prefactor * ( ((3 * vreds) / (4 * np.pi))**(-2/3) + 2)**(3/4)

# version where k_b is found from numerical sessile drop shapes - semianalytical
# G’ = G*0.384834 = 10.60624516*0.384834=4.081651509
G = 4
# G = 10.60624516 # *0.384834
mingap = 850e-9 #1350e-9
k_bs = 1/cap_length * np.sqrt( G*(vreds)**(-2/3) + 4 )
# k_bs = 1/cap_length * np.sqrt( G*(vreds)**(-0.64545015) + 4.07305507 )
F = 0.42
ca_lowlim = ( k_bs * mingap / (0.716 * 2.123 * F) )**(1.5)

# # for small droplet limit
# # version where k_b is found from numerical sessile drop shapes - semianalytical
# G = 10.60624516
# mingap = 1350e-9
# k_bs = 1/cap_length * np.sqrt( G*(vreds)**(-0.64545015) + 4.07305507 )
# ca_lowlim_2 = ( k_bs * mingap * ( (3*vreds/4/np.pi)**(-1/3) )**(-4/5) / (0.871 * 0.3) )**(5/4)
ca_lowlim_2 = ( k_bs * mingap / (0.871 * F) )**(5/4) * vreds**(1/3)

plt.plot(vreds, ca_lowlim, '--', color='C2')
# plt.plot(vreds, ca_lowlim_2, '--', color='C3')

plt.xscale('log')
plt.yscale('log')
plt.xlim(vred_min/1.1, vred_max*1.1)
plt.ylim(5e-5, 0.019)
plt.ylabel('Capillary number $\mu v / \sigma$')
# plt.xlabel('Droplet volume, dimensionless $\\tilde{V}=3V/(4 \pi) \cdot \left(\\frac{\sigma}{g \\rho}\\right)^{-3/2}$')
plt.xlabel('Droplet volume, unitless $\\tilde{V}=3V/(4 \pi a^3)$')
plt.tight_layout()
fig.savefig('figures/stability-phase-diagram-1.png', dpi=300)
plt.show()

