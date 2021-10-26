import numpy as np
import matplotlib.pyplot as plt

exp_data = np.loadtxt('misc_data/experimental_stability/Droplet instability_clean.txt', delimiter='\t', skiprows=1)
# Experimental data is done for water-PEG mixtyre. Density ~1000 kg/m^3, surface tension 25 mN/m, and high viscosity.
cap_length = 0.00159637714 #m
mu_air = 1.81e-5 # Pa*s
sigma = 25e-3 # N/m
cap_num_coeff = 0.000724 # gives Capillary number when multiplied by speed in m/s
exp_data_dimensionless = np.copy(exp_data)
exp_data_dimensionless[:,0] *= 1e-9/(cap_length**3) # 1e-9 factor is for converting from uL to m^3
exp_data_dimensionless[:,1:] *= 1e-3*cap_num_coeff # 1e-3 factor is for converting from mm/s to m/s

vred_max = np.max(exp_data_dimensionless[:,0])
vred_min = np.min(exp_data_dimensionless[:,0])

fig, ax = plt.subplots(figsize=(4,6))
plt.errorbar(x=exp_data_dimensionless[:,0], y=exp_data_dimensionless[:,1], yerr=exp_data_dimensionless[:,2],
             capsize=5, linestyle='', marker='o', color='grey')
plt.errorbar(x=exp_data_dimensionless[:,0], y=exp_data_dimensionless[:,3], yerr=exp_data_dimensionless[:,4],
             capsize=5, linestyle='', marker='o', color='grey')
plt.fill_between(x=exp_data_dimensionless[:,0], y1=np.zeros_like(exp_data_dimensionless[:,0]),
                 y2=exp_data_dimensionless[:,3], color='goldenrod', alpha=0.3)
plt.fill_between(x=exp_data_dimensionless[:,0], y1=exp_data_dimensionless[:,3], y2=exp_data_dimensionless[:,1],
                 color='C0', alpha=0.4)
plt.fill_between(x=exp_data_dimensionless[:,0], y1=exp_data_dimensionless[:,1],
                 y2=0.08*np.ones_like(exp_data_dimensionless[:,0]), color='goldenrod', alpha=0.3)
plt.annotate('Stable levitation', color='C0', xy=(10, 0.0015), fontsize=20, alpha=0.6, ha='center')

# Limit due to wall climbing by drag
vreds = np.logspace(np.log10(vred_min), np.log10(vred_max), 100)
def cd_factor(vreds, CaOhm2):
    Re = 1*CaOhm2*(3/4/np.pi*vreds)**(1/3)
    return 1 + 1/6*(Re)**(2/3)
# cd_factors=1
# ca_lim = (1/cd_factors)*2.0/9*(3/4*vreds/np.pi)**(2/3)
# plt.plot(vreds, ca_lim)
cd_factors=cd_factor(vreds, CaOhm2=1e-2/(0.0025889525)**2)
ca_lim = (1/cd_factors)*2.0/9*(3/4*vreds/np.pi)**(2/3)
# plt.plot(vreds, ca_lim, '--', color='black')

# more rigorous solving for Ca_max
from scipy.optimize import root

def func(Ca, Oh, vred):
    cd_f = cd_factor(vred, Ca/((Oh)**2))
    return Ca - (1/cd_f)*2.0/9*(3/4*vred/np.pi)**(2/3)

cas = []
for v in vreds:
    r = root(func, x0=1e-2, args=(0.0025889525, v))
    cas.append(r.x)

plt.plot(vreds, cas, '--', color='black')

#limit due to very small gap under the droplet

# ## Version where curvature is found as k_b = sqrt(k_0**2 + 2/caplen**2)
# mingap = 2000e-9
# prefactor = ((mingap/2.123/0.3)/cap_length)**(3/2) * 1
# ca_lowlim = prefactor * ( ((3 * vreds) / (4 * np.pi))**(-2/3) + 2)**(3/4)

# version where k_b is found from numerical sessile drop shapes - semianalytical
G = 10.111
# G = 10.60624516
mingap = 1350e-9
k_bs = 1/cap_length * np.sqrt( G*(vreds)**(-2/3) + 4 )
# k_bs = 1/cap_length * np.sqrt( G*(vreds)**(-0.64545015) + 4.07305507 )
ca_lowlim = ( k_bs * mingap / (2.123 * 0.3) )**(1.5)

# # for small droplet limit
# # version where k_b is found from numerical sessile drop shapes - semianalytical
# G = 10.60624516
# mingap = 1350e-9
# k_bs = 1/cap_length * np.sqrt( G*(vreds)**(-0.64545015) + 4.07305507 )
# ca_lowlim_2 = ( k_bs * mingap * ( (3*vreds/4/np.pi)**(-1/3) )**(-4/5) / (0.871 * 0.3) )**(5/4)


plt.plot(vreds, ca_lowlim, '--', color='C2')
# plt.plot(vreds, ca_lowlim_2, '--', color='C3')

plt.xscale('log')
plt.yscale('log')
# plt.xlim(vred_min/1.1, vred_max*1.1)
# plt.ylim(0.5e-4, 0.041)
plt.ylabel('Capillary number $\mu u / \sigma$')
plt.xlabel('Droplet volume, dimensionless $\\tilde{V}=V \cdot \left(\\frac{\sigma}{g \\rho}\\right)^{-3/2}$')
plt.tight_layout()
plt.show()

