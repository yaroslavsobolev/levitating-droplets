import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Experimental data is done for water-PEG mixtyre. Density ~1000 kg/m^3, surface tension 25 mN/m, and high viscosity.
cap_length0 = 0.00159637714 #m
mu_air = 1.81e-5 # Pa*s
sigma = 25e-3 # N/m
density_air = 1.225
cap_num_coeff = 0.000724 # gives Capillary number when multiplied by speed in m/s

vred_max = 98.3
vred_min = 1.22

fig, ax = plt.subplots(figsize=(4,6))

sigma0 = 25e-3

for sigma_factor in [1,2,5]:
    sigma = sigma0*sigma_factor
    cap_length = cap_length0*np.sqrt(sigma_factor)
    Oh_number = mu_air/np.sqrt( sigma * density_air * cap_length )
    # Limit due to wall climbing by drag
    vreds = np.logspace(np.log10(vred_min), np.log10(vred_max), 100)
    def cd_factor(vreds, CaOhm2):
        Re = CaOhm2*(3/4/np.pi*vreds)**(1/3)
        return 1 + 1/6*(Re)**(2/3)
    # cd_factors=1
    # ca_lim = (1/cd_factors)*2.0/9*(3/4*vreds/np.pi)**(2/3)
    # plt.plot(vreds, ca_lim)
    cd_factors=cd_factor(vreds, CaOhm2=1e-2/(Oh_number)**2)
    ca_lim = (1/cd_factors)*2.0/9*(3/4*vreds/np.pi)**(2/3)
    # plt.plot(vreds, ca_lim, '--', color='black')

    # more rigorous solving for Ca_max
    def func(Ca, Oh, vred):
        cd_f = cd_factor(vred, Ca/((Oh)**2))
        return Ca - (1/cd_f)*2.0/9*(3/4*vred/np.pi)**(2/3)

    cas = []
    for v in vreds:
        r = root(func, x0=1e-2, args=(Oh_number, v))
        cas.append(r.x[0])
    cas = np.array(cas)

    plt.plot(vreds, cas, '--', color='black')

    #limit due to very small gap under the droplet
    mingap = 2000e-9
    prefactor = ((mingap/2.123/0.3)/cap_length)**(3/2) * 1
    ca_lowlim = prefactor * ( ((3 * vreds) / (4 * np.pi))**(-2/3) + 2)**(3/4)
    plt.plot(vreds, ca_lowlim, '--', color='C2')

    plt.fill_between(x=vreds, y1=ca_lowlim, y2=cas, color='C0', alpha=0.5)

plt.xscale('log')
plt.yscale('log')
plt.xlim(vred_min/1.1, vred_max*1.1)
plt.ylim(0.3e-4, 0.041)
plt.ylabel('Capillary number $\mu v / \sigma$')
plt.xlabel('Droplet volume, dimensionless $\\tilde{V}=V \cdot \left(\\frac{\sigma}{g \\rho}\\right)^{-3/2}$')
plt.tight_layout()
plt.show()

