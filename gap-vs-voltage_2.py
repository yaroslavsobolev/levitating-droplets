import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
cmap = matplotlib.cm.get_cmap('viridis')
# prefactor = 4.1
prefactor = 4.7

def time_to_voltage(t):
    return (t-1.15)/(5-1.15)*100/prefactor

def plot_file(target_file, u, color):
    data = np.loadtxt(target_file, skiprows=5)
    times = data[:,0]
    start = np.argwhere(times > 1.15)[0][0]
    data = data[start:,:]
    rgba = cmap(0.5)
    plt.plot(time_to_voltage(data[:, 0]), data[:, 1] * 1e3, label='U={0:.1f} m/s'.format(u),
             color=color)
    plt.plot([time_to_voltage(data[-1, 0]), time_to_voltage(data[-1, 0])], [data[-1, 1]*1e3, 0], color='black', linestyle='--')
    # plt.plot(time_to_voltage(data[:,0])/u**(2/3), data[:,1]*1e3/u**(2/3))
    # plt.plot(time_to_voltage(data[:, 0]) / u**(2/3), data[:, 1] * 1e3 / data[0, 1])
    return time_to_voltage(data[-1, 0])

files_list = ['comsol_results/gap-vs-voltage/0p5-meter-per-second/gap_vs_time_1.txt',
              'comsol_results/gap-vs-voltage/0.8mps/gap_vs_time_1.txt',
              'comsol_results/gap-vs-voltage/1 meter-per-second/gap_vs_time_1.txt',
              'comsol_results/gap-vs-voltage/1.2mps/gap_vs_time_1.txt',
              'comsol_results/gap-vs-voltage/1.5mps/gap_vs_time_1.txt',
              'comsol_results/gap-vs-voltage/2-meter-per-second/gap_vs_time_1.txt']
Us = np.array([0.5, 0.8, 1, 1.2, 1.5, 2])
last_Vs = []
colors = cmap(np.linspace(0,1,Us.shape[0]))
fig, ax = plt.subplots(dpi=300, figsize=(3.75, 3.42))
for i,target_file in enumerate(files_list):
    last_Vs.append(plot_file(target_file, Us[i], color=colors[i]))
# plt.legend()
plt.ylabel('Air gap under the droplet at\nclosest separation, μm')
plt.xlabel('Voltage applied to the droplet, V')
plt.xlim(0, 55/prefactor)
plt.ylim(0, 20)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.tight_layout()
# fig.savefig('figures/comsol_electro_gaps.png', dpi=300)
plt.show()

data = np.loadtxt('misc_data/wired_voltage_thresh_vs_speed.txt', delimiter='\t', skiprows=1)

fig,ax1 = plt.subplots(figsize=(4.5*0.9,4*0.9))
tfem1 = plt.plot(Us, last_Vs, color='C0', linewidth=5, label='Theory (2D FEM)', alpha=0.6)
e1 = plt.plot(data[:,0]/1000, data[:,1], 'o', label='Experiment', color='C0', alpha=0.7, markersize=5)

### Linear fits
# z = np.polyfit(data[:,0]/1000, data[:,1], 1)
# p = np.poly1d(z)
# xp = np.linspace(0, 2, 100)
# plt.plot(xp, p(xp), '--', color='C0', alpha=0.5)
#
# z = np.polyfit(Us, last_Vs, 1)
# p = np.poly1d(z)
# xp = np.linspace(0, 2, 100)
# plt.plot(xp, p(xp), '--', color='C0', alpha=0.5)

plt.ylim(0,18)
plt.xlim(0,3)

xdata = data[:,0]/1000
ydata = data[:,1]

####################### Analytical theory

density_of_liquid = 1.05 #grams per cubic centimeter
sigma = 25e-3 # N/m
cap_length = 0.0015579 #m
mu_air = 1.81e-5 # dynamic viscosity of air in Pa * s
epsilon = 8.85e-12 # vacuum permittivity
cap_num_coeff = 0.000724 # gives Capillary number when multiplied by speed in m/s
droplet_volume = 25e-6 * (1e-1) ** 3 # 25 microliters. Volume in m**3
volume_reduced = droplet_volume * 3 / (4 * np.pi * (cap_length ** 3)) # dimensionless volume (see paper)
kappa_b = 2 / cap_length * np.sqrt(1 + volume_reduced ** (-2 / 3) ) # curvature in inverse meters (see paper)

def func(x, a):
    return a*x**(2/3)
popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
xs = np.linspace(0, 3, 100)
popt = [7]
tana1 = plt.plot(xs, func(xs, *popt), color='C0', linestyle=':',
         label='Theory (2D analytical)')

def cap_num(v):
    return cap_num_coeff * v

def Hmin(W):
    return 0.304 * np.exp(-0.471 * W) + 0.421

def gap_for_gap(U, h0, v):
    return (0.102 + 0.538 * np.exp( -0.576 * epsilon * U**2 / (sigma * h0) * (6 * cap_num(v))**(-2/3) )) * \
           (1 / kappa_b) * (6 * cap_num(v))**(2/3)

def plot_gaps_for_given_v(v, Umin, Umax, Nsteps):
    Us = np.linspace(Umin, Umax, Nsteps)
    def func(h0, U, v):
        return gap_for_gap(U, h0, v) - h0
    return Us, np.array([root(func, x0=2e-5, args=(U, v)).x[0] for U in Us])

def mingap_for_given_U_and_v(U, v):
    def func(h0, U, v):
        return gap_for_gap(U, h0, v) - h0
    h0_here = root(func, x0=3e-6, args=(U, v), tol=1e-14).x[0]
    Ws = epsilon * U ** 2 / (sigma * h0_here) * (6 * cap_num(v)) ** (-2 / 3)
    hmin = h0_here * Hmin(Ws)
    return hmin

def U_critical_vs_v(v, h_critical):
    def func(U, h_critical, v):
        return mingap_for_given_U_and_v(U, v) - h_critical
    return root(func, x0=7.89213033 * v**(2/3), args=(h_critical, v), tol=1e-14).x[0]

def func(v, h_critical, prefactor_here):
    return np.array([U_critical_vs_v(x, h_critical) * prefactor_here for x in v])

popt, pcov = curve_fit(func, xdata, ydata, p0=[1.35e-6, 1/5])
print(popt)

vs = np.linspace(0.5, 2.86, 50)
best_h_critical = popt[0] #1.35e-6
best_prefactor = popt[1]
Us_critical = np.array([U_critical_vs_v(v, best_h_critical) * best_prefactor for v in vs])
plt.plot(vs, Us_critical)
# plt.plot(vs, 28 * vs**(2/3))
# plt.show()


plt.xlabel('Flight velocity V, m/s')

color = 'C0'
ax1.set_ylabel('Wired critical voltage, U', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'C1'
ax2.set_ylabel('Wireless critical voltage, U', color=color)
ax2.tick_params(axis='y', labelcolor=color)

wireless_data = np.loadtxt('misc_data/wireless_voltage_thresh_vs_velocity.txt', skiprows=0, delimiter=' ')
e2 = ax2.errorbar(x=wireless_data[:,0], y=wireless_data[:,1], yerr=wireless_data[:,2],
             capsize=4, fmt='o', color=color,
             alpha=0.6, label='Experiment')
plt.ylim(0, 86)
ax2.axhline(y=np.mean(wireless_data[:,1]), color=color, label='Theory (2D analytical)', alpha=0.7)

handles, labels = ax1.get_legend_handles_labels()
order = [1,2,0]
l1 = ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right')
l1.get_frame().set_linewidth(3)
# l = ax1.legend([(e1,tana1)],['Experiment'])
# ax1.legend()
# ax2.legend(loc='upper left')
plt.tight_layout()
fig.savefig('figures/voltage_vs_speed_noleg_2.png', dpi=800)
plt.show()

sqr_factor = 1666.625 #volts
xdata = (data[:,0]/1000)*18.5e-6/25e-3
ydata = data[:,1]/sqr_factor
def func(x, a):
    return a*x**(2/3)
popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
# xs = np.linspace(0,3,100)
plt.plot(xdata, ydata, 'o')
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='Theory (2D analytical)')
plt.legend()
plt.show()