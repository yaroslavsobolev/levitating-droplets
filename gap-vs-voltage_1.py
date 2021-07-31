import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
cmap = matplotlib.cm.get_cmap('viridis')
prefactor = 4.1
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
for i,target_file in enumerate(files_list):
    last_Vs.append(plot_file(target_file, Us[i], color=colors[i]))
plt.legend()
plt.ylabel('Air gap under the droplet at closest separation, μm')
plt.xlabel('Voltage applied to the droplet, V')
plt.xlim(0, 55/prefactor)
plt.ylim(0, 20)
plt.show()

data = np.loadtxt('misc_data/wired_voltage_thresh_vs_speed.txt', delimiter='\t', skiprows=1)


plt.plot(Us, last_Vs, 'o', color='C1', label='Theory (2D FEM)')
plt.plot(data[:,0]/1000, data[:,1], 'o', label='Experiment')
z = np.polyfit(data[:,0]/1000, data[:,1], 1)
p = np.poly1d(z)
xp = np.linspace(0, 2, 100)
plt.plot(xp, p(xp), '--', color='C0', alpha=0.5)

z = np.polyfit(Us, last_Vs, 1)
p = np.poly1d(z)
xp = np.linspace(0, 2, 100)
plt.plot(xp, p(xp), '--', color='C1', alpha=0.5)
plt.xlabel('Flight velocity V, m/s')
plt.ylabel('Critical voltage, U')
plt.ylim(0,120/prefactor)
plt.xlim(0,3)

xdata = data[:,0]/1000
ydata = data[:,1]
def func(x, a):
    return a*x**(2/3)
popt, pcov = curve_fit(func, xdata, ydata)
print(popt)
xs = np.linspace(0,3,100)
plt.plot(xs, func(xs, *popt), 'r-',
         label='Theory (2D analytical)')
plt.legend()
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