import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('misc_data/cyclohexanol_surface_tension_vs_temperature.txt', delimiter='\t', skiprows=1)
plt.plot(data[:, 0], data[:, 1], '-o', color='C0')
plt.show()

# cuve fit an exponent to data
from scipy.optimize import curve_fit

def func(x, a, b):
    return a + b*x

popt, pcov = curve_fit(func, data[:, 0], data[:, 1])
print(popt)

# plot data and fit
xs = np.linspace(20, 100, 100)
plt.plot(data[:, 0], data[:, 1], '-o', color='C0')
plt.plot(xs, func(xs, *popt), '--', color='C1')
plt.show()


parameters = np.array([popt[0], popt[1]])
np.savetxt('misc_data/cyclohexanol_surface_tension_vs_t_linefit_parameters.txt', parameters)

# plt.show()
