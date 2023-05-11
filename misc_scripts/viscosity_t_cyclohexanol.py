import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('misc_data/rheometry/vs_temperature/cyclohexanol_viscosity-vs-t.csv', delimiter='\t', skiprows=1)
data[0, 0] = 273+20
data[0, 1] = 57.5
plt.plot(data[:, 0], data[:, 1], '-o', color='C0')
plt.show()

# cuve fit an exponent to data
from scipy.optimize import curve_fit

def func(x, a, b):
    return a * np.exp(b/(x))

popt, pcov = curve_fit(func, data[:, 0], data[:, 1])
print(popt)

# plot data and fit
xs = np.linspace(270, 400, 100)
plt.plot(data[:, 0], data[:, 1], '-o', color='C0')
plt.plot(xs, func(xs, *popt), '--', color='C1')
plt.show()


parameters = np.array([popt[0], popt[1]])
np.savetxt('misc_data/rheometry/vs_temperature/cyclohexanol_viscosity_arrhenius_parameters.txt', parameters)

# plt.show()
