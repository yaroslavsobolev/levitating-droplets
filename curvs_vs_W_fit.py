import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize

Ws = np.load('analytical/ws.npy')
cs = np.load('analytical/curvs.npy')
# plt.plot(Ws, cs)

xdata = Ws
ydata = cs
def func(x, a, b, c):
    return a*np.exp(-b*x) + c
popt, pcov = optimize.curve_fit(func, xdata, ydata)
print(popt)
# xs = np.linspace(0,3,100)
fig = plt.figure(1)
ax=fig.add_subplot(211)
plt.plot(xdata, ydata, 'o',label='Data')
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='Fit')
plt.legend()
ax=fig.add_subplot(212)
plt.plot(xdata, func(xdata, *popt)-ydata, 'C0')

plt.show()