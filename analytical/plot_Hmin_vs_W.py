import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
Ws = np.load('analytical/downstream/Ws.npy')
# np.load('analytical/downstream/Bs.npy', np.array(Bs))
Hmins = np.load('analytical/downstream/Hmins.npy')
# plt.plot(Ws, Hmins)
# plt.show()

xdata = Ws
ydata = Hmins
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# def func(x, a, b, c):
#     return a * x**(b) + c

# def func(x, a, b, c):
#     return a * np.exp(-(x**b)) + c

# bounds = [[-np.inf, 0.576, -np.inf], [np.inf, 0.576001, np.inf]]
bounds = [[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]]
# bounds = [[-np.inf, -np.inf, -0.1], [np.inf, np.inf, 0.1]]
# bounds = [[0.716, -np.inf, -np.inf], [0.71601, np.inf, np.inf]]
popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)
print(popt)
fig, axarr = plt.subplots(2, 1, figsize=(5, 4.75), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(left=0.2)
ax = axarr[0]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(Ws, Hmins, 'C0', label='Numerical result')
ax.plot(xdata, func(xdata, *popt), 'C1',
         label='Exponential fit\n%.3f$\cdot \exp(-$%5.3f$\cdot W) +$%5.3f' % tuple(popt))
ax.legend()
ax.set_ylabel('$H_{min}$')
ax = axarr[1]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.fill_between(x=xdata, y1=0, y2=func(xdata, *popt)-ydata, color='black', alpha=0.3)
ax.axhline(y=0, color='black')
ax.set_ylabel('Residual')
ax.set_xlabel('$W$')

plt.savefig('figures/Hmin_vs_W.png', dpi=300)
plt.show()