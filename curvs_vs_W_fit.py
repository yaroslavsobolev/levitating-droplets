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
fig, axarr = plt.subplots(2, 1, figsize=(5, 4.75), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(left=0.2)
ax = axarr[0]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(xdata, ydata, color='C0',label='Numerical result')
ax.plot(xdata, func(xdata, *popt), color='C1',
         label="Exponential fit\n$0.102 + 0.538\exp(-0.576 \cdot W)$")
# plt.xlabel('$W$')
ax.set_ylabel("$H''(- \infty)$")
ax.legend()
ax=axarr[1]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.fill_between(x=xdata, y1=0, y2=func(xdata, *popt)-ydata, color='black', alpha=0.3)
ax.axhline(y=0, color='black')
ax.set_ylabel('Residual')
ax.set_xlabel('$W$')
fig.savefig('figures/Hcurvature_vs_W.png', dpi=300)

plt.show()