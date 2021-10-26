import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

def fit_power(xdata, ydata, color):
    def func(x,a,b):
        return a*x**b
    b0 = -9
    popt, pcov = curve_fit(func, xdata, ydata, sigma=ydata,
                           p0=(0, b0), bounds=([-np.inf, b0-1e-4], [np.inf, b0+1e-4]))
    plt.plot(xdata, func(xdata, *popt), '--', color=color, alpha=0.5)
    return popt

fig, ax = plt.subplots(figsize=(4.5*0.9,4*0.9))
alpha=0.9
plt.yscale('log')
plt.xscale('log')
data = np.loadtxt('misc_data/print_volume/volume_vs_speed/DNA-PEG.txt', delimiter='\t', skiprows=1)
data[:,0] = data[:,0]/1000
color='C0'
plt.errorbar(x=data[:,0], y=data[:,1], yerr=data[:,2], color=color, capsize=4, fmt='o', label='DNA-PEG', alpha=alpha)
print(fit_power(data[:,0], data[:,1], color))

data = np.loadtxt('misc_data/print_volume/volume_vs_speed/custom_detergent.txt', delimiter='\t', skiprows=1)
data[:,0] = data[:,0]/1000
color='C1'
plt.errorbar(x=data[:,0], y=data[:,1], yerr=data[:,2], color=color, capsize=4, fmt='o', label='Custom\ndetergent', alpha=alpha)
print(fit_power(data[:,0], data[:,1], color))

data = np.loadtxt('misc_data/print_volume/volume_vs_speed/commercial_detergent.txt', delimiter='\t', skiprows=1)
data[:,0] = data[:,0]/1000
color='C2'
plt.scatter(data[:,0], data[:,1], color=color, label='Commercial\ndetergent', alpha=alpha)
print(fit_power(data[:,0], data[:,1], color))

plt.grid()
handles, labels = ax.get_legend_handles_labels()
order = [1,2,0]
# l1 = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right')

plt.xlim(0.65, 2)
plt.ylim(0.01, 1000)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
# ax.set_xticks([0.7, 0.8, 0.9, 2], labels=['0.7', '0.8', '0.9', '2'], minor=True)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.1f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
ax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.1f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
plt.xlabel('Velocity, m/s')
plt.ylabel('Volume of single printed dot, pL')
plt.tight_layout()
fig.savefig('figures/volume_vs_speed.png', dpi=300)
plt.show()

