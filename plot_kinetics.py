import numpy as np
import matplotlib
import matplotlib.pyplot as plt


cmap = matplotlib.cm.get_cmap('viridis')
colors = cmap(np.linspace(0,1,5))
factor = 1/444600
data = np.loadtxt('misc_data/reaction_progress/spectra_for_reaction_progress.txt', delimiter='\t', skiprows=3)
wavelength = data[:,0]
baseline = (data[:,1] + data[:,2] + data[:,3] )/3

fig = plt.figure(1, dpi=300, figsize=(3.8, 2.6))
plt.plot(wavelength, factor*(data[:,3]-baseline), color=colors[0])
for i in range(4):
    plt.plot(wavelength, factor*(data[:,4 + i*3] - baseline), color=colors[i+1])
plt.ylabel('Fluorescence intensity, a.u.')
plt.xlabel('Wavelength, nm')
plt.tight_layout()
fig.savefig('figures/reaction-progress.png', dpi=300)
plt.show()