import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('comsol_results/drop-to-drop-attraction/force_vs_distance.txt', delimiter='\t',skiprows=1)
r = (data[:,0]-2.8)*2
F = data[:,2]/1e-12

plt.plot(r, F)
plt.xlabel('Droplet-droplet separation, mm')
plt.ylabel('Attractive force per applied voltage squared, pN/V$^2$')
plt.show()
