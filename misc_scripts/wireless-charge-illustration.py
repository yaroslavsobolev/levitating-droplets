import numpy as np
import matplotlib.pyplot as plt

F = 5
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.linspace(0, 1, 1000)
ys = np.sin(xs * np.pi * 2 * F)
plt.plot(xs, ys, color='black')

thresh1 = 0.7
cs = 0.6 * np.exp(-xs * 5)
plt.plot(xs, cs, '--', color='C0')
plt.plot(xs, cs + thresh1, color='C0')
plt.plot(xs, cs - thresh1, color='C0')
plt.xlabel('Time')
plt.ylabel('Voltage')

plt.show()