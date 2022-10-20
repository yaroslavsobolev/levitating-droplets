import numpy as np
import matplotlib.pyplot as plt

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

F = 5
fig, ax = plt.subplots(figsize=(8, 1))
simpleaxis(ax)

xs = np.linspace(0, 1, 1000)
ys = np.zeros_like(xs)
from_t = 0.2
to_t = 0.6
ys[np.logical_and(xs > from_t, xs < to_t)] = 1
plt.plot(xs, ys, color='black')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.ylim(-0.1, 1.1)
plt.xlim(0, 1.1)

# ax.set_axisbelow(True)

ax.get_xaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)

# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off

fig.savefig('wireless-square-pulse-illustration.png', dpi=300)

plt.show()