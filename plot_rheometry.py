import numpy as np
import matplotlib.pyplot as plt

def read_line_from_file(filename, line_to_read):
    a_file = open(filename)
    for position, line in enumerate(a_file):
        if position == line_to_read:
            res = line
            break
    return res

data_filename = 'misc_data/rheometry/rheometry_2021-11-30.txt'
data = np.loadtxt(data_filename, delimiter='\t', skiprows=3)
header = read_line_from_file(data_filename, line_to_read=2)[:-1].split('\t')

fig = plt.figure(1, figsize=(3.7, 3.5))
colors = ['black', 'C1', 'C2', 'C3', 'C4', 'C5', 'C0']
linewidths = [1, 3, 3, 1, 1, 1, 3]
for col_id in range(1, data.shape[1]):
    plt.loglog(data[:, 0], data[:, col_id], label=header[col_id], color=colors[col_id], linewidth=linewidths[col_id],
               alpha=0.7)
plt.ylim(10, 5000)
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
order = [5,0,1,2,3,4]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.xlabel('Shear rate, $s^{-1}$')
plt.ylabel('Viscosity, mPa$\cdot$s')
plt.tight_layout()
# fig.savefig('figures/rheometry.png', dpi=300)
plt.show()