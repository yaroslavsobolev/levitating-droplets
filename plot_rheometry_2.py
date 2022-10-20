import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def read_line_from_file(filename, line_to_read):
    a_file = open(filename)
    for position, line in enumerate(a_file):
        if position == line_to_read:
            res = line
            break
    return res

data_filename = 'misc_data/rheometry/rheometry_2022-10-20.txt'
data = np.loadtxt(data_filename, delimiter='\t', skiprows=3)
header = read_line_from_file(data_filename, line_to_read=2)[:-1].split('\t')

#olumns = PEG25%	PEG30%	PVA25%	PVA30%	DNA 7.6mg/mL	PEG25%_DNA5.3mg/mL	PVA25%_DNA3.5mg/mL	SLN	Detergent

fig = plt.figure(1, figsize=(3.7, 3.5))
colors = ['black', 'C3', 'C3', 'C6', 'C6', 'grey', 'C5', 'C2', 'C1', 'C0']
linewidths = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3]
linestyles = ['-', '--', '-', '--', '-', '-', '-', '-', '-', '-']
for col_id in range(1, data.shape[1]):
    plt.loglog(data[:, 0], data[:, col_id], label=header[col_id], color=colors[col_id], linewidth=linewidths[col_id],
               alpha=0.7, linestyle=linestyles[col_id])
plt.ylim(10, 5000)
# plt.legend()
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [5,0,1,2,3,4]
# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
ax = plt.gca()
simpleaxis(ax)
ax.add_patch(
     patches.Rectangle(
        (50, 220),
        1950,
        650,
        fill=False,      # remove background
        color='black'
     ) )

plt.xlabel('Shear rate, $s^{-1}$')
plt.ylabel('Viscosity, mPa$\cdot$s')
plt.tight_layout()
fig.savefig('figures/rheometry.png', dpi=300)
plt.show()