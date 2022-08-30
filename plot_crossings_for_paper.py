import numpy as np
import matplotlib.pyplot as plt
import pickle

target_folder = 'F:/yankai_levitation_transfer_highspeed_videos/'\
                '2020_sep_25/'\
                'FASTS4_1.25Vp10hz_Synced_000000/'
misc_folder = target_folder + 'misc/'
freq = 10
limc=2.5/2*1.1*100

# start with a rectangular Figure
f5, ax_scatter = plt.subplots(figsize=(3.5, 3))

# the scatter plot:
applied_voltage = np.loadtxt(misc_folder + 'applied_voltage_one_period.txt')
ts = np.loadtxt(misc_folder + 'time_one_period.txt')
ax_scatter.plot(ts, applied_voltage, color='black')
xs = np.loadtxt(misc_folder + 'time_within_period.txt')
with open(misc_folder + 'colors.pickle', 'rb') as handle:
    colors = pickle.load(handle)
crossings = np.loadtxt(misc_folder + 'crossings.txt')
for i, crossing in enumerate(crossings):
    time_within_period = xs[i]
    ax_scatter.scatter(time_within_period, crossing, s=30,
                       color=colors[i], alpha=0.2)
print('Mean threshhold voltage at events: {0:.2f} V'.format(np.mean(np.abs(crossings))))
# f5.set_title('Locations of events at the respective field cycle')
ax_scatter.set_ylabel('Applied potential, V')
ax_scatter.set_xlabel('Time within a single period, s')
ax_scatter.axhline(y=0, color='grey')
# ax_scatter.scatter(xs, crossings)
ax_scatter.set_ylim((-limc, limc))


target_folder = 'F:/yankai_levitation_transfer_highspeed_videos/'\
                '2020_sep_25/'\
                'FASTS4_2300fps_1.3Vpp10hz_Synced-4_000001/'
misc_folder = target_folder + 'misc/'
# the scatter plot:
applied_voltage = np.loadtxt(misc_folder + 'applied_voltage_one_period.txt')
ts = np.loadtxt(misc_folder + 'time_one_period.txt')
ax_scatter.plot(ts, applied_voltage, color='black')
xs = np.loadtxt(misc_folder + 'time_within_period.txt')
with open(misc_folder + 'colors.pickle', 'rb') as handle:
    colors = pickle.load(handle)
crossings = np.loadtxt(misc_folder + 'crossings.txt')
for i, crossing in enumerate(crossings):
    time_within_period = xs[i]
    ax_scatter.scatter(time_within_period, crossing, s=30,
                       color=colors[i], alpha=0.2)
plt.tight_layout()
f5.savefig('figures/voltage_thresholds.png', dpi=300)
plt.show()