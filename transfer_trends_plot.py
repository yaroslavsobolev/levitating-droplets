import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker as mticker
import pickle

def remove_duplicate_dicts_from_list(l):
    seen = set()
    new_l = []
    for d in l:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l

records = pickle.load(open('fluorescence_data/2020_oct/oct_08/Exp_Acquisition 1 2020-10-07 13_58_43 »» Detector1.group._processed.pickle',
                           "rb"))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_15/excel_Acquisition 1 2020-10-14 14_28_45 »» Detector1.group._processed.pickle',
                           "rb")))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_20/Acquisition 1 2020-10-21 17_11_10 »» Detector1.group._processed.pickle',
                           "rb")))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_26/Acquisition 1 2020-10-26 18_53_29 »» Detector1.group._processed.pickle',
                           "rb")))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_27/specturm for plot/spectrum for plot_oct_27._processed.pickle',
                           "rb")))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_28/New Session »» Detector1.group._processed.pickle',
                           "rb")))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_29/New Session »» Detector1.group._processed.pickle',
                           "rb")))

records = remove_duplicate_dicts_from_list(records)
# for vpp in [1.25, 1.75]:
#     recs = [r for r in records if (r['rpm'] == 300 and r['vpp']==vpp)]
#     xs = np.array([r['cycles'] for r in recs])
#     ys = [r['vol_per_droplet'] for r in recs]
#     plt.scatter(2*xs, ys, label='Vpp={0:.2f} V'.format(vpp))
# plt.ylim(0, 200)
# plt.xlabel('Length of printing sequence in dots')
# plt.ylabel('Volume of single printed dot, pL')
# plt.legend()
# plt.show()

# for r in records:
#     try:
#         print(r['vpp'])
#     except TypeError:
#         print('List is: {0}'.format(r))

fig2,ax = plt.subplots()






possible_vpps = [1.25, 1.75, 2]

for vpp in possible_vpps:
    recs = [r for r in records if r['vpp']==vpp]
    xs = np.array([r['rpm'] for r in recs])
    #group by RPM
    unique_rpms = list(set(list(xs)))
    xs = []
    errs = []
    ys = []
    for rpm in unique_rpms:
        xs.append(rpm)
        vols = [r['vol_per_droplet'] for r in recs if r['rpm']==rpm]
        plt.scatter([rpm]*len(vols), vols, color='black', alpha=0.4)
        ys.append(np.mean(np.array(vols)))
        if len(vols)>1:
            errs.append(np.std(np.array(vols)))
        else:
            errs.append(np.NaN)
    # ys = [r['vol_per_droplet'] for r in recs]
    # vpps = [r['vpp'] for r in records]
    # Olo's gramophone
    # np.append(0.01)
    # xs.append(1200)
    plt.errorbar(x=xs, y=ys, yerr=errs, markersize=5, marker='o', capsize=5, linestyle='none', label='Vpp={0:.2f} V'.format(vpp))

plt.plot([1200, 1200], [0.01, 0.1], linewidth=5, label='Gramophone')
plt.yscale('log')
plt.xscale('log')
# formatter = ScalarFormatter()
# formatter.set_scientific(False)
# ax.xaxis.set_major_formatter(formatter)
ax2 = ax.secondary_xaxis('top', functions=(lambda x:x*1.6*2*np.pi*(11.8e-3)/60, lambda x:x/(1.6*2*np.pi*(11.8e-3)/60)))
ax2.set_xlabel("Droplet's linear speed relative to the drum, m/s ")
plt.ylim(0.005, 400)

for ax_here in [ax, ax2]:
    ax_here.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax_here.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_here.tick_params(which='major', length=4)
    ax_here.tick_params(which='minor', length=4)

# ax2.xaxis.set_major_formatter(ScalarFormatter())
plt.xlabel('Speed of rotation, rpm')
plt.ylabel('Volume of single printed dot, pL')
plt.legend()
plt.show()