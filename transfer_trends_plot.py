import numpy as np
import matplotlib.pyplot as plt
import pickle

records = pickle.load(open('fluorescence_data/2020_oct/oct_08/Exp_Acquisition 1 2020-10-07 13_58_43 »» Detector1.group._processed.pickle',
                           "rb"))
records.extend(pickle.load(open('fluorescence_data/2020_oct/oct_15/excel_Acquisition 1 2020-10-14 14_28_45 »» Detector1.group._processed.pickle',
                           "rb")))

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
ax2 = ax.secondary_xaxis('top', functions=(lambda x:x*1.6*2*np.pi*(11.8e-3)/60, lambda x:x/(1.6*2*np.pi*(11.8e-3)/60)))
ax2.set_xlabel("Droplet's linear speed relative to the drum, m/s ")
plt.ylim(0.005, 400)
plt.xlabel('Speed of rotation, rpm')
plt.ylabel('Volume of single printed dot, pL')
plt.legend()
plt.show()