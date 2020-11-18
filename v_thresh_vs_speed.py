import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import configparser

def rpm_to_speed(x):
    return x*1.68 * 2 * np.pi * (11.8e-3) / 60
# list of file names and RPMs
file_list = [\
        ['E:/fastcam/2020_nov_10/FASTS4_11-10_1000rpm_000000/', 1000],
        ['E:/fastcam/2020_nov_10/FASTS4_11-10_950rpm_000001/', 950],
        ['E:/fastcam/2020_nov_10/FASTS4_11-10_900rpm_000000/', 900],
        ['E:/fastcam/2020_nov_10/FASTS4_11-10_850rpm_000000/', 850],
        ['F:/yankai_levitation_transfer_highspeed_videos/'
                        '2020_sep_25/'
                        'FASTS4_2300fps_1.3Vpp10hz_Synced-4_000001/', 300],
        ['F:/yankai_levitation_transfer_highspeed_videos/'
         '2020_oct_15/'
         'FASTS4_2020-10-15_1409_000000/img0000/', 500]
    ]

xs = []
ys = []
yerr = []
for r in file_list:
    file_name = r[0]
    rpm = r[1]
    misc_folder = file_name + 'misc/'
    crossings = np.loadtxt(misc_folder + 'crossings.txt')
    print('Mean threshhold voltage at events: {0:.2f} V'.format(np.mean(np.abs(crossings))))
    xs.append(rpm_to_speed(rpm))
    ys.append(np.mean(np.abs(crossings)))
    yerr.append(np.std(np.abs(crossings)))

plt.errorbar(xs, ys, yerr=yerr, capsize=4, fmt='o')
plt.ylim(0, 90)
plt.xlim(0, 2.5)
plt.xlabel("Droplet's velocity, m/s")
plt.ylabel('Threshold voltage on electrode, V')
plt.show()