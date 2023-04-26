import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('misc_data/experimental_stability/2021-12-08c.xlsx', sheet_name='Sheet1',
                     header=0)
# for col in df.columns:
#     print(col)

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

fig, ax = plt.subplots(dpi=300, figsize=(4.5, 3.1))
plt.xscale('log')

viscosity_factor = 1/1000

unstable_color = 'goldenrod'

marker_dict = {'Other': 'o',
               'Organic solvent': 'v',
               'Alkanes': '^',
               'Liquid polymer': '<',
               'Polymer in water': '>',
               'Surfactant': 's', # square
               'Glycol': 'D', #diamond
               'Silicon oil': 'X', # large X
               'Silicon oil, literature': 'P', #large plus
               'Detergent': '*'} # star

for index, row in df.iterrows():
    # print(row)
    if index > 62:
        break
    print('{0} => {1} => {2}'.format(index, row['Type'], row['Details']))
    if row['Success'] == 1:
        color = 'C0'
    elif row['Success'] == 0.7:
        color = 'yellowgreen'
    elif row['Success'] == 0.5:
        color = unstable_color
    else:
        color = unstable_color

    # if row['Details'] == 'silicon oil': # literature data from doi:10.3929/ethz-b-000279153
    #     marker = 's'
    # else:
    #     marker = 'o'
    marker = marker_dict[row['Type']]
    if row['Viscosity [mPa*s] @(100/s)'] > 0:
        plt.plot([row['Viscosity [mPa*s] @(1/s)']*viscosity_factor, row['Viscosity [mPa*s] @(100/s)']*viscosity_factor],
                 [row['Surface tension [mN/m]'], row['Surface tension [mN/m]']],
                 color=color, alpha=0.5)
    if marker == 'P':
        s = 35
    elif marker == '*':
        s = 45
    else:
        s = 25
    plt.scatter([row['Viscosity [mPa*s] @(1/s)']*viscosity_factor],
                 [row['Surface tension [mN/m]']],
                 color=color, marker=marker, alpha=0.7, s=s)

# data = df.loc[df['Success'] == 1]
# viscosities = data['Viscosity [Pa*s] @(1/s)']
# gammas = data['Surface tension [mN/m]']
# plt.scatter(viscosities, gammas, alpha=0.5, color='C0')

# plt.xlim(np.min(viscosities)/2, np.max(viscosities)*2)
# plt.ylim(np.min(gammas)/1.1, np.max(gammas)*1.1)

plt.xlabel('Viscosity $\mu$, Pa·s')
plt.ylabel('Surface tension $\sigma$, mN·m$^{-1}$')

# # failures
# data = df.loc[df['Success'] == 0]
# viscosities = data['Viscosity [Pa*s] @(1/s)']
# gammas = data['Surface tension [mN/m]']
# plt.scatter(viscosities, gammas, alpha=0.5, color='C3')

plt.xlim(0.0002, 15)
plt.ylim(10, 77)
plt.tight_layout()
simpleaxis(ax)

fig.savefig('figures/liquid_params.png', dpi=800)

plt.show()