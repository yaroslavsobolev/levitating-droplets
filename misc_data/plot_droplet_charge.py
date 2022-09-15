import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('charge-on-droplet/charge-on-droplet.txt', sep='\t')

fig, ax = plt.subplots(figsize=(6*1.3, 1.5*1.3))

# remove frame (axes)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# draw boxplot
sns.set(style="darkgrid")
sns.boxplot(data=df, orient="h", whis=8)#, boxprops=dict(alpha=.3))

# make boxplot filling semitransparent
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))

# overlay individual points
for i,col in enumerate(df.columns):
    points = df[col]
    plt.scatter(points, [i]*points.shape[0], alpha=0.65)

# add vertical line at zero, add labels, save figure
plt.axvline(x=0, color='red', linewidth=1, alpha=0.7)
plt.xlabel('Droplet\'s net charge, nC')
plt.tight_layout()
fig.savefig('charge-on-droplet/droplet-net-cnarge.png', dpi=300)
plt.show()