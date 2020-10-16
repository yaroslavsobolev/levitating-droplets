import numpy as np
import matplotlib.pyplot as plt


def get_binomial_runs(N = 100000):
    return np.random.randint(2, size=N)

def count_runs(xs):
    seqlens = []
    count = 0
    for x in xs:
        if x:
            count+=1
        else:
            if count > 1:
                seqlens.append(count)
            count=0
    return seqlens

seqlens = np.array(count_runs(get_binomial_runs()))
plt.hist(seqlens, bins=np.arange(20),density=True)
plt.show()