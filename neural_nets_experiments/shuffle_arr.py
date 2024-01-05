import numpy as np
from time import time
from utils import rand_perm_k


def shrt(n):
    s = str(n)
    i = len(s) - 1
    while i >= 0 and s[i] == '0':
        i -= 1
    return s[:i + 1] + '*10^' + str(len(s) - i - 1)


Ds = [10 ** 7]
Ks = [10 ** 5, 5 * 10 ** 5, 10 ** 6]

n_runs = 10
impl_ids = [0, 1, 2, 3]
stats = []

for D in Ds:
    for K in Ks:
        for i in impl_ids:
            times = []
            for j in range(n_runs):
                t0 = time()
                perm = rand_perm_k(D, K, i)
                t1 = time()
                times.append(t1 - t0)

            mu = np.mean(times)
            sigma = np.std(times)

            stats.append((mu, sigma))

            print(f'D={shrt(D)}, K={shrt(K)}: Alg {i} takes time {mu:.3f}, sigma={sigma:.3f}')
