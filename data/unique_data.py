import numpy as np
import glob
import os

#def cdf(seq):
#    print seq.shape
#    seen = list()

#    for i in range(64):
#        dat = seq[:,i:(i+1)]
#        _, inds, p = np.unique(dat, return_index=1, return_counts=True)
#        print max(list(p))
#        seen += list(inds)
#    return list(set(seen))

def cdf(seq):
    return list(np.vstack({tuple(row) for row in seq}))

for i in range(8):
    i += 1
    total_dat = []
    total_dat_len = 0
    total_dat_red_len = 0
    paths = sorted(glob.glob("csv/*/*%d*" % i))
    for path in paths:
        dat = np.asarray(np.genfromtxt(path, delimiter=","))
        dat_len = len(dat)
        total_dat.append(dat)
        dat_red_len = len(cdf(dat))
        base_path = os.path.basename(path)
        print("in %s total amount was %d unique amount was %d" % (base_path, dat_len, dat_red_len))
    total_dat = np.concatenate(total_dat, axis=0)
    total_dat_len = len(total_dat)
    total_dat_red_len = len(cdf(total_dat))
    print("Split: %d had %d data points of which %d was unique data points" % (i, total_dat_len, total_dat_red_len))
    print
