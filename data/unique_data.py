import numpy as np
import glob
import os

def check_dataset_fast(seq):
    seen = list()
    for i in range(64):
        dat = seq[:,(i-1):i]
        _, inds = np.unique(dat,return_index=1)
        seen += list(inds)
    return list(set(seen))

for i in range(8):
    i += 1
    total_dat_len = 0
    total_dat_red_len = 0
    paths = sorted(glob.glob("csv/*/*%d*" % i))
    for path in paths:
        dat = np.asarray(np.genfromtxt(path, delimiter=","))
        dat_len = len(dat)
        total_dat_len += dat_len
        dat_red_len = len(check_dataset_fast(dat))
        total_dat_red_len += dat_red_len
        base_path = os.path.basename(path)
        print("in %s total amount was %d unique amount was %d" % (base_path, dat_len, dat_red_len))
    print("Split: %d had %d data points of which %d was unique data points" % (i, total_dat_len, total_dat_red_len))
    print
