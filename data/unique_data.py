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
    seq = [tuple([format(elem, '.8g') for elem in row]) for row in seq] 
    uniq = dict()
    for idx, row in enumerate(seq):
        if row in uniq:
            uniq[row] += [idx]
        else:
            uniq[row] = [idx]
    return uniq #list(np.vstack({tuple(row) for row in seq}))
for i in range(1): # use 8 for all data
    i += 1
    total_dat = dict()
    total_dict = dict()
    paths = sorted(glob.glob("csv/*/*%d*" % i))
    for path in paths:
        print("loading: %s ..." % path)
        dat = np.genfromtxt(path, delimiter=",")
        total_dat[path] = dat
        total_dict[path] = cdf(dat)

    for path in paths:
        dat_len = len(total_dat[path])
        print("calculating uniques and duplicates for %s ..." % path)
        dat_red_len = len(total_dict[path])
        dups = dat_len - dat_red_len
        base_path = os.path.basename(path)
        print("in %s total amount was %d unique amount was %d, %d was duplicates" % (base_path, dat_len, dat_red_len, dups))

    print("    @@@@@@@@@")
    print("Running for duplicates within split")
    print("    @@@@@@@@@")
    f_dup = open("duplicates.txt", "w")
    for path in paths:
        print("running duplicates for %s" % path, file=f_dup)
        john = total_dict[path]
        for k, v in total_dict[path].items():
            if len(v)>1:
                print("DUPLICATES WITHIN SPLIT AT:", file=f_dup)
                print(v, file=f_dup)
    f_dup.close()
    print("    @@@@@@@@@")
    print("Running for overlapping duplicates")
    print("    @@@@@@@@@")
    i = 0.0
    f_overlap = open("overlapping.txt", "w")
    for path in paths:
        print("running overlapping duplicates for %s" % path, file=f_overlap)
        own_dict = total_dict[path]
        own_base = os.path.basename(path)
        for k, v in total_dict.items():
            d_base = os.path.basename(k)
            if k == path:
                #print("aint checking %s do'h ..." % k)
                continue # don't check own path
            else:
                print("  checking for %s" %k, file=f_overlap)
                for k1, v1 in v.items():
                    if k1 in own_dict:
                        i += 1.0
                        print("%s indexes:" % own_base, file=f_overlap)
                        print(own_dict[k1], file=f_overlap)
                        print("%s indexes:" % d_base, file=f_overlap)
                        print(v1, file=f_overlap)
                        print("", file=f_overlap)
    f_overlap.close()
    i /= 2 # because duplicates are run from both sides
    print("total amount of overlapping duplicates: %f" % i)
