# -*- coding: utf-8 -*-
import numpy as np
import os
import gzip
import sys

def save_gz(path, arr):
    tmp_path = os.path.join("/tmp", os.path.basename(path) + ".tmp.npy")
    np.save(tmp_path, arr)
    os.system("gzip -c %s > %s" % (tmp_path, path))
    os.remove(tmp_path)

def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)

if len(sys.argv) != 4:
    sys.exit("Usage: python create_validation_split.py <train_path> <path_to> <split%> say trainpath folder, not file")
train_path = sys.argv[1]
path_to = sys.argv[2]
split = float(sys.argv[3])
if split>=1:
    sys.exit("Split must be <1")

B_path = train_path
B_path += "/Btrn.npy.gz"
S_path = train_path
S_path += "/Strn.npy.gz"

B_dat = load_gz(B_path)
S_dat = load_gz(S_path)

B_n = np.size(B_dat,axis=0)
S_n = np.size(S_dat,axis=0)

B_seq = np.arange(0,B_n)
S_seq = np.arange(0,S_n)

np.random.shuffle(B_seq)
np.random.shuffle(S_seq)

B_n_val = int(np.floor(B_n * split))
S_n_val = int(np.floor(S_n * split))

B_split = np.zeros((B_n,1))
B_split[B_seq[0:B_n_val]] = 1

S_split = np.zeros((S_n,1))
S_split[S_seq[0:S_n_val]] = 1

B_path_to = path_to
B_path_to += "/Bsplit.npy.gz"
S_path_to = path_to
S_path_to += "/Ssplit.npy.gz"

save_gz(B_path_to, B_split)
save_gz(S_path_to, S_split)

print("Saved in the following directory: %s" %path_to)