# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import sys

# For working with .png extensions
import skimage.transform
import skimage.io
if len(sys.argv) != 3:
    sys.exit("Usage: python dataConverter.py <.csv/.png> <globPath> Notive that \
    csv needs complete path where as png only works with path to one directory at a time")

extension = sys.argv[1]
file_path = sys.argv[2]

def save_gz(path, arr):
    tmp_path = os.path.join("/tmp", os.path.basename(path) + ".tmp.npy")
    np.save(tmp_path, arr)
    os.system("gzip -c %s > %s" % (tmp_path, path))
    os.remove(tmp_path)

file_paths = glob.glob(file_path)

# Case of .csv
if extension == '.csv':
    for path in file_paths:
        print "Opening: %s" % path
        dat = np.genfromtxt(path, delimiter=',').astype('float32')
        save_path = path.replace(extension, ".npy.gz")
        save_gz(save_path,dat)
        print "Saved to %s" % save_path

# Case of .png
if extension == '.png':
    # First split B and S
    B_paths = []
    S_paths = []
    for path in file_paths:
        if os.path.basename(path)[0] == 'S':
            S_paths.append(path)
        else:
            if os.path.basename(path)[0] == 'B':
                B_paths.append(path)
            else:
                sys.exit("Path is not S or B: %s" % path)
    # Make the .npy.gz files from all the .pngs
    S_pngs = np.zeros((len(S_paths),3,96,96),dtype='float32')
    B_pngs = np.zeros((len(B_paths),3,96,96),dtype='float32')
    sets = [['/S_pngs.npy.gz',S_pngs, S_paths], ['/B_pngs.npy.gz',B_pngs, B_paths]]
    for subset, pngs, paths in sets:
        for idx, path in enumerate(paths):
            im = skimage.io.imread(path)
            pngs[idx,:,:,:] = im.transpose(2,0,1)
        save_path = os.path.dirname(paths[0])
        save_path += subset
        save_gz(save_path, pngs)
        print "Saved to %s" % save_path
        
