# -*- coding: utf-8 -*-
import numpy as np
import os
import gzip
import sys
import glob

paths_csv = "./data/csv/*"

def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)

def save_gz(path, arr):
    tmp_path = os.path.join("/tmp", os.path.basename(path) + ".tmp.npy")
    np.save(tmp_path, arr)
    os.system("gzip -c %s > %s" % (tmp_path, path))
    os.remove(tmp_path)

def convert_data():
    file_paths = glob.glob(paths_csv)
    for path in file_paths:
        print "Opening: %s" % path
        dat = np.genfromtxt(path, delimiter=',').astype('float32')
        save_path = path.replace('.csv', ".npy.gz")
        save_path = save_path.replace('csv', 'numpy')
        save_gz(save_path,dat)
        print "Saved to %s" % save_path

def load_data(CVsplit):
    print "loadData started!"
    if(len(glob.glob('./data/csv/*'))!=len(glob.glob('./data/numpy/*'))):
        print "converting data ..."
        convert_data();
    xb_train = load_gz('data/numpy/Btrn%s.npy.gz' % CVsplit).astype('float32')
    tb_train = np.zeros((xb_train.shape[0],1), dtype='float32')
    xs_train = load_gz('data/numpy/Strn%s.npy.gz' % CVsplit).astype('float32')
    ts_train = np.ones((xs_train.shape[0],1), dtype='float32')
    xb_test = load_gz('data/numpy/Btst%s.npy.gz' % CVsplit).astype('float32')
    tb_test = np.zeros((xb_test.shape[0],1), dtype='float32')
    xs_test = load_gz('data/numpy/Stst%s.npy.gz' % CVsplit).astype('float32')
    ts_test = np.ones((xs_test.shape[0],1), dtype='float32')
    xb_valid = load_gz('data/numpy/Bval%s.npy.gz' % CVsplit).astype('float32')
    tb_valid = np.zeros((xb_valid.shape[0],1), dtype='float32')
    xs_valid = load_gz('data/numpy/Sval%s.npy.gz' % CVsplit).astype('float32')
    ts_valid = np.ones((xs_valid.shape[0],1), dtype='float32')
    
    return xb_train, xb_valid, xb_test, tb_train, tb_valid, tb_test, xs_train, xs_valid, xs_test, ts_train, ts_valid, ts_test