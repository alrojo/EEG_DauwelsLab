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
    
# This function requires that all data is placed in data/dat%s/%s/ %(mset,data_type) as a .npy.gz file.
# Future work: Make a stochastic loader (amount of data might exceed memory capacity)
def load_data(mset, data_type):
    print('Loading data ...')
    if data_type == 'csv':
        # Should make an error handler if path is not valid and return options ...
        xb_train = load_gz('data/dat%s/%s/Btrn.npy.gz' % (mset,data_type)).astype('float32')
        tb_train = np.zeros((xb_train.shape[0],1), dtype='float32')
        xs_train = load_gz('data/dat%s/%s/Strn.npy.gz' % (mset,data_type)).astype('float32')
        ts_train = np.ones((xs_train.shape[0],1), dtype='float32')
        xb_test = load_gz('data/dat%s/%s/Btst.npy.gz' % (mset,data_type)).astype('float32')
        tb_test = np.zeros((xb_test.shape[0],1), dtype='float32')
        xs_test = load_gz('data/dat%s/%s/Stst.npy.gz' % (mset,data_type)).astype('float32')
        ts_test = np.ones((xs_test.shape[0],1), dtype='float32')
    else:
        xb_train = load_gz('data/dat%s/%s/trn/B_pngs.npy.gz' % (mset,data_type)).astype('float32')
        tb_train = np.zeros((xb_train.shape[0],1), dtype='float32')
        xs_train = load_gz('data/dat%s/%s/trn/S_pngs.npy.gz' % (mset,data_type)).astype('float32')
        ts_train = np.ones((xs_train.shape[0],1), dtype='float32')
        xb_test = load_gz('data/dat%s/%s/tst/B_pngs.npy.gz' % (mset,data_type)).astype('float32')
        tb_test = np.zeros((xb_test.shape[0],1), dtype='float32')
        xs_test = load_gz('data/dat%s/%s/tst/S_pngs.npy.gz' % (mset,data_type)).astype('float32')
        ts_test = np.ones((xs_test.shape[0],1), dtype='float32')
    print('Making train/val splits ...')
    Bsplit = load_gz('./data/dat%s/Bsplit.npy.gz' % mset).astype('float32').astype('int').ravel()
    Ssplit = load_gz('./data/dat%s/Ssplit.npy.gz' % mset).astype('float32').astype('int').ravel()
    print(xb_train.shape)
    print(Bsplit.shape)
    xb_valid = xb_train[Bsplit]
    tb_valid = tb_train[Bsplit]
    xs_valid = xs_train[Ssplit]
    ts_valid = ts_train[Ssplit]
    xb_train = xb_train[(-Bsplit+1)]
    tb_train = tb_train[(-Bsplit+1)]
    xs_train = xs_train[(-Ssplit+1)]
    ts_train = ts_train[(-Ssplit+1)]
    
    return xb_train, xb_valid, xb_test, tb_train, tb_valid, tb_test, xs_train, xs_valid, xs_test, ts_train, ts_valid, ts_test