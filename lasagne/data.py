# -*- coding: utf-8 -*-
# General libs
import numpy as np
import glob

# User libs
import utils

def convert_data(paths_from):
	file_paths = glob.glob(paths_from)
	if file_paths == []:
		sys.exit("no paths found ..!")
	for path in file_paths:
		print "Opening: %s" % path
		dat = np.genfromtxt(path, delimiter=',').astype('float32')
		save_path = path.replace('.csv', '.npy.gz')
		save_path = save_path.replace('csv', 'numpy')
		utils.save_gz(save_path,dat)
		print "Saved to %s" % save_path

def load_data(CVsplit):
	print "loadData started!"
	xb_train = utils.load_gz('data/numpy/train/Btrn%s.npy.gz' % CVsplit).astype('float32')
	tb_train = np.zeros((xb_train.shape[0],1), dtype='float32')
	xs_train = utils.load_gz('data/numpy/train/Strn%s.npy.gz' % CVsplit).astype('float32')
	ts_train = np.ones((xs_train.shape[0],1), dtype='float32')
	xb_valid = utils.load_gz('data/numpy/train/Bval%s.npy.gz' % CVsplit).astype('float32')
	tb_valid = np.zeros((xb_valid.shape[0],1), dtype='float32')
	xs_valid = utils.load_gz('data/numpy/train/Sval%s.npy.gz' % CVsplit).astype('float32')
	ts_valid = np.ones((xs_valid.shape[0],1), dtype='float32')

	return xb_train, xb_valid, tb_train, tb_valid, xs_train, xs_valid, ts_train, ts_valid

def load_test(CVsplit):
	print "loadTest started!"
	xb_test = utils.load_gz('data/numpy/test/Btst%s.npy.gz' % CVsplit).astype('float32')
	tb_test = np.zeros((xb_test.shape[0],1), dtype='float32')
	xs_test = utils.load_gz('data/numpy/test/Stst%s.npy.gz' % CVsplit).astype('float32')
	ts_test = np.ones((xs_test.shape[0],1), dtype='float32')

	return xb_test, tb_test, xs_test, ts_test
