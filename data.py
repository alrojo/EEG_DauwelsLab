# -*- coding: utf-8 -*-
# General libs
import glob
import numpy as np
import subprocess
import os

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


TRAIN_PATH = "data/numpy/train/Btrn1.npy.gz"
def load_train(CVsplit):
    if not os.path.isfile(TRAIN_PATH):
        print("Downloading and extracting train ...")
        subprocess.call("bash download_train.sh", shell=True)
        subprocess.call("python create_data.py train", shell=True)
    else:
        print "Train already downloaded ..."
    xb_train = utils.load_gz('data/numpy/train/Btrn%s.npy.gz' % CVsplit).astype('float32')
    tb_train = np.zeros((xb_train.shape[0]), dtype='float32')
    xs_train = utils.load_gz('data/numpy/train/Strn%s.npy.gz' % CVsplit).astype('float32')
    ts_train = np.ones((xs_train.shape[0]), dtype='float32')
    xb_valid = utils.load_gz('data/numpy/train/Bval%s.npy.gz' % CVsplit).astype('float32')
    tb_valid = np.zeros((xb_valid.shape[0]), dtype='float32')
    xs_valid = utils.load_gz('data/numpy/train/Sval%s.npy.gz' % CVsplit).astype('float32')
    ts_valid = np.ones((xs_valid.shape[0]), dtype='float32')

    return xb_train, xb_valid, tb_train, tb_valid, xs_train, xs_valid, ts_train, ts_valid


TEST_PATH = "data/numpy/test/Btst1.npy.gz"
def load_test(CVsplit):
    if not os.path.isfile(TEST_PATH):
        print("Downloading and extracting test ...")
        subprocess.call("bash download_test.sh", shell=True)
        subprocess.call("python create_data.py test", shell=True)
    else:
        print "Test already downloaded ..."
    xb_test = utils.load_gz('data/numpy/test/Btst%s.npy.gz' % CVsplit).astype('float32')
    tb_test = np.zeros((xb_test.shape[0]), dtype='float32')
    xs_test = utils.load_gz('data/numpy/test/Stst%s.npy.gz' % CVsplit).astype('float32')
    ts_test = np.ones((xs_test.shape[0]), dtype='float32')

    return xb_test, tb_test, xs_test, ts_test


def load_data(split, train=True):
    dict_out = dict()
    if train:
        xb_train, xb_valid, tb_train, tb_valid, xs_train, xs_valid, \
            ts_train, ts_valid = load_train(split)

        dict_out['Xb_train'] = xb_train
        dict_out['Xs_train'] = xs_train
        dict_out['tb_train'] = tb_train
        dict_out['ts_train'] = ts_train
        X_valid = np.concatenate([xb_valid, xs_valid], axis=0)
        t_valid = np.concatenate([tb_valid, ts_valid], axis=0)
        idcs_valid = list(range(X_valid.shape[0]))
        np.random.shuffle(idcs_valid)
        dict_out['X_valid'] = X_valid[idcs_valid]
        dict_out['t_valid'] = t_valid[idcs_valid]
    else:
        xb_test, tb_test, xs_test, ts_test = load_test(split)
        X_test = np.concatenate([xb_test, xs_test], axis=0)
        t_test = np.concatenate([tb_test, ts_test], axis=0)
        idcs_test = list(range(X_test.shape[0]))
        np.random.shuffle(idcs_test)
        dict_out['X_test'] = X_test[idcs_test]
        dict_out['t_test'] = t_test[idcs_test]

    return dict_out


class gen_data():
    def __init__(self, split, num_iterations=10000, batch_size=100,
                 data_fn=load_data, train=True):
        print("initializing data generator!")
        self._num_iterations = num_iterations
        self._batch_size = batch_size
        self._data_dict = load_data(split, train)
        print(self._data_dict.keys())
        if 'Xb_train' in self._data_dict.keys():
            if 'tb_train' in self._data_dict.keys():
                print("Training is found!")
                self._idcs_train_b = np.arange(0, self._data_dict['Xb_train'].shape[0])
                self._idcs_train_s = np.arange(0, self._data_dict['Xs_train'].shape[0])
                self._num_features = self._data_dict['Xb_train'].shape[-1]
        if 'X_valid' in self._data_dict.keys():
            if 't_valid' in self._data_dict.keys():
                print("Valid is found!")
                self._idcs_valid = np.arange(0, self._data_dict['X_valid'].shape[0])
                self._num_features = self._data_dict['X_valid'].shape[-1]
        if 'X_test' in self._data_dict.keys():
            if 't_test' in self._data_dict.keys():
                print("Test is found!")
                self._idcs_test = np.arange(0, self._data_dict['X_test'].shape[0])
                self._num_features = self._data_dict['X_test'].shape[-1]



    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train_b)
        np.random.shuffle(self._idcs_train_s)

    def _batch_init(self):
        batch_holder = dict()
        batch_holder["X"] = np.zeros((self._batch_size, self._num_features), dtype="float32")
        batch_holder["t"] = np.zeros((self._batch_size,), dtype="float32")
        return batch_holder

    def gen_valid(self):
        batch = self._batch_init()
        i = 0
        for idx in self._idcs_valid:
            batch['X'][i] = self._data_dict['X_valid'][idx]
            batch['t'][i] = self._data_dict['t_valid'][idx]
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init()
                i = 0
        if i != 0:
            yield batch, i

    def gen_test(self):
        batch = self._batch_init()
        i = 0
        for idx in self._idcs_test:
            batch['X'][i] = self._data_dict['X_test'][idx]
            batch['t'][i] = self._data_dict['t_test'][idx]
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init()
                i = 0
        if i != 0:
            yield batch, i

    def gen_train(self):
        batch = self._batch_init()
        nb = self._data_dict['Xb_train'].shape[0]
        ns = self._data_dict['Xs_train'].shape[0]
        iteration = 0
        i = 0
        while True:
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train_s:
                batch['X'][i] = self._data_dict['Xb_train'][idx]
                batch['X'][i+1] = self._data_dict['Xs_train'][idx]
                batch['t'][i] = self._data_dict['tb_train'][idx]
                batch['t'][i+1] = self._data_dict['ts_train'][idx]
                i += 2
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init()
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break
            else:
                continue
            break


if __name__ == '__main__':
    d_train = load_data(1)
    for key, value in d_train.iteritems():
        print(key, value.shape)
    d_test = load_data(1, train=False)
    for key, value in d_test.iteritems():
        print(key, value.shape)
    batch_size = 10
    num_features = 64
    num_iterations = 10
    data_generator = gen_data(split=4, num_iterations=num_iterations, batch_size=batch_size)
    tot_batches = []
    for batch in data_generator.gen_train():
        tot_batches.append(batch)
        assert batch['X'].shape == (batch_size, num_features)
        assert batch['t'].shape == (batch_size,)
    assert len(tot_batches) == num_iterations
    sum_idx = 0
    for batch, idx in data_generator.gen_valid():
        sum_idx += idx
        assert batch['X'].shape == (batch_size, num_features)
        assert batch['t'].shape == (batch_size,)
    assert sum_idx == len(data_generator._idcs_valid)
    data_generator = gen_data(split=4, num_iterations=num_iterations,
                              batch_size=batch_size, train=False)
    sum_idx = 0
    for batch, idx in data_generator.gen_test():
        sum_idx += idx
        assert batch['X'].shape == (batch_size, num_features)
        assert batch['t'].shape == (batch_size,)
    assert sum_idx == len(data_generator._idcs_test)
