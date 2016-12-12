# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import time

import numpy as np

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

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz').astype('float32')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz').astype('float32')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz').astype('float32')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz').astype('float32')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

save_gz('./data/X_train.npy.gz', X_train)
save_gz('./data/X_val.npy.gz', X_val)
save_gz('./data/X_test.npy.gz', X_test)
save_gz('./data/y_train.npy.gz', y_train)
save_gz('./data/y_val.npy.gz', y_val)
save_gz('./data/y_test.npy.gz', y_test)
