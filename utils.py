# utility file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics as sk


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out


def auc(t, p):
    return sk.roc_auc_score(t, p)


def acc(t, p):
    predictions = np.argmax(p, axis=1)
    return np.mean(t == predictions)


def roc(t, p):
    return sk.roc_curve(t, p)


def conf_matrix(p, t, num_classes):
    if p.ndim == 1:
        p = one_hot(p, num_classes)
    if t.ndim == 1:
        t = one_hot(t, num_classes)
    return np.dot(p.T, t)
