# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:03:37 2015

@author: alexander
"""
import numpy as np

def myconf(y, t):
    conf_mat = np.zeros((2,2))
    conf_mat[0,0] = np.multiply(y, t).sum()
    conf_mat[0,1] = np.multiply((y*-1+1), t).sum()
    conf_mat[1,0] = np.multiply(y, (t*-1+1)).sum()
    conf_mat[1,1] = np.multiply((y*-1+1), (t*-1+1)).sum()
    return conf_mat

def one_hot(vec, m=None):
#    print(vec.shape)
    if m is None:
        m = int(np.max(vec)) + 1

    return np.eye(m)[vec]

def conf_matrix(p, t, num_classes):
    if p.ndim == 1:
        p = one_hot(p, num_classes)
    if t.ndim == 1:
        t = one_hot(t, num_classes)
    return np.dot(p.T, t)

def printAcc(y, t, type, thresh):
    acc = np.mean(y == t)
    conf_mat = myconf(y, t)
    #conf_mat = conf_matrix(t, y, 2)
    TP = conf_mat[0,0]
    FP = conf_mat[1,0]
    FN = conf_mat[0,1]
    TN = conf_mat[1,1]
#    TPR = TP/(TP+FN)
#    SPC = TN/(FP+TN)

#    print "THRESH: %.2f" % thresh
#    print "accuracy_%s: %.2f" % (type, acc)
#    print "CONF MAT_%s" % type
#    print conf_mat
#    print "TPR_%s: %.3f" % (type, TPR)
#    print "SPC_%s: %.3f" % (type, SPC)
#    print "Recall_%s: %.3f" % (type, recall)
    return acc, TP, FP, FN, TN

