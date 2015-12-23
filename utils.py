# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T
import time
import gzip
import os
from sklearn import metrics as sk

def add_dims_seq(dat):    
    for i in range(len(dat)):
        dat[i] = np.expand_dims(dat[i], 2)
#        dat[i] = np.expand_dims(dat[i], 3) only using for 2D Conv
    return dat        

def sigmoid(x):
    return 1 / (1 + T.exp(-x))

def Cross_Ent(y, t):
    return -t * T.log(y) - (1 - t) * T.log(1 - y)
    
def accuracy(p, t):
    y = p>0.5
#    print("p sum = %.5f" % p.sum())
#    print("t sum = %.5f" % t.sum())
#    print("y sum = %.5f" % y.sum())
    r = sk.accuracy_score(y.astype('int'),t.astype('int'))
    #r = np.mean(y==t)
    return r;

def auc(p, t):
    return sk.roc_auc_score(t, p);

def roc(p, t):
    return sk.roc_curve(t, p)
    
def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)
    
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
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