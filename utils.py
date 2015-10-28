# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T
import time

def add_dims_csv(dat):    
    for i in range(len(dat)):
        dat[i] = np.expand_dims(dat[i], 1)
        dat[i] = np.expand_dims(dat[i], 3)
    return dat        

def softmax(x):
    return 1 / (1 + T.exp(-x))

def Cross_Ent(y, t):
    return -t * T.log(y) - (1 - t) * T.log(1 - y)
    
def accuracy(p, t):
    return 0;

def auc(p, t):
    return 0;
    
def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)
    
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())