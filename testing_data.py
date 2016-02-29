import data
import numpy as np

xb_train, xb_valid, xb_test, _, _, _, \
    xs_train, xs_valid, xs_test, _, _, _ \
    = data.load(1)

print np.size(xb_train, axis=0).shape
print np.size(xb_valid, axis=0).shape
print np.size(xb_test, axis=0).shape

print xs_train.shape
print xs_valid.shape
print xs_test.shape

for idx_1 in range(size(xb_train, axis=0)):
    for idx_2 in range(size(xb_test, axis=0)):
        if xb_train[idx_1] == xb_test[idx_2]
            print "a no. in trap %d %d" % (idx_1, idx_2)
