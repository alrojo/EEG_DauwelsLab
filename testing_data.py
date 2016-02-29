import data
import numpy as np

xb_train, xb_valid, xb_test, _, _, _, \
    xs_train, xs_valid, xs_test, _, _, _ \
    = data.load_data(1)

print np.shape(xb_train)
print np.size(xb_train, axis=0)
print np.size(xb_valid, axis=0)
print np.size(xb_test, axis=0)

print xs_train.shape
print xs_valid.shape
print xs_test.shape

for idx_1 in range(np.size(xs_train, axis=0)):
    if (idx_1 % 1000) == 0:
        print idx_1
    for idx_2 in range(np.size(xs_test, axis=0)):
        nums=0
        for elem_1, elem_2 in zip(xs_train[idx_1].tolist(), xs_test[idx_2].tolist()):
            if (abs(elem_1-elem_2))<0.0000001:
                nums+=1
            if nums > 60:
                print "a trap! %d %d" % (idx_1, idx_2)
