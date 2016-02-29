import data

xb_train, xb_valid, xb_test, _, _, _, \
    xs_train, xs_valid, xs_test, _, _, _ \
    = data.load(1)

print xb_train.shape
print xb_valid.shape
print xb_test.shape

print xs_train.shape
print xs_valid.shape
print xs_test.shape
