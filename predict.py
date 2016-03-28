import sys
import numpy as np
import importlib
import lasagne as nn
import theano
from theano import tensor as T
import os
import glob

# repo libs
import data
import utils

if not (2 <= len(sys.argv) <= 3):
	sys.exit("usage: python predict.py <split> [subset=test]")

sym_y = T.imatrix('target_output')
sym_x = T.tensor3()

CVsplit = sys.argv[1]
m_paths = "metadata/FOR_ENSEMBLE/" + CVsplit + "/*"
print "m_paths"
print m_paths
metadata_path_all = glob.glob(m_paths)
print "length of metadata_path_all"
print(len(metadata_path_all))

if len(sys.argv) >= 3:
	subset = sys.argv[2]
	assert subset in ['train', 'valid', 'test', 'train_valid']
else:
	subset = "test"

num_seq = 64

if subset == "test":
	xb_test, _, xs_test, _ = data.load_test(CVsplit)
	dat = utils.add_dims_seq([xb_test, xs_test])
	xb_test = dat[0]
	xs_test = dat[1]
elif subset == "train":
	sys.exit(subset + ": not implemented yet")
elif subset == "train_valid":
	sys.exit(subset + ": not implemented yet")
else:
	sys.exit(subset + ": not implemented yet")

X = np.vstack((xb_test, xs_test))
n = np.size(X, axis=0)

print("X shape:")
print(X.shape)
print("n:")
print(n)

for metadata_path in metadata_path_all:

	print "Loading metadata file %s" % metadata_path

	metadata = np.load(metadata_path)
	config_name = metadata['config_name']
	config = importlib.import_module("configurations.%s" % config_name)
	print "Using configurations: '%s'" % config_name
	print "Build model"

	l_in, l_out = config.build_model()

	print "Build eval function"
	inference = nn.layers.get_output(
		l_out, sym_x, deterministic=True)
	print "Load parameters"
	nn.layers.set_all_param_values(l_out, metadata['param_values'])
	print "Compile functions"
	predict = theano.function([sym_x], inference)
	print "Predict"
	predictions = []
	batch_size = config.batch_size
	num_batches = n // batch_size

	for i in range(num_batches):
		idx = range(i*batch_size, (i+1)*batch_size)
		x_batch = X[idx]
		p = predict(x_batch)
		predictions.append(p)

	if n - (num_batches * batch_size):
		print "Computing rest"
		rest = n - num_batches * batch_size
		idx = range(n-rest, n)
		x_batch = X[idx]
		out = predict(x_batch)
		predictions.append(out)

	predictions = np.concatenate(predictions, axis = 0)
	predictions_path = os.path.join("predictions/" + CVsplit, os.path.basename(metadata_path).replace("dump_", "predictions_").replace(".pkl", ".npy"))

	print "Storing predictions in %s" % predictions_path
	np.save(predictions_path, predictions)
