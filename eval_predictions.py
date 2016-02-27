import sys
import numpy as np
import glob

if len(sys.argv) != 4:
	sys.exit("Usage: python eval_predictions.py <predictions_path> <CVsplit [subset=test]")

CVsplit = sys.argv[2]
p_path_all = sys.argv[1] + CVsplit + "/*"
print p_path_all
predictions_path_all = glob.glob(p_path_all)
print(len(predictions_path_all))

first_run = False
for predictions_path in predictions_path_all:
	print(predictions_path)
	if not first_run:
		predictions = np.load(predictions_path)
		first_run = True
	else:
		predictions = predictions + np.load(predictions_path)
print "shape of predictions and max"
print(predictions.shape)
print(predictions.max())
predictions = predictions / len(predictions_path_all) # evening it out
print(predictions.max())
import data

if len(sys.argv) == 4:
	subset = sys.argv[3]
	assert subset in ['train', 'valid', 'test', 'train_valid']

if subset == "test":
	_, _, _, _, _, tb_test, \
		_, _, _, _, _, ts_test \
		= data.load_data(CVsplit)
elif subset == "train":
	sys.exit(subset + ": not implemented yet")
elif subset == "train_valid":
	sys.exit(subset + ": not implemented yet")
else:
	sys.exit(subset + ": not implemented yet")

t = np.vstack((tb_test, ts_test))
n = np.size(t, axis=0)

print("t shape:")
print(t.shape)
print("n:")
print(n)

import utils
acc = utils.auc(predictions, t)

print("AUC (%s) is: %.5f" % (subset, acc))
