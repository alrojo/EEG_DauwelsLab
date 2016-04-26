from __future__ import division
import sys
import numpy as np
import glob
import sklearn


if len(sys.argv) < 3:
	sys.exit("Usage: python eval_predictions.py <CVsplit> <model> [subset=test]")

prob = float(sys.argv[1])
model = sys.argv[2]
total_cm = np.array([[0., 0.], [0., 0.]])
total_AUC = 0.0
total_TPR = 0.0
total_SPC = 0.0
total_PPV = 0.0
for CVsplit in range(8):
    CVsplit += 1
    CVsplit = str(CVsplit)
    print("---- %s ----" % CVsplit)
    p_path_all = "predictions/" + model + "/" + CVsplit + "/*"
    #print p_path_all
    predictions_path_all = glob.glob(p_path_all)
    #print(len(predictions_path_all))

    first_run = True
    for predictions_path in predictions_path_all:
#        print(predictions_path)
        if first_run:
            predictions = np.load(predictions_path)
            first_run = False
	else:
            predictions = predictions + np.load(predictions_path)
#    print "shape of predictions and max"
#    print(predictions.shape)
#    print(predictions.max())
    predictions = predictions / len(predictions_path_all) # evening it out
#    print(predictions.max())
    import data

    if len(sys.argv) == 4:
        subset = sys.argv[3]
        assert subset in ['train', 'valid', 'test', 'train_valid']
    else:
        subset = "test"

    if subset == "test":
        xb_test, tb_test, _, ts_test 	= data.load_test(CVsplit)
    elif subset == "train":
        sys.exit(subset + ": not implemented yet")
    elif subset == "train_valid":
        sys.exit(subset + ": not implemented yet")
    else:
        sys.exit(subset + ": not implemented yet")

    t = np.vstack((tb_test, ts_test))
    n = np.size(t, axis=0)

    import utils
    AUC = utils.auc(predictions, t)
    total_AUC += AUC

    predictions = predictions>prob
    predictions = (predictions-1)*-1
    hard_preds = predictions#>prob
    t = (t-1)*-1

## Below is for getting FPs
#    FP_list = []
#    counter = 0
#    for idx in range(len(tb_test)):
#        if predictions[idx]==0:#predictions[idx]:
#            FP_list.append(np.array([xb_test[idx]]))
    #        print(idx)
#    FP_list = np.concatenate(FP_list, axis=0)
#    print(FP_list.shape)
#    np.save('FP_8.npy', FP_list)
#    assert False
    cm = sklearn.metrics.confusion_matrix(t, hard_preds)
    total_cm += cm #np.true_divide(cm, cm.sum())

    TN = cm[1, 1]
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    print("TP %d, FN %d" % (TP, FN))
    print("FP %d, TN %d" % (FP, TN))
    TPR = TP / (TP + FN)
    SPC = TN / (FP + TN)
    PPV = TP / (TP + FP)
    print("TPR: %.5f" % TPR)
    print("SPC: %.5f" % SPC)
    print("PPV: %.5f" % PPV)

    total_TPR += TPR
    total_SPC += SPC
    total_PPV += PPV
    print("AUC (%s) is: %.5f" % (subset, AUC))

print("FINAL RESULTS")
print(total_cm)
#print(np.true_divide(total_cm, 8.0))
print("AUC = %.5f" % (total_AUC/8.0))
print("TPR = %.5f" % (total_TPR/8.0))
print("SPC = %.5f" % (total_SPC/8.0))
print("PPV = %.5f" % (total_PPV/8.0))
