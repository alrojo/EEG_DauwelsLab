# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:51:13 2015

@author: s145706
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob

if not (len(sys.argv) == 3):
    sys.exit("Usage: python debug_metadata.py <train/test> <topX> <plots>")

test = sys.argv[1]
topX = int(sys.argv[2])
plots = sys.argv[1]
#metadata_path = sys.argv[3]

#MLPs = glob.glob("./metadata/*MLP*")
#Convs = glob.glob("./metadata/save/*")
RNNs = glob.glob("./metadata/FOR_DEBUGGING/dump_LSTM-1-201603*")

metasets = [(RNNs, "Recurrent Neural Network", "r")]#, (MLPs, "Multi Layer Perceptron", "b")]
plt.figure(1)
for metadata_paths, name, colours in metasets: # colours not used anymore
    print name
    all_aucs_train = []
    all_aucs_valid = []
    all_aucs_test = []
    all_roc_tprs_train = []
    all_roc_tprs_valid = []
#    all_roc_tprs_test = []
    all_roc_fprs_train = []
    all_roc_fprs_valid = []
#    all_roc_fprs_test = []
    all_roc_thresholds_train = []
    all_roc_thresholds_valid = []
#    all_roc_thresholds_test = []
    for metadata_path in metadata_paths:
        print "----"
#        print "Loading metadata file %s" % metadata_path
        metadata = np.load(metadata_path)
        acc_eval_valid = metadata['accuracy_eval_valid']
        acc_eval_train = metadata['accuracy_eval_train']
#        acc_eval_test = metadata['accuracy_eval_test']
        auc_eval_valid = metadata['auc_eval_valid']
        auc_eval_train = metadata['auc_eval_train']
#        auc_eval_test = metadata['auc_eval_test']
        roc_tpr_eval_valid = metadata['roc_tpr_eval_valid']
        roc_tpr_eval_train = metadata['roc_tpr_eval_train']
#        roc_tpr_eval_test = metadata['roc_tpr_eval_test']
        roc_fpr_eval_valid = metadata['roc_fpr_eval_valid']
        roc_fpr_eval_train = metadata['roc_fpr_eval_train']
#        roc_fpr_eval_test = metadata['roc_fpr_eval_test']
        roc_thresholds_eval_valid = metadata['roc_thresholds_eval_valid']
        roc_thresholds_eval_train = metadata['roc_thresholds_eval_train']
#        roc_thresholds_eval_test = metadata['roc_thresholds_eval_test']
        
        # Running through top X best and making an averaged top X validations
        best_val = np.zeros(shape=topX, dtype='float32')
#        best_test = np.zeros(shape=topX, dtype='float32')
        max_val = np.zeros(shape=topX, dtype=int)
        #for x,v,t in zip(auc_eval_train, auc_eval_valid, auc_eval_test):
        #    print "@train = %.5f - validation = %.5f - test = %.5f" % (np.asarray(x), np.asarray(v), np.asarray(t))
        print metadata_path
        for i in range(topX):
            max_val[i] = np.argmax(auc_eval_valid)
            print max_val[i]
            all_aucs_train.append(auc_eval_train[max_val[i]])
            all_aucs_valid.append(auc_eval_valid[max_val[i]])
#            all_aucs_test.append(auc_eval_test[max_val[i]])
            all_roc_tprs_train.append(roc_tpr_eval_train[max_val[i]])
            all_roc_tprs_valid.append(roc_tpr_eval_valid[max_val[i]])
#            all_roc_tprs_test.append(roc_tpr_eval_test[max_val[i]])
            all_roc_fprs_train.append(roc_fpr_eval_train[max_val[i]])
            all_roc_fprs_valid.append(roc_fpr_eval_valid[max_val[i]])
#            all_roc_fprs_test.append(roc_fpr_eval_test[max_val[i]])
            all_roc_thresholds_train.append(roc_thresholds_eval_train[max_val[i]])
            all_roc_thresholds_valid.append(roc_thresholds_eval_valid[max_val[i]])
#            all_roc_thresholds_test.append(roc_thresholds_eval_test[max_val[i]])
            if test == "test":
                print "Train = %.5f - Validation = %.5f - Test = %.5f" %(np.asarray(auc_eval_train[max_val[i]]), np.asarray(auc_eval_valid[max_val[i]]), np.asarray(auc_eval_test[max_val[i]]))
            else:
                print "Train = %.5f - Validation = %.5f" %(np.asarray(all_aucs_train).mean(), np.asarray(all_aucs_valid).mean())
            auc_eval_valid[max_val[i]] = 0.0 # To make it take the next max value
#    cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#    for i, col in enumerate(cols):
#        plt.plot(all_roc_fprs_test[i], all_roc_tprs_test[i], col)
#    ymin = 0.98
#    ymax = 1.0
#    xmin = 0.0
#    xmax = 0.02
#    plt.axis([xmin,xmax,ymin,ymax])
#    plt.subplot(222)
#    plt.plot(all_roc_fprs_test[4], all_roc_tprs_test[4])
#    plt.subplot(222)
#    plt.plot(all_roc_fprs_test[5], all_roc_tprs_test[5])
    print("@@@OVERALL@@@")
    if test == "test":
        print "Train = %.5f - Validation = %.5f - Test = %.5f" %(np.asarray(all_aucs_train).mean(), np.asarray(all_aucs_valid).mean(), np.asarray(all_aucs_test).mean())
    else:
        print "Train = %.5f - Validation = %.5f" %(np.asarray(all_aucs_train).mean(), np.asarray(all_aucs_valid).mean())
    print
#plt.show()
#            print max_val[i]
                
#            if test == "test":
#                print "Best"
#                print "Validation = %.5f - Test = %.5f" %(best_val[0], best_test[0])
#                print "Averaged on top %d" % topX
#                print "Validation = %.5f - Test = %.5f" %(best_val.mean(), best_test.mean())
#            else:
#                print "Best"
#                print "Validation = %.5f" % best_val[0]
#                print "Averaged on top %d" % topX
#                print "Validation = %.5f" % best_val.mean()
                                
