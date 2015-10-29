# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy as np
import lasagne as nn
import sys
import importlib
from datetime import datetime, timedelta
import cPickle as pickle
import time
import gzip

# repo libs
import utils
import data

# Sys parameters
if len(sys.argv) != 5:
    sys.exit("Usage: python train.py <config_name> <CVNumber> <data_type> <num_epochs>")
config_name = sys.argv[1]
mset = sys.argv[2]
data_type = sys.argv[3]
num_epochs = int(sys.argv[4])

# Configuration file
config = importlib.import_module("configurations.%s" % config_name)
optimizer = config.optimizer
lambda_reg = config.lambda_reg
#num_epochs = config.epochs
batch_size = config.batch_size
print "Using configurations: '%s'" % config_name

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_path = "metadata/dump_%s" % experiment_id

print "Experiment id: %s" % experiment_id

# Data loading
xb_train, xb_valid, xb_test, tb_train, tb_valid, tb_test, \
    xs_train, xs_valid, xs_test, ts_train, ts_valid, ts_test \
    = data.load_data(mset, data_type)

if data_type == 'csv':
    print "@@@so it's a CSV@@@"
    dat = utils.add_dims_csv([xb_train, xb_valid, xb_test, xs_train, xs_valid, xs_test])
    xb_train = dat[0]
    xb_valid = dat[1]
    xb_test = dat[2]
    xs_train = dat[3]
    xs_valid = dat[4]
    xs_test = dat[5]

#### REMOVE THIS
def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)
        
xb_train = load_gz('./data/X_train.npy.gz')
tb_train = np.expand_dims(load_gz('./data/y_train.npy.gz'),1)
xb_valid = load_gz('./data/X_val.npy.gz')
tb_valid = np.expand_dims(load_gz('./data/y_val.npy.gz'),1)
xb_test = load_gz('./data/X_test.npy.gz')
tb_test = np.expand_dims(load_gz('./data/y_test.npy.gz'),1)
xs_train = xb_train
ts_train = tb_train
xs_valid = xb_valid
ts_valid = tb_valid
xs_test = xb_test
ts_test = tb_test
####
john = [xb_train, xb_valid, xb_test, tb_train, tb_valid, tb_test, \
    xs_train, xs_valid, xs_test, ts_train, ts_valid, ts_test]
for i in john:
    print(i.shape)

nb_train = np.size(xb_train, axis=0)
ns_train = np.size(xs_train, axis=0)

# define symbolic Theano variables
x = T.ftensor4('x')
t = T.matrix('t')

# define model: logistic regression
print("Building network ...")
l_in, l_out = config.build_model()

print("Building cost function ...")
out_train = nn.layers.get_output(l_out, x, deterministic=False)
out_eval = nn.layers.get_output(l_out, x, deterministic=True)

cost = T.mean(utils.Cross_Ent(out_train, t))
# Retrieve all parameters from the network
all_params = nn.layers.get_all_params(l_out)
# Setting the weights
if hasattr(config, 'set_weights'):
    nn.layers.set_all_param_values(l_out, config.set_weights())
# Compute SGD updates for training
print("Computing updates ...")
if hasattr(config, 'learning_rate_schedule'):
    learning_rate_schedule = config.learning_rate_schedule              # Import learning rate schedule
else:
    learning_rate_schedule = { 0: config.learning_rate }
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

all_grads = T.grad(cost, all_params)
if optimizer == "rmsprop":
    updates = nn.updates.rmsprop(all_grads, all_params, learning_rate)
elif optimizer == "adagrad":
    updates = nn.updates.adagrad(all_grads, all_params, learning_rate)
elif optimizer == "nag":
    updates = nn.updates.nesterov_momentum(all_grads, all_params, learning_rate, 0.9)
else:
    sys.exit("please choose either <rmsprop/adagrad/nag> as second input param")
# Theano functions for training and computing cost
print "config.batch_size %d" %batch_size
if hasattr(config, 'build_model'):
    print("has build model")
print("Compiling functions ...")

# compile theano functions
train = theano.function([x, t], [cost, out_train], updates=updates)
predict = theano.function([x], out_eval)
print('Functions loaded')
print('@@@@STARTING TO TRAIN@@@@')
# Start timers
start_time = time.time()
prev_time = start_time

all_losses_train = []
all_accuracy_train = []
all_auc_train = []
all_accuracy_eval_train = []
all_accuracy_eval_valid = []
all_accuracy_eval_test = []
all_auc_eval_train = []
all_auc_eval_valid = []
all_auc_eval_test = []
for epoch in range(num_epochs):
    if 1==1:#(i + 1) % config.validate_every == 0:
        sets = [('train', xb_train, xs_train, tb_train, ts_train, all_accuracy_eval_train, all_auc_eval_train),
                ('valid', xb_valid, xs_valid, tb_valid, ts_valid, all_accuracy_eval_valid, all_auc_eval_valid),
                ('test', xb_test, xs_test, tb_test, ts_test, all_accuracy_eval_test, all_auc_eval_test)]
        for subset, xb, xs, tb, ts, all_accuracy, all_auc in sets:
            X = np.vstack((xb,xs))
            y = np.vstack((tb,ts))
            n = np.size(X,axis=0)
            print "  validating: %s loss" % subset
            preds = []
            num_batches = n // batch_size
            for i in range(num_batches):
                idx = range(i*batch_size, (i+1)*batch_size)
                x_batch = X[idx]
                out = predict(x_batch)
                preds.append(out)
            # Computing rest
            rest = n - num_batches * batch_size
            idx = range(n-rest, n)
            x_batch = X[idx]
            out = predict(x_batch)
            preds.append(out)
            # Making metadata
            predictions = np.concatenate(preds, axis = 0)
            print(predictions.shape)
            print(y.shape)
            acc_eval = np.mean(np.equal(np.argmax(predictions, axis=1), y),
                      dtype=theano.config.floatX)
            #acc_eval = utils.accuracy(predictions, y)
            all_accuracy.append(acc_eval)
            print "  average evaluation accuracy (%s): %.5f" % (subset, acc_eval)
            auc_eval = utils.auc(predictions, y)
    #        all_auc.append(auc_eval)
    #        print "  average evaluation AUC (%s): %.5f" % (subset, auc_eval)
    print
    if (epoch % 5) == 0:
        print "Epoch %d of %d" % (epoch + 1, num_epochs)

    if epoch in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[epoch])
        print "  setting learning rate to %.7f" % lr
        learning_rate.set_value(lr)
    print "Shuffling data"
    seq_names_b = np.arange(0,nb_train)
    seq_names_s = np.arange(0,ns_train)
    np.random.shuffle(seq_names_b)     
    np.random.shuffle(seq_names_s) 
    xb_train = xb_train[seq_names_b]
    xs_train = xs_train[seq_names_s]
    num_batches = nb_train // batch_size

    print("Train ...")

    losses = []
    preds = []
    label = []
    for i in range(num_batches):
        xb_batch = xb_train[i:i + batch_size]
        shuf = np.arange(0,ns_train) # Shuffles the spikes at every batch
        np.random.shuffle(shuf)
        xs_batch = xs_train[shuf[i:i + batch_size]]
        x_batch = np.vstack((xb_batch,xs_batch))
        tb_batch = tb_train[i:i + batch_size]
        ts_batch = ts_train[shuf[i:i + batch_size]]
        t_batch = np.vstack((tb_batch,ts_batch))
        loss, out = train(x_batch, t_batch)
        preds.append(out)
        losses.append(loss)
        label.append(t_batch)

    predictions = np.concatenate(preds, axis = 0)
    labels = np.concatenate(label, axis = 0)
    loss_train = np.mean(losses)
    all_losses_train.append(loss_train)
    acc_train = np.mean(np.equal(np.argmax(predictions, axis=1), labels),
                      dtype=theano.config.floatX)
    #acc_train = utils.accuracy(predictions, labels)
    all_accuracy_train.append(acc_train)
    #auc_train = utils.auc(predictions, labels)
    #all_auc_train.append(auc_train)
    if 1==1:
        print "  average training loss: %.5f" % loss_train
        print "  average training accuracy: %.5f" % acc_train
    #    print "  average auc: %.5f" % auc_train
        


    now = time.time()
    time_since_start = now - start_time
    time_since_prev = now - prev_time
    prev_time = now
#    est_time_left = time_since_start * num_epochs
#    eta = datetime.now() + timedelta(seconds=est_time_left)
#    eta_str = eta.strftime("%c")
    print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
#    print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
    print

    if (epoch >= config.start_saving_at) and ((epoch % config.save_every) == 0):
        print "  saving parameters and metadata"
        with open((metadata_path + "-%d" % (epoch) + ".pkl"), 'w') as f:
            pickle.dump({
                'config_name': config_name,
                'param_values': nn.layers.get_all_param_values(l_out),
                'losses_train': all_losses_train,
                'accuracy_train': all_accuracy_train,
                'auc_train': all_auc_train,
                'accuracy_eval_valid': all_accuracy_eval_valid,
                'accuracy_eval_train': all_accuracy_eval_train,
                'accuracy_eval_test': all_accuracy_eval_test,
                'auc_eval_train': all_auc_eval_train,
                'auc_eval_valid': all_auc_eval_valid,
                'auc_eval_test': all_auc_eval_test,
                'time_since_start': time_since_start,
                'i': i,
                }, f, pickle.HIGHEST_PROTOCOL)

        print "  stored in %s" % metadata_path