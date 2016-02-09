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
import string

# repo libs
import utils
import data

print "Setting sys parameters ..."
if len(sys.argv) != 4:
	sys.exit("Usage: python train.py <config_name> <CVNumber1,2,3> <num_epochs>")
config_name = sys.argv[1]
CVsplit = sys.argv[2]
num_epochs = int(sys.argv[3])

print "Defining symbolic variables ..."
sym_x = T.tensor3('x')
sym_t = T.matrix('t')

print "Loading config file: '%s'" % config_name
config = importlib.import_module("configurations.%s" % config_name)
print "Setting config params ..."
optimizer = config.optimizer
print "Optimizer: %s" % optimizer
lambda_reg = config.lambda_reg
print "Lambda: %.5f" % lambda_reg
#num_epochs = config.epochs
batch_size = config.batch_size
print "Batch size: %d" % batch_size
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s-%s" % (config_name, CVsplit, timestamp)
metadata_path = "metadata/dump_%s" % experiment_id

print "Experiment id: %s" % experiment_id

print "Loading data ..."
xb_train, xb_valid, xb_test, tb_train, tb_valid, tb_test, \
	xs_train, xs_valid, xs_test, ts_train, ts_valid, ts_test \
	= data.load_data(CVsplit)

print "Preprocesssing data ..."
dat = utils.add_dims_seq([xb_train, xb_valid, xb_test, xs_train, xs_valid, xs_test])
xb_train = dat[0]
xb_valid = dat[1]
xb_test = dat[2]
xs_train = dat[3]
xs_valid = dat[4]
xs_test = dat[5]

print "Data shapes ..."
dataset = [xb_train, xb_valid, xb_test, tb_train, tb_valid, tb_test, \
	xs_train, xs_valid, xs_test, ts_train, ts_valid, ts_test]
for dat in dataset:
	print(dat.shape)

nb_train = np.size(xb_train, axis=0)
ns_train = np.size(xs_train, axis=0)

print "DEBUG: max train values"
print(xb_train.max())

print("Building network ...")
l_in, l_out = config.build_model()

Xt = np.zeros((2048, 64, 1), dtype='float32')
all_layers = nn.layers.get_all_layers(l_out)
num_params = nn.layers.count_params(l_out)
print("  number of parameters: %d" % num_params)
print("  layer output shapes:")
for layer in all_layers:
	name = string.ljust(layer.__class__.__name__, 32)
	print("    %s %s" % (name, nn.layers.get_output(layer, sym_x).eval({sym_x: Xt}).shape))

print("Building cost function ...")
out_train = nn.layers.get_output(
	l_out, deterministic=False)
out_eval = nn.layers.get_output(
	l_out, deterministic=True)
TOL=1e-5

lambda_reg = config.lambda_reg
params = nn.layers.get_all_params(l_out, regularizable=True)
reg_term = sum(T.sum(p**2) for p in params)
cost = T.mean(utils.Cross_Ent(T.clip(out_train, TOL, 1-TOL), sym_t))
cost += lambda_reg * reg_term
print "Retreiving all parameters ..."
all_params = nn.layers.get_all_params(l_out, trainable=True)

if hasattr(config, 'set_weights'):
	print "Setting weights from config file ..."
	nn.layers.set_all_param_values(l_out, config.set_weights())

print("Computing updates ...")
if hasattr(config, 'learning_rate_schedule'):
	learning_rate_schedule = config.learning_rate_schedule              # Import learning rate schedule
else:
	learning_rate_schedule = { 0: config.learning_rate }
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

print "Getting gradients ..."
all_grads = T.grad(cost, all_params)
print "Print configuring updates ..."
cut_norm = config.cut_grad
updates, norm_calc = nn.updates.total_norm_constraint(all_grads, max_norm=cut_norm, return_norm=True)

if optimizer == "rmsprop":
	updates = nn.updates.rmsprop(updates, all_params, learning_rate)
elif optimizer == "adadelta":
	updates = nn.updates.adadelta(updates, all_params, learning_rate)
elif optimizer == "adagrad":
	updates = nn.updates.adagrad(updates, all_params, learning_rate)
elif optimizer == "nag":
	momentum_schedule = config.momentum_schedule
	momentum = theano.shared(np.float32(momentum_schedule[0]))
	updates = nn.updates.nesterov_momentum(updates, all_params, learning_rate, momentum)
else:
	sys.exit("please choose either <rmsprop/adagrad/adadelta/nag> in configfile")


print("Compiling functions ...")

train = theano.function(
	[sym_x, sym_t], [cost, out_train, norm_calc], updates=updates, on_unused_input='warn')

predict = theano.function(
	[sym_x], out_eval, on_unused_input='ignore')
print('@@@@STARTING TO TRAIN@@@@')
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
all_roc_tpr_eval_train = []
all_roc_tpr_eval_valid = []
all_roc_tpr_eval_test = []
all_roc_fpr_eval_train = []
all_roc_fpr_eval_valid = []
all_roc_fpr_eval_test = []
all_roc_thresholds_eval_train = []
all_roc_thresholds_eval_valid = []
all_roc_thresholds_eval_test = []

for epoch in range(num_epochs):
	if 1==1:#(i + 1) % config.validate_every == 0:
		print "--------- Validation ----------"
		sets = [('train', True, xb_train, xs_train, tb_train, ts_train,
			all_accuracy_eval_train, all_auc_eval_train,
			all_roc_tpr_eval_train, all_roc_fpr_eval_train, all_roc_thresholds_eval_train),
			('valid', True, xb_valid, xs_valid, tb_valid, ts_valid,
			all_accuracy_eval_valid, all_auc_eval_valid,
			all_roc_tpr_eval_valid, all_roc_fpr_eval_valid, all_roc_thresholds_eval_valid),
			('test', True, xb_test, xs_test, tb_test, ts_test,
			all_accuracy_eval_test, all_auc_eval_test,
			all_roc_tpr_eval_test, all_roc_fpr_eval_test, all_roc_thresholds_eval_test)]
		for subset, Print, xb, xs, tb, ts, all_accuracy, all_auc, all_roc_tpr, all_roc_fpr, all_roc_thresholds in sets:
			X = np.vstack((xb,xs))
			y = np.vstack((tb,ts))
			n = np.size(X,axis=0)
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
				acc_eval = utils.accuracy(predictions, y)
				all_accuracy.append(acc_eval)

				auc_eval = utils.auc(predictions, y)
				all_auc.append(auc_eval)
    
				roc_eval_fpr, roc_eval_tpr, roc_eval_thresholds = utils.roc(predictions, y)
				all_roc_fpr.append(roc_eval_fpr)
				all_roc_tpr.append(roc_eval_tpr)
				all_roc_thresholds.append(roc_eval_thresholds)
				if Print:
					print "  validating: %s loss" % subset
					print "  average evaluation accuracy (%s): %.5f" % (subset, acc_eval)
					print "  average evaluation AUC (%s): %.5f" % (subset, auc_eval)
					print
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

print("---------- Train ----------")

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
		loss, out, norm = train(x_batch, t_batch)
		# print(norm)
	if np.isnan(loss):
		raise RuntimeError("Loss is NaN.")
	preds.append(out)
	losses.append(loss)
	label.append(t_batch)
	predictions = np.concatenate(preds, axis = 0)
	labels = np.concatenate(label, axis = 0)
	loss_train = np.mean(losses)
	all_losses_train.append(loss_train)
	acc_train = utils.accuracy(predictions, labels)
	all_accuracy_train.append(acc_train)
	auc_train = utils.auc(predictions, labels)
	all_auc_train.append(auc_train)
	if 1==1:
		print "  average training loss: %.5f" % loss_train
		print "  average training accuracy: %.5f" % acc_train
		print "  average auc: %.5f" % auc_train    


	now = time.time()
	time_since_start = now - start_time
	time_since_prev = now - prev_time
	prev_time = now
#	est_time_left = time_since_start * num_epochs
#	eta = datetime.now() + timedelta(seconds=est_time_left)
#	eta_str = eta.strftime("%c")
	print "  %s since start (%.2f s)" % (utils.hms(time_since_start), time_since_prev)
#	print "  estimated %s to go (ETA: %s)" % (utils.hms(est_time_left), eta_str)
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
			'roc_tpr_eval_train': all_roc_tpr_eval_train,
			'roc_tpr_eval_valid': all_roc_tpr_eval_valid,
			'roc_tpr_eval_test': all_roc_tpr_eval_test,
			'roc_fpr_eval_train': all_roc_fpr_eval_train,
			'roc_fpr_eval_valid': all_roc_fpr_eval_valid,
			'roc_fpr_eval_test': all_roc_fpr_eval_test,
			'roc_thresholds_eval_train': all_roc_thresholds_eval_train,
			'roc_thresholds_eval_valid': all_roc_thresholds_eval_valid,
			'roc_thresholds_eval_test': all_roc_thresholds_eval_test,
			'time_since_start': time_since_start,
			'i': i,
			}, f, pickle.HIGHEST_PROTOCOL)

		print "  stored in %s" % metadata_path
