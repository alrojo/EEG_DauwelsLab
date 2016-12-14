# logistic regression

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn, sparse_softmax_cross_entropy_with_logits

import data

tb_log_freq = 50
save_freq = 500
valid_every = 50
max_to_keep = 200
batch_size = 2000
num_classes = 2
num_iterations = 1501
num_features = 64
learning_rate = 0.0001
clip_norm = 1


data_gen = lambda split: data.gen_data(split, num_iterations=num_iterations, batch_size=batch_size)

def model():
    print("building model ...")
    with tf.variable_scope('train'):
        print("building model ...")
        X_pl = tf.placeholder(tf.float32, [None, num_features])
        print("X_pl", X_pl.get_shape())
        t_pl = tf.placeholder(tf.int32, [None,])
        print("t_pl", t_pl.get_shape())
        is_training_pl = tf.placeholder(tf.bool)
        X_bn = batch_norm(X_pl, is_training=is_training_pl)
        print("X_bn", X_bn.get_shape())
        l1 = fully_connected(X_pl, num_outputs=100, activation_fn=relu)#, normalizer_fn=batch_norm)
        print("l1", l1.get_shape())
        l1_drop = dropout(l1, is_training=is_training_pl)
        print("l1_drop", l1_drop.get_shape())
        l_out = fully_connected(l1_drop, num_outputs=num_classes, activation_fn=None)
        print("l_out", l_out.get_shape())
        l_out_softmax = tf.nn.softmax(l_out)
        tf.contrib.layers.summarize_variables()

    with tf.variable_scope('metrics'):
        loss = sparse_softmax_cross_entropy_with_logits(l_out, t_pl)
        print("loss", loss.get_shape())
        loss = tf.reduce_mean(loss)
        print("loss", loss.get_shape())
        tf.summary.scalar('train/loss', loss)
        argmax = tf.to_int32(tf.argmax(l_out, 1))
        print("argmax", argmax.get_shape())
        correct = tf.to_float(tf.equal(argmax, t_pl))
        print("correct,", correct.get_shape())
        accuracy = tf.reduce_mean(correct)
        print("accuracy", accuracy.get_shape())

    with tf.variable_scope('optimizer'):
        print("building optimizer ...")
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, global_norm = (
            tf.clip_by_global_norm(gradients, clip_norm))
        clipped_grads_and_vars = zip(clipped_gradients, variables)

        tf.summary.scalar('train/global_gradient_norm', global_norm)

        train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

    return X_pl, t_pl, is_training_pl, l_out, l_out_softmax, loss, accuracy, train_op, global_step


def setup_validation_summary():
    acc = tf.placeholder(tf.float32)
    auc = tf.placeholder(tf.float32)
    valid_summaries = [
        tf.summary.scalar('validation/acc', acc),
        tf.summary.scalar('validation/auc', auc)
    ]
    return tf.merge_summary(valid_summaries), acc, auc

if __name__ == '__main__':
    model = model()
