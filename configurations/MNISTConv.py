# -*- coding: utf-8 -*-
#### DONT RUN THIS, THIS IS FOR SANITY CHECK ONLY

import lasagne as nn
import utils
import theano.tensor as T

#validate_every = 40
start_saving_at = 0
save_every = 5
#write_every_batch = 10

epochs = 25
batch_size = 64
optimizer = "nag"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.0002,
    250: 0.00005,
    275: 0.000025,
}

def build_model():
    l_in = nn.layers.InputLayer((None, 1, 28, 28))
    l_conv1_a = nn.layers.Conv2DLayer(l_in, num_filters=16, filter_size=(3, 3), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_conv1_b = nn.layers.Conv2DLayer(l_conv1_a, num_filters=32, filter_size=(3, 3), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_pool1 = nn.layers.MaxPool2DLayer(l_conv1_b, pool_size=(2, 2))    
    l_conv2_a = nn.layers.Conv2DLayer(l_pool1, num_filters=64, filter_size=(3,3), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_conv2_b = nn.layers.Conv2DLayer(l_conv2_a, num_filters=32, filter_size=(1,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_h1 = nn.layers.DenseLayer(nn.layers.DropoutLayer(l_conv2_b), num_units=256, nonlinearity=nn.nonlinearities.leaky_rectify)
    l_out = nn.layers.DenseLayer(l_h1, num_units=10, nonlinearity=utils.softmax)
    
    return l_in, l_out
