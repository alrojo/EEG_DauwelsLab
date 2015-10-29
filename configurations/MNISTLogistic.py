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
    l_out = nn.layers.DenseLayer(l_in, num_units=10, nonlinearity=utils.softmax)
    
    return l_in, l_out
