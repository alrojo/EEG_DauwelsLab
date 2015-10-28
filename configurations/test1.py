import lasagne

#validate_every = 40
start_saving_at = 0
save_every = 5
#write_every_batch = 10

epochs = 300
batch_size = 64
optimizer = "nag"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.011,
    250: 0.005,
    275: 0.0025,
}

def build_model():
    l_in = lasagne.layers.InputLayer((None, 1, 64, 1))
    l_conv1_a = lasagne.layers.Conv2DLayer(l_in, num_filters=8, filter_size=(3,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_conv1_b = lasagne.layers.Conv2DLayer(l_in, num_filters=8, filter_size=(5,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_conv1_c = lasagne.layers.Conv2DLayer(l_in, num_filters=8, filter_size=(7,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_con_a = lasage.layers.ConcatLayer([l_conv1_a, l_conv1_b, l_conv1_c], axis=2)
    l_h1 = lasagne.layers.DenseLayer(nn.layers.DropoutLayer(l_con_a), num_units=150, nonlinearity=nn.nonlinearities.leaky_rectify)
    l_out = lasagne.layers.DenseLayer(l_h1, num_units=1, nonlinearity=utils.softmax)
    
    return l_in, l_out
