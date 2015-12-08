import lasagne as nn
import utils

#validate_every = 40
start_saving_at = 0
save_every = 5
#write_every_batch = 10

epochs = 25
batch_size = 64
optimizer = "rmsprop"
lambda_reg = 0.0005
cut_grad = 20

learning_rate_schedule = {
    0: 0.001,
    250: 0.0005,
    275: 0.00025,
}

def build_model():
    l_in = nn.layers.InputLayer((None, 1, 64, 1))
    l_conv1_a = nn.layers.Conv2DLayer(l_in, num_filters=8, filter_size=(3,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_conv1_b = nn.layers.Conv2DLayer(l_in, num_filters=8, filter_size=(5,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_conv1_c = nn.layers.Conv2DLayer(l_in, num_filters=8, filter_size=(7,1), nonlinearity=nn.nonlinearities.leaky_rectify)
    l_con_a = nn.layers.ConcatLayer([l_conv1_a, l_conv1_b, l_conv1_c], axis=2)
    l_h1 = nn.layers.DenseLayer(nn.layers.DropoutLayer(l_con_a), num_units=200, nonlinearity=nn.nonlinearities.leaky_rectify)
    l_out = nn.layers.DenseLayer(l_h1, num_units=1, nonlinearity=utils.softmax)
    
    return l_in, l_out
