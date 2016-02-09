import lasagne as nn
import utils

#validate_every = 40
start_saving_at = 0
save_every = 20
#write_every_batch = 10

recurrent=False
epochs = 300
batch_size = 128
optimizer = "rmsprop"
lambda_reg = 0.0005
cut_grad = 20
N_L1 = 200
seq_len = 64
n_inputs = 1
learning_rate_schedule = {
    0: 0.001,
    250: 0.0005,
    275: 0.00025,
}

def build_model():
    l_in = nn.layers.InputLayer(shape=(None, seq_len, n_inputs))
    l_dim = nn.layers.DimshuffleLayer(l_in, (0,2,1))
    l_reshape_a = nn.layers.ReshapeLayer(l_dim, (-1, seq_len))
    l_h1 = nn.layers.DenseLayer(nn.layers.DropoutLayer(l_reshape_a), num_units=N_L1, nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l_out = nn.layers.DenseLayer(l_h1, num_units=1, nonlinearity=utils.sigmoid)
    
    return l_in, l_out
