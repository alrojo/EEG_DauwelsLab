import lasagne
from BatchNormLayer import batch_norm
#validate_every = 40
start_saving_at = 0
save_every = 5
#write_every_batch = 10

recurrent=True
epochs = 300
batch_size = 2048
N_CONV_A = 16
N_CONV_B = 16
N_CONV_C = 16
F_CONV_A = 3
F_CONV_B = 5
F_CONV_C = 7
N_L1 = 100
N_LSTM_F = 100
n_inputs = 1
num_classes = 1
seq_len = 64
optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.001,
    250: 0.0005,
    275: 0.00025,
}

def build_model():
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    l_dim_a = lasagne.layers.DimshuffleLayer(
	l_in, (0,2,1))

    l_conv_a = lasagne.layers.Conv1DLayer(
	incoming=l_dim_a, num_filters=N_CONV_A, pad='same',
	filter_size=F_CONV_A, stride=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_conv_a_b = batch_norm(l_conv_a)

    l_conv_b = lasagne.layers.Conv1DLayer(
	incoming=l_dim_a, num_filters=N_CONV_B, pad='same',
	filter_size=F_CONV_B, stride=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_conv_b_b = batch_norm(l_conv_b)
    
    l_conv_c = lasagne.layers.Conv1DLayer(
	incoming=l_dim_a, num_filters=N_CONV_C, pad='same',
	filter_size=F_CONV_C, stride=1, nonlinearity=lasagne.nonlinearities.rectify)
    l_conv_c_b = batch_norm(l_conv_c)

    l_c_a = lasagne.layers.ConcatLayer([l_conv_a_b, l_conv_b_b, l_conv_c_b], axis=1)
    l_dim_b = lasagne.layers.DimshuffleLayer(
	l_c_a, (0,2,1))
    l_c_b = lasagne.layers.ConcatLayer([l_in, l_dim_b], axis=2)
    # 2. First Dense Layer    
    l_reshape_a = lasagne.layers.ReshapeLayer(
        l_c_b, (-1,n_inputs+N_CONV_A*3))
    l_1 = lasagne.layers.DenseLayer(
        l_reshape_a, num_units=N_L1, nonlinearity=lasagne.nonlinearities.rectify)
    l_1_b = batch_norm(l_1)

    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_1_b, (-1, seq_len, N_L1))

    # 2. LSTM Layer
    l_forward = lasagne.layers.LSTMLayer(l_reshape_b, N_LSTM_F)

    l_reshape_c = lasagne.layers.ReshapeLayer(
        l_forward, (-1, N_LSTM_F))
    # Our output layer is a simple dense connection, with 1 output unit
#    l_2 = lasagne.layers.DenseLayer(
#	lasagne.layers.dropout(l_reshape_b, p=0.5), num_units=N_L2, nonlinearity=lasagne.nonlinearities.rectify)
    # 5. Output Layer
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape_c, num_units=num_classes, nonlinearity=lasagne.nonlinearities.sigmoid)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (-1, seq_len, num_classes))

    return l_in, l_out
