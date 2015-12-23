import lasagne

#validate_every = 40
start_saving_at = 0
save_every = 5
#write_every_batch = 10

epochs = 300
batch_size = 2048
N_LSTM_F = 100
n_inputs = 1
num_classes = 2
seq_len = 700
optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
    0: 0.001,
    250: 0.0005,
    275: 0.00025,
}

def build_model(mask=None):
    # 1. Input layer
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))
    # 2. LSTM Layer
    l_forward = lasagne.layers.LSTMLayer(l_in, N_LSTM_F, mask=mask)

    l_reshape_b = lasagne.layers.ReshapeLayer(
        l_forward, (-1, N_LSTM_F))
    # Our output layer is a simple dense connection, with 1 output unit
#    l_2 = lasagne.layers.DenseLayer(
#	lasagne.layers.dropout(l_reshape_b, p=0.5), num_units=N_L2, nonlinearity=lasagne.nonlinearities.rectify)
    # 5. Output Layer
    l_recurrent_out = lasagne.layers.DenseLayer(
        l_reshape_b, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    # Now, reshape the output back to the RNN format
    l_out = lasagne.layers.ReshapeLayer(
        l_recurrent_out, (batch_size, seq_len, num_classes))

    return l_in, l_out
