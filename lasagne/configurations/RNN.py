import numpy
import lasagne

# Setting configurations
numpy.random.seed(1)

validate_every = 1
start_saving_at = 0
save_every = 1

epochs = 300
batch_size = 2048
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
}


# Define the model
def build_model():

	# 1. Input layer
	l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))

	# 2. LSTM Layer
	l_forward = lasagne.layers.LSTMLayer(l_in, N_LSTM_F)
	l_forward_last = lasagne.layers.SliceLayer(l_forward, indices=-1, axis=1)

	# 3. Output Layer
	l_out = lasagne.layers.DenseLayer(
	l_forward_last, num_units=num_classes, nonlinearity=lasagne.nonlinearities.sigmoid)

	return l_in, l_out
