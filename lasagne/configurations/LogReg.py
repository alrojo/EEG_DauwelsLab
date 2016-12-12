import lasagne
#from BatchNormLayer import batch_norm
validate_every = 1
start_saving_at = 0
save_every = 1
#write_every_batch = 10

epochs = 300
batch_size = 2048
n_inputs = 1
num_classes = 1
seq_len = 64
optimizer = "rmsprop"
lambda_reg = 0.0001
cut_grad = 20

learning_rate_schedule = {
	0: 0.001,
}

def build_model():
	# 1. Input layer
	l_in = lasagne.layers.InputLayer(shape=(None, seq_len, n_inputs))

	# 2. Output Layer
	l_out = lasagne.layers.DenseLayer(
	l_in, num_units=num_classes, nonlinearity=lasagne.nonlinearities.sigmoid)

	return l_in, l_out
