# Setup

##Installation and setup

### Basics
>> sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy

### Python
>> wget "http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh"

>> bash Anaconda2-4.0.0-Linux-x86_64.sh

### Theano and Lasagne
>> pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

>> pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

(Append --user if you are not working as root and want it in your user directory)

### Testing Lasagne and Theano

>> mkdir -p ~/code/mnist

>> cd ~/code/mnist

>> wget https://github.com/Lasagne/Lasagne/raw/master/examples/mnist.py

>> python mnist.py mlp 5

output should look something like this:

```
Loading data...
Downloading train-images-idx3-ubyte.gz
Downloading train-labels-idx1-ubyte.gz
Downloading t10k-images-idx3-ubyte.gz
Downloading t10k-labels-idx1-ubyte.gz
Building model and compiling functions...
Starting training...
Epoch 1 of 5 took 11.483s
  training loss:		1.225167
  validation loss:		0.407454
  validation accuracy:		88.55 %
Epoch 2 of 5 took 11.299s
  training loss:		0.565556
  validation loss:		0.309781
  validation accuracy:		90.92 %
```

### BLAS check (optional)
Using direct BLAS can give a significant speed-up, but is not nessesary.

>> mkdir tmp && cd tmp

>> wget https://github.com/Theano/Theano/raw/master/theano/misc/check_blas.py

>> python check_blas.py

Should say something like:

`Total execution time: 31.37s on CPU (with direct Theano binding to blas).`

### CUDA install

Disclaimer, have not tested this as I have CUDA on my machines.

Please follow this guide the Nvidia CUDA header. 

https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04

### Testing it all works





# About

## EEG_DauwelsLab
This github project is dedicated to spike detection in EEG data from epileptical patients.

The code is Produced by A. Rosenberg Johansen<sup>1</sup>.

The project is supervisioned by PhD. Justin Dauwels<sup>2</sup> lab.

The data is supplied by MD. Sydney Cash<sup>3</sup> and MD. M. Brandon Westover<sup>3</sup>.

The code as been used for ICASSP 2016 submission, using a dataset of five patients, and is now being used for a journal paper with a dataset of 100 patients.

<sup>1</sup>: Technical University of Denmark, DTU Compute, Lyngby, Denmark

<sup>2</sup>: Nanyang Technological University, School of Electrical and Electronic Engineering, Singapore

<sup>3</sup>: Massechusetts General Hospital, Neurology Department, and Harvard Medical School, USA

## Data
The data set consists of 30 minutes EEG recordings sampled from over 100 patients.
The dataset has been preprocessed into 64 length sliding windows, and labelled by M.D. to either be containing "epileptical spike" or "no epileptical spike".
The data is split into eight equal size training, validation and test splits with no overlapping patients.

## Code
The code can be run through *train.py*, dumps will be saved in *metadata/* and can be evaluated using *debug_metadata.py*. Configurations for the different models used can be found in *configurations/*

## Current model performance(averaged over all CV splits)
Training AUC: 0.999
Valid AUC: 0.999
Test AUC: 0.998

### LSTMLayer w. L^2 = 0.0001
As of 4'th April 2016

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/RNN.py

Training AUC: 0.999

Validation AUC: 0.999


### LSTMLayer
As of 23'th December 2015

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/ConvBN_DenseBN_LSTM.py

Training AUC: 0.998

Validation AUC: 0.997

### MultiConv w. L^2 = 0.0005
As of 11'th December 2015

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/MultiConv.py

Training AUC: 0.996

Validation AUC: 0.961


### MLP w. L^2 = 0.0005
As of 11'th December

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/MLP.py

Training AUC: 0.998

Validation AUC: 0.973



### Logistic Regression
As of 5'th December 2015

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/LogisticRegression.py

Training AUC: 0.835

Validation AUC: 0.672

### MLP w. L^2 = 0.0001
As of 5'th December 2015

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/MLP.py

Training AUC: 0.999

Validation AUC: 0.972

## Next to train
Done training with LSTM, giving 0.999 AUC Train, 0.999 AUC valid and 0.998 AUC test.
