# Recreating Results

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

>> THEANO_FLAGS=device=cpu python mnist.py mlp 5

output should look something like this.

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


### BLAS and CUDA

Disclaimer, have not tested this as I have it setup on my machines already.

Please follow this guide from the BLAS part

https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04

remember to make the ~/.theanorc file as mentioned in the guide.

A cool tip: use the command.

>> nvidia-smi

To check for available GPU resources. Should look something like.

```
Mon Apr 25 04:42:52 2016       
+------------------------------------------------------+                       
| NVIDIA-SMI 352.63     Driver Version: 352.63         |                       
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:05:00.0     Off |                  N/A |
| 58%   84C    P2   141W / 250W |  11864MiB / 12287MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX TIT...  Off  | 0000:06:00.0     Off |                  N/A |
| 22%   57C    P8    17W / 250W |     23MiB / 12287MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX TIT...  Off  | 0000:09:00.0     Off |                  N/A |
| 24%   61C    P8    18W / 250W |     23MiB / 12287MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX TIT...  Off  | 0000:0A:00.0     Off |                  N/A |
| 53%   84C    P2   211W / 250W |  11864MiB / 12287MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      8592    C   python                                       11838MiB |
|    3      2619    C   python                                       11839MiB |
+-----------------------------------------------------------------------------+
```

### Testing CUDA and GPUs work

>> mkdir -p ~/code/mnist

>> cd ~/code/mnist

>> wget https://github.com/Lasagne/Lasagne/raw/master/examples/mnist.py

>> THEANO_FLAGS=device=gpu python mnist.py mlp 5

output should look something like this (notice it is much faster!):

```
Using gpu device 1: GeForce GTX TITAN X (CNMeM is disabled, cuDNN Version is too old. Update to v5, was 3007.)
Loading data...
Building model and compiling functions...
Starting training...
Epoch 1 of 5 took 1.878s
  training loss:		1.198237
  validation loss:		0.402080
  validation accuracy:		88.43 %
Epoch 2 of 5 took 1.898s
  training loss:		0.561074
  validation loss:		0.306817
  validation accuracy:		91.06 %
```

### Install EEG_Dauwelslab repo

Go to where you want to install the repo.

>> mkdir EEG_projects && cd EEG_projects

>> git clone https://github.com/alrojo/EEG_Dauwelslab.git

### Setting up training/validation data

Now given the dataset (you can enquire Justin Dauwels at jdauwels at ntu.edu.sg for access).

Place the train and validation data in the following folder `$PATH_TO_DIR/EEG_Dauwelslab/data/csv/train/`

Such that.

>> cd $PATH_TO_DIR/EEG_Dauwelslab/data/csv/train/

>> ls

```
Btrn1.csv  Btrn3.csv  Btrn5.csv  Btrn7.csv  Bval1.csv
Bval3.csv  Bval5.csv  Bval7.csv  Strn1.csv  Strn3.csv
Strn5.csv  Strn7.csv  Sval1.csv  Sval3.csv  Sval5.csv
Sval7.csv  Btrn2.csv  Btrn4.csv  Btrn6.csv  Btrn8.csv
Bval2.csv  Bval4.csv  Bval6.csv  Bval8.csv  Strn2.csv
Strn4.csv  Strn6.csv  Strn8.csv  Sval2.csv  Sval4.csv
Sval6.csv  Sval8.csv
```

Make the data into a numpy format

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

>> python create_data train

```
you csv path is: ./data/csv/train/*
Converting data ...
Opening: ./data/csv/train/Sval2.csv
Saved to ./data/numpy/train/Sval2.npy.gz
Opening: ./data/csv/train/Bval6.csv
Saved to ./data/numpy/train/Bval6.npy.gz
...
Opening: ./data/csv/train/Bval8.csv
Saved to ./data/numpy/train/Bval8.npy.gz
```

### Train the first model

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

Training models is performed by the following command.

Usage: python train.py <config_name> <CVNumber1,2,3> <num_epochs>

>> python train.py RNN 1 151

```
Using gpu device 1: GeForce GTX TITAN X (CNMeM is disabled, cuDNN Version is too old. Update to v5, was 3007.)
Setting sys parameters ...
Defining symbolic variables ...
Loading config file: 'RNN'
Setting config params ...
Optimizer: rmsprop
Lambda: 0.00010
Batch size: 2048
Experiment id: RNN-1-20160425-052845
Loading data ...
loadData started!
Preprocesssing data ...
Data shapes ...
(405429, 64, 1)
(64446, 64, 1)
(405429, 1)
(64446, 1)
(12763, 64, 1)
(2010, 64, 1)
(12763, 1)
(2010, 1)
DEBUG: max train values
84.267
Building network ...
  number of parameters: 41401
  layer output shapes:
    InputLayer                       (2048, 64, 1)
    LSTMLayer                        (2048, 64, 100)
    SliceLayer                       (2048, 100)
    DenseLayer                       (2048, 1)
Building cost function ...
Computing updates ...
Getting gradients ...
Print configuring updates ...
cut_norm: 20
Compiling functions ...
@@@@STARTING TO TRAIN@@@@
--------- Validation ----------
  validating: train loss
  average evaluation accuracy (train): 0.54948
  average evaluation AUC (train): 0.72353

  validating: valid loss
  average evaluation accuracy (valid): 0.64233
  average evaluation AUC (valid): 0.73729

Epoch 1 of 151
  setting learning rate to 0.0010000
Shuffling data
---------- Train ----------
  average training loss: 0.55832
  average training accuracy: 0.76965
  average auc: 0.82934
  00:01:05 since start (65.98 s)

  saving parameters and metadata
  stored in metadata/dump_RNN-1-20160425-052845
--------- Validation ----------
  validating: train loss
  average evaluation accuracy (train): 0.89933
  average evaluation AUC (train): 0.94912

  validating: valid loss
  average evaluation accuracy (valid): 0.90007
  average evaluation AUC (valid): 0.92259

Epoch 2 of 151
Shuffling data
---------- Train ----------
  average training loss: 0.32418
  average training accuracy: 0.89727
  average auc: 0.94564
  00:02:13 since start (67.51 s)

  saving parameters and metadata
  stored in metadata/dump_RNN-1-20160425-052845
--------- Validation ----------
  validating: train loss
  average evaluation accuracy (train): 0.93125
  average evaluation AUC (train): 0.98539

  validating: valid loss
  average evaluation accuracy (valid): 0.96249
  average evaluation AUC (valid): 0.98593

```

Cool tip: use `nohup <commands> > file &` to run script in background.

>> nohup python train.py RNN 1 151 > out_file &

>> ctrl+c

Cool tip: use `nvidia-smi` to see usage.

Cool tip: In the .theanorc file you specify the device. Setting device=gpu0 will make it use gpu numbered as 0, whereas device=gpu1 will make it use device 1. Using this will allow you to run multiple scripts simultanously.

Cool tip: during "testing" the script, you will generate metadata files. You are probably not interested in these. To remove, do as follows.

>> cd $PATH_TO_DIR/EEG_Dauwelslab/metadata/

>> rm -rf dump*

Remember to keep the other files, such that a `ls` in the metadata directory will give you

>> ls

```
FOR_DEBUGGING  FOR_ENSEMBLE  __init__.py
```

## Get results

### Training models

Run the following commands (to train a model for every split)

Cool tip: The model should only take up about 2400 mb memory. You should be able to run 4 concurrently on a Nvidia Titan X.

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

>> python train.py RNN 1 151

>> python train.py RNN 2 151

>> python train.py RNN 3 151

>> python train.py RNN 4 151

>> python train.py RNN 5 151

>> python train.py RNN 6 151

>> python train.py RNN 7 151

>> python train.py RNN 8 151

### Debugging all models

Use the debugging tool `debug_model.py` to find maximas on the validation set.

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
