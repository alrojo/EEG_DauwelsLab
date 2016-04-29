# Recreating Results

## Installation and setup

Installation and setup consists of installing some basic dependancies, Theano, Lasagne, CUDA, BLAS.

The non-trivial part of this installation is getting CUDA and Theano to work.

For detailed Theano, CUDA and BLAS guide, Please visit.

http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

The guide below is a modification of https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04

As I prefer using Conda over apt-get python install.

### Basics
>> sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev

### Python
>> wget "http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh"

>> bash Anaconda2-4.0.0-Linux-x86_64.sh

### Theano and Lasagne
>> pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

>> pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

(Use the flag --user if you are not working as root and want it in your user directory)

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

Now given the train/valid dataset (you can enquire Justin Dauwels at jdauwels at ntu.edu.sg for access).

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

>> python train.py RNN 1 150

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

Cool tip: In the `~/.theanorc` file you specify the device. Setting `device=gpu0` will make it use gpu numbered as 0, whereas device=gpu1 will make it use device 1. Using this will allow you to utilize multiple GPUs concurrently.

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

This will take ~24 hours on a GPU

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

>> bash batch_train.sh

### Debugging all models
Use the debugging tool `debug_model.py` to find maximas on the validation set.

This tool will take a given `path` (being the latest epoch, epoch no. 149) and a `topX` argument. The `topX` argument will list the highest validation epochs in sorted order.

Usage `python debug_model.py <path> <topX>`

Debug all of the models, you should get the following response.

Example `python debug_model.py metadata/dump_RNN-1-20160425-055244-149.pkl 20`

Do this for every split (`...RNN-1-...`, `...RNN-2-...`, ..., `...RNN-8-...`)

### Choosing epoch
Next step is choosing which epoch to sample weigths for the neural network from, given the use of `debug_model` you will have validations for each epoch and train/validation for every epoch.

The rule used when picking the best epoch is as follows:
- The epoch must be after the first 20 epochs (the network will often not have picked up features of high describtive value before after a certain amount of training)
- Pick highest valued validation.
- For ensemble, epochs picked must be at least 10 epochs apart (often you will find well performing epochs following each other, as they have very similar weights. Picking epochs apart will increase variance).

Using these rules you should end up with:

Given no ensemble:

Epoch for 1st split: 22

Epoch for 2nd split: 39

Epoch for 3rd split: 20

Epoch for 4th split: 137

Epoch for 5th split: 26

Epoch for 6th split: 35

Epoch for 7th split: 81

Epoch for 8th split: 140

Given ensemble:

Epochs for 1st split: 22, 39, 64

Epochs for 2nd split: 39, 64, 129

Epochs for 3rd split: 20, 30, 40

Epochs for 4th split: 137, 93, 121

Epochs for 5th split: 26, 41, 53

Epochs for 6th split: 35, 46, 56

Epochs for 7th split: 81, 131, 146

Epochs for 8th split: 140, 91, 109

### Making predictions

Move the epochs from:

`$PATH_TO_DIR/EEG_Dauwelslab/metadata/dump_RNN-...`

To:

`$PATH_TO_DIR/EEG_Dauwelslab/metadata/FOR_ENSEMBLE/RNN/<split>`

Example:

>> cd $PATH_TO_DIR/EEG_Dauwelslab/metadata/

>> mv metadata/dump_RNN-1-20160425-055244-149.pkl FOR_ENSEMBLE/RNN/1

#### Get test set
Now given the test dataset (you can enquire Justin Dauwels at jdauwels at ntu.edu.sg for access).

Place the test data in the following folder `$PATH_TO_DIR/EEG_Dauwelslab/data/csv/test/`

Such that.

>> cd $PATH_TO_DIR/EEG_Dauwelslab/data/csv/train/

>> ls

```
Btst1.csv  Btst3.csv  Btst5.csv  Btst7.csv  Stst1.csv
Stst3.csv  Stst5.csv  Stst7.csv  Btst2.csv  Btst4.csv
Btst6.csv  Btst8.csv  Stst2.csv  Stst4.csv  Stst6.csv
Stst8.csv
```

Make the data into a numpy format

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

>> python create_data test

```
you csv path is: ./data/csv/test/*
Converting data ...
Opening: ./data/csv/test/Btst7.csv
Saved to ./data/numpy/test/Btst7.npy.gz
Opening: ./data/csv/test/Stst2.csv
Saved to ./data/numpy/test/Stst2.npy.gz
...
Opening: ./data/csv/test/Stst4.csv
Saved to ./data/numpy/test/Stst4.npy.gz
```

#### Computing predictions

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

>> bash batch_predictions.sh

(predictions will be saved in $PATH_TO_DIR/EEG_Dauwelslab/predictions/RNN/<split>)

#### Evaluating predictions

>> cd $PATH_TO_DIR/EEG_Dauwelslab/

Please note that `eval_predictions.py` takes `p` being probability for spike/background as first system argument.

`eval_predictions.py <probability> <model>`

For `p=0.5`

>> python eval_predictions.py 0.5 RNN

(`eval_predictions.py` will average probabilities over all predictions in `$PATH_TO_DIR/EEG_Dauwelslab/predictions/RNN/<split>`)

For no ensemble you should get the following results from `eval_predictions.py 0.5 RNN`:
```
-- SPLIT 1 of 8 --
Using gpu device 1: GeForce GTX TITAN X (CNMeM is disabled, cuDNN Version is too old. Update to v5, was 3007.)
loadTest started!
TP 3198, FN 345
FP 1095, TN 35298
TPR: 0.90262
SPC: 0.96991
PPV: 0.74493
AUC (test) is: 0.98532
-- SPLIT 2 of 8 --
loadTest started!
TP 1971, FN 39
FP 0, TN 64446
TPR: 0.98060
SPC: 1.00000
PPV: 1.00000
AUC (test) is: 0.99959
-- SPLIT 3 of 8 --
loadTest started!
TP 1426, FN 336
FP 421, TN 46761
TPR: 0.80931
SPC: 0.99108
PPV: 0.77206
AUC (test) is: 0.99484
-- SPLIT 4 of 8 --
loadTest started!
TP 2735, FN 3
FP 0, TN 68887
TPR: 0.99890
SPC: 1.00000
PPV: 1.00000
AUC (test) is: 1.00000
-- SPLIT 5 of 8 --
loadTest started!
TP 807, FN 35
FP 0, TN 53119
TPR: 0.95843
SPC: 1.00000
PPV: 1.00000
AUC (test) is: 0.99974
-- SPLIT 6 of 8 --
loadTest started!
TP 1746, FN 0
FP 1780, TN 57430
TPR: 1.00000
SPC: 0.96994
PPV: 0.49518
AUC (test) is: 0.99991
-- SPLIT 7 of 8 --
loadTest started!
TP 2595, FN 215
FP 865, TN 92763
TPR: 0.92349
SPC: 0.99076
PPV: 0.75000
AUC (test) is: 0.98956
-- SPLIT 8 of 8 --
loadTest started!
TP 2810, FN 55
FP 6236, TN 77167
TPR: 0.98080
SPC: 0.92523
PPV: 0.31063
AUC (test) is: 0.99241

FINAL RESULTS
[[  17288.    1028.]
 [  10397.  495871.]]
AUC = 0.99517
TPR = 0.94427
SPC = 0.98086
PPV = 0.75910
```
For no ensemble you should get the following results from `eval_predictions.py 0.5 RNN`:
```
-- SPLIT 1 of 8 --
Using gpu device 1: GeForce GTX TITAN X (CNMeM is disabled, cuDNN Version is too old. Update to v5, was 3007.)
loadTest started!
TP 3464, FN 79
FP 1095, TN 35298
TPR: 0.97770
SPC: 0.96991
PPV: 0.75982
AUC (test) is: 0.98117
-- SPLIT 2 of 8 --
loadTest started!
TP 2005, FN 5
FP 0, TN 64446
TPR: 0.99751
SPC: 1.00000
PPV: 1.00000
AUC (test) is: 0.99992
-- SPLIT 3 of 8 --
loadTest started!
TP 1758, FN 4
FP 823, TN 46359
TPR: 0.99773
SPC: 0.98256
PPV: 0.68113
AUC (test) is: 0.99966
-- SPLIT 4 of 8 --
loadTest started!
TP 2737, FN 1
FP 0, TN 68887
TPR: 0.99963
SPC: 1.00000
PPV: 1.00000
AUC (test) is: 1.00000
-- SPLIT 5 of 8 --
loadTest started!
TP 808, FN 34
FP 0, TN 53119
TPR: 0.95962
SPC: 1.00000
PPV: 1.00000
AUC (test) is: 1.00000
-- SPLIT 6 of 8 --
loadTest started!
TP 1746, FN 0
FP 54, TN 59156
TPR: 1.00000
SPC: 0.99909
PPV: 0.97000
AUC (test) is: 1.00000
-- SPLIT 7 of 8 --
loadTest started!
TP 2595, FN 215
FP 433, TN 93195
TPR: 0.92349
SPC: 0.99538
PPV: 0.85700
AUC (test) is: 0.99348
-- SPLIT 8 of 8 --
loadTest started!
TP 2810, FN 55
FP 3905, TN 79498
TPR: 0.98080
SPC: 0.95318
PPV: 0.41847
AUC (test) is: 0.99472

FINAL RESULTS
[[  1.79230000e+04   3.93000000e+02]
 [  6.31000000e+03   4.99958000e+05]]
AUC = 0.99612
TPR = 0.97956
SPC = 0.98751
PPV = 0.83580
```

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
