# EEG_DauwelsLab
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

### Logistic Regression
As of 5'th December

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/LogisticRegression.py

Training AUC: 0.835

Validation AUC: 0.672

### MLP
As of 5'th December

Model config: https://github.com/alrojo/EEG_DauwelsLab/blob/master/configurations/MLP.py

Training AUC: 0.999

Validation AUC: 0.972

## Next to train
Convolutional models

LSTM models

Convolutional + LSTM models
