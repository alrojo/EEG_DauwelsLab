# EEG_DauwelsLab
This github project is dedicated to spike detection in EEG data from epileptical patients.
The code is Produced by A. Rosenberg Johansen[^f1].
The project is supervisioned by PhD. Justin Dauwels[^f2] lab.
The data is supplied by M.D. Sydney Cash[^f2] and M.D. M. Brandon Westover[^f3].

The code as been used for ICASSP 2016 submission, using a dataset of five patients, and is now being used for a journal paper with a dataset of 100 patients.

**Data**
The data set consists of 30 minutes EEG recordings sampled from over 100 patients.
The dataset has been preprocessed into 64 length sliding windows, and labelled by M.D. to either be containing "epileptical spike" or "no epileptical spike".
The data is split into eight equal size training, validation and test splits with no overlapping patients.

**Code**
The code can be run through *train.py*, dumps will be saved in *metadata/* and can be evaluated using *debug_metadata.py*. Configurations for the different models used can be found in *configurations/*

[^f1] Technical University of Denmark, DTU Compute, Lyngby, Denmark
[^f2] Nanyang Technological University, School of Electrical and Electronic Engineering, Singapore
[^f3] Massechusetts General Hospital, Neurology Department, and Harvard Medical School, USA
