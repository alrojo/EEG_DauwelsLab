# Recreating Results

## Installation and setup

The installation guide is aimed at running TensorFlow on a K80 GPU on the AWS EC2 p2.xlarge instances.
It will take about 2-3 days and cost about $75.

1. Go to the AWS EC2 console.
2. Start a ubuntu 16.04 instance on a p2.xlarge instance.
3. Connect to the instance
4. Follow [this guide](http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/) until Bazel install (we won't need Bazel).
5. Install [docker](https://docs.docker.com/engine/installation/) with user groups
6. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Run the following commands

>$git clone https://github.com/alrojo/EEG_Dauwelslab.git

>$nvidia-docker run -v ~/EEG_Dauwelslab:/mnt/ -it alrojo/tf-sklearn-gpu

>$cd mnt

>$nohup bash train_all.sh > out &

Now click ctrl+q ctrl+p, this should get you out of the docker instance without shutting it down.
Exit the AWS instance and let it run for about 48 hours, you can see with `nvidia-smi` if the job is still running.

Once the job has completed connect to your instance with port forwarding, such as.

>$ssh -i "key.pem" -L 8888:localhost:8888 ubuntu@instance.com

>$nvidia-docker run -v ~/EEG_Dauwelslab:/mnt/ -p 8888:8888 -it alrojo/tf-sklearn-gpu

>$./run_jupyter

Now go into your browser and type `localhost:8888`.
This should open an ipynb at the docker root.
Click into `mnt` and open the `predict.ipynb` and run all code blogs from the top down.
The ipynb will print out auc and accuracies for the splits and in total.
Further, In your `EEG\_Dauwelslab` folder you should have a list of targets and predictions in numpy format.

# About

## EEG\_DauwelsLab
This github project is dedicated to spike detection in EEG data from epileptical patients.

The code is Produced by A. Rosenberg Johansen<sup>1</sup>.

The project is supervisioned by PhD. Justin Dauwels<sup>2</sup> lab.

The data is supplied by MD. Sydney Cash<sup>3</sup> and MD. M. Brandon Westover<sup>3</sup>.

The code as been used for ICASSP 2016 submission, using a dataset of five patients, and is now being used for a journal paper with a dataset of 96 patients.

<sup>1</sup>: Technical University of Denmark, DTU Compute, Lyngby, Denmark

<sup>2</sup>: Nanyang Technological University, School of Electrical and Electronic Engineering, Singapore

<sup>3</sup>: Massechusetts General Hospital, Neurology Department, and Harvard Medical School, USA

## Data
The data set consists of 30 minutes EEG recordings sampled from over 100 patients.
The dataset has been preprocessed into 64 length sliding windows, and labelled by M.D. to either be containing "epileptical spike" or "no epileptical spike".
The data is split into eight equal size training, validation and test splits with no overlapping patients.
