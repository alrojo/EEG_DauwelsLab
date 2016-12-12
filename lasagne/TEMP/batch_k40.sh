#!/bin/sh
#PBS -N test1AndMLP
#PBS -q k40_interactive
#PBS -l walltime=36:00:00
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -M s145706@student.dtu.dk
#PBS -m abe
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi
module load cuda
python train.py ConvMulti 3 200
python train.py ConvMulti 4 200
python train.py ConvMulti 5 200
