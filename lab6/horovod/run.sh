#!/bin/bash
#SBATCH -p pp22
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o horovod.out.%j 

source /home/pp22/share/lab6/env/bin/activate
horovodrun -np 8 -H $SLURM_JOB_NODELIST:8 python tf2_mnist_horovod.py