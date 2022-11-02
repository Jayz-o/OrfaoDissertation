#!/bin/bash
#PBS -N gpu_test
#PBS -l select=1:ncpus=10:mpiprocs=1:walltime=00:01:00
#PBS -P CSCI1528
#PBS -q gpu_1
#PBS -o /mnt/lustre/users/jorfao/runs/gpu_test.out
#PBS -e /mnt/lustre/users/jorfao/runs/gpu_test.err
#PBS -M 216082337@student.uj.ac.za

module load chpc/python/anaconda/3-2021.05
module load chpc/cuda/11.2/PCIe/11.2


export PYTHONPATH="${PYTHONPATH}:/mnt/lustre/users/jorfao/Documents/Orfao_Masters"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/mnt/lustre/users/jorfao/orfao_masters/lib"

eval "$(conda shell.bash hook)"
conda activate /mnt/lustre/users/jorfao/orfao_masters

python /mnt/lustre/users/jorfao/Documents/Orfao_Masters/gpu_test.py

conda deactivate