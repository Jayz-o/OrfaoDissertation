#!/bin/bash
#PBS -N casia_gan
#PBS -l select=1:ncpus=10:mpiprocs=1:walltime=12:00:00
#PBS -P CSCI1528
#PBS -q gpu_4
#PBS -o /mnt/lustre/users/jorfao/runs/casia_gan.out
#PBS -e /mnt/lustre/users/jorfao/runs/casia_gan.err
#PBS -m abe
#PBS -M 216082337@student.uj.ac.za

module purge
module add chpc/python/anaconda/3-2019.10
module load chpc/cuda/11.2/SXM2/11.2
#conda activate /home/USERNAME/myenv
#module load chpc/cuda/11.2/PCIe/11.2
#source /mnt/lustre/users/jorfao/ml/bin/activate
eval "$(conda shell.bash hook)"

conda activate /home/jorfao/ML

export PYTHONPATH="${PYTHONPATH}:/mnt/lustre/users/jorfao/Documents/Orfao_Masters"

python /mnt/lustre/users/jorfao/Documents/Orfao_Masters/GAN/GANTraining/train_casia_gan.py

conda deactivate /home/jorfao/ML