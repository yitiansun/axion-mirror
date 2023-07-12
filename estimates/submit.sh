#!/bin/bash

#SBATCH --job-name=hirax_f
#SBATCH --partition=iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=gpu:1

source /n/home07/yitians/setup_jaxgpu.sh

srun python 3_snr.py --population "fullinfo" --config "HIRAX-1024-nnu30-nra3-ndec3" --use_tqdm