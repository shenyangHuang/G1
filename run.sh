#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=test.txt 
#SBATCH --error=test_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:a100l:2                   # Ask for 1 titan xp gpu:rtx8000:1 
#SBATCH --mem=128G #64G                             # Ask for 32 GB of RAM
#SBATCH --constraint="dgx&ampere"
#SBATCH --time=24:00:00    #48:00:00                   # The job will run for 1 day

module load python/3.10
source $SCRATCH/G1/.venv/bin/activate
pwd

source run_3B.sh