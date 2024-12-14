#!/bin/bash

#SBATCH --job-name=EXP3fullprogress

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -w chimera12
#SBATCH -N 1
#SBATCH --account=haehn
#SBATCH --qos=haehn_unlim
#SBATCH --mem=128gb
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --time=30-00:00:00
#SBATCH --partition=DGXA100
#SBATCH --export=HOME
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huuthanhvy.nguyen001@umb.edu
#SBATCH --gres=gpu:A100:1
#SBATCH --requeue

# Load necessary environment
. /etc/profile
. ~/.bashrc

# Log diagnostic information
echo "Using $SLURM_CPUS_ON_NODE CPUs"
echo "Job started at: $(date)"

# Activate Conda environment
ulimit -n 4096
conda activate sbatch2

# Experiment number passed as an argument
EXP_NUM=$1

# Run the Python script for the experiment
python EXP3fullprogressrun.py $EXP_NUM

# Log completion
echo "Job completed for experiment $EXP_NUM"
echo "Job ended at: $(date)"