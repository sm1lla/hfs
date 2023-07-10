#!/bin/bash 
#SBATCH --job-name="hfs-exp" 
#SBATCH --mem=100G  
#SBATCH --exclude=ic922-[01-02] 
#SBATCH --account=renard 
#SBATCH --partition=magic 
#SBATCH --time=2-00:00:00 
#SBATCH --chdir=/hpi/fs00/home/kathrin.thenhausen/hfs-1/hfs/
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=kathrin.thenhausen@student.hpi.de 
#SBATCH --verbose 
#SBATCH --output=/hpi/fs00/home/kathrin.thenhausen/hfs-1/hfs/results/result2/slurm_%j_%N_%x_%A_%a.out 
#SBATCH --error=/hpi/fs00/home/kathrin.thenhausen/hfs-1/hfs/results/result2/slurm_%j_%N_%x_%A_%a.err 
conda init bash
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate env
export PYTHONPATH=~/hfs-1/

python experiments/hip.py