#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -J "kill venv"
#SBATCH --mem=500g
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
module load python/3.10.13
module load cuda

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py
