#!/bin/bash
#SBATCH --job-name=my_run_7
#SBATCH --time=72:00:00
#SBATCH --mem=20gb
#SBATCH --output=./logs/%jmy_run_7.log
#SBATCH --error=./logs/%jmy_run_7.err
#SBATCH -p gpu
##SBATCH --gres=gpu:1
pwd; hostname; date
python main.py