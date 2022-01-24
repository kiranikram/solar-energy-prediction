#! /bin/bash

#SBATCH --job-name="Train"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpagi.kironde@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output ./logs/job%J.output
#SBATCH --error ./logs/jo%J.err
#SBATCH --gres=gpu:1
#SBATCH --partition=normal


module load cuda/11.2
module load python/3.7.9

cd ..
python3 predict_transformer.py --env=camber
