#!/bin/bash
#SBATCH -D /users/username/some/folder/or/other    # Working directory
#SBATCH --job-name train_transformer               # Job name
#SBATCH --partition=gengpu                         # Select the correct partition.
#SBATCH --nodes=1                                  # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=8                        # Use 8 cores, most of the procesing happens on the GPU
#SBATCH --mem=24MB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=24:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:1                               # Use one gpu.
#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o results/%x_%j.o                         # [%j is replaced with the jobid, %x with the job name]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpagi.kironde@city.ac.uk

#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

#Remove any unwanted modules
module purge

#Modules required
#This is an example you need to select the modules your code needs.
module load python/3.7.12
module load libs/nvidia-cuda/11.2.0/bin

#Run your script.
python3 train_transformer.py