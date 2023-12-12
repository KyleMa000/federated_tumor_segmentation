#!/bin/bash
# The interpreter used to execute the script

# directives that convey submission options:

#SBATCH --job-name=federatedlearning
#SBATCH --mail-user=mhaoyuan@umich.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=13:00:00
#SBATCH --account=eecs598s007f23_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python3.10-anaconda/2023.03

# The application(s) to execute along with its input arguments and options:
python train.py