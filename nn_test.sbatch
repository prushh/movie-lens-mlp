#!/bin/bash
#SBATCH --job-name=test_nn
#SBATCH --error=err_nn_test.txt
#SBATCH --output=test_nn.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=giosue.cotugno@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source venv/bin/activate

pip3 install --no-cache-dir -r requirements.txt

python3 -u main.py mlp -s 0