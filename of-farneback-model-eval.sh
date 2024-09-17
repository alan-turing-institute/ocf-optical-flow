#!/bin/bash
#SBATCH --qos turing
#SBATCH --account=vjgo8416-climate
#SBATCH --nodes 1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --time 45:00:00
#SBATCH --job-name of-fb-eval

# drop into baskerville
module purge; module load baskerville

module load bask-apps/live
module load Miniforge3/24.1.2-0

eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)" 
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

#activate conda env
CONDA_ENV_PATH="/bask/projects/v/vjgo8416-climate/of_cloud"
conda activate "${CONDA_ENV_PATH}"

python of-farneback-model-eval.py