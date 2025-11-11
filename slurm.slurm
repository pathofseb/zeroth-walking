#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="zbot-train-walk1"
#SBATCH --time=18:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --constraint="(a100|h100|h200)&(gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/walk1_output.txt
#SBATCH --error=slurm_outputs/walk1_error.txt
#SBATCH --mail-user=sebasrb@ntnu.no
#SBATCH --mail-type=ALL

# Get the project root directory (parent of slurms folder)
# SLURM_SUBMIT_DIR is the directory where sbatch was called from

echo "Running from directory: $WORKDIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"

# Create output directory if it doesn't exist
mkdir -p slurm_outputs

# Path to your conda env (change if needed)
ENV_PATH="/cluster/work/$USER/ksim-zeroth/env/"

module purge
module load Anaconda3/2024.02-1

# Create conda environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
  conda create -y --prefix ${ENV_PATH} python=3.11
fi

# Use Python directly from the environment (don't rely on activation)
source activate ${ENV_PATH}

# Install uv if not already installed
pip install -r requirements.txt

# Install dependencies using uv

echo "Installed dependencies:"
pip freeze

# Verify GPU backend
echo "Checking JAX backend:"
python -c "import jax; print(jax.default_backend())"  # Should print "gpu"

# Set MuJoCo to use headless rendering backend (osmesa)
export MUJOCO_GL=osmesa
echo "Set MUJOCO_GL=osmesa for headless rendering"

# Run training
python train.py



# Example: to load pretrained checkpoint and run in view mode, uncomment:
# python train.py load_from_ckpt_path=zbot_walking_task/run_9/checkpoints/ckpt.bin # run_mode=view

conda deactivate

echo "Job finished"
