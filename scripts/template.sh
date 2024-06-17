#!/bin/bash

# ---- SLURM SETTINGS ---- #

# -- Job Specific -- #
#SBATCH --job-name="Demo Job"   # What is your job called?
#SBATCH --output=%j-output.txt  # Output file - Use %j to inject job id, like %j-output.txt
#SBATCH --error=%j-error.txt    # Error file - Use %j to inject job id, like %j-error.txt

#SBATCH --partition=GPU        # Which group of nodes do you want to use? Use "GPU" for graphics card support
#SBATCh --time=7-00:00:00       # What is the max time you expect the job to finish by? DD-HH:MM:SS

# -- Resource Requirements -- #
#SBATCH --mem=5G                # How much memory do you need?
#SBATCH --ntasks-per-node=4     # How many CPU cores do you want to use per node (max 64)?
#SBATCH --nodes=1               # How many nodes do you need to use at once?
#SBATCH --gpus=1               # Do you require a graphics card? How many (up to 3 per node)? Remove the first "#" to activate.

# -- Email Support -- #
#SBATCH --mail-type=END         # What notifications should be emailed about? (Options: NONE, ALL, BEGIN, END, FAIL, QUEUE)

# ---- YOUR SCRIPT ---- #
cd ../DAFormer
module activate python-libs
conda activate daformer_2 # activates the virtual environment

python -m demo.image_demo demo/demo.png work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth
