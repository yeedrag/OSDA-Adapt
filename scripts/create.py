import os
import argparse

def create_folder_and_files(name):
    # Create the folder
    os.makedirs(name, exist_ok=True)

    # Content of the train bash script
    train_script_content = f"""#!/bin/bash

# ---- SLURM SETTINGS ---- #

# -- Job Specific -- #
#SBATCH --job-name="gta2cs"   # What is your job called?
#SBATCH --output=%j-output.txt  # Output file - Use %j to inject job id, like %j-output.txt
#SBATCH --error=%j-error.txt    # Error file - Use %j to inject job id, like %j-error.txt

#SBATCH --partition=GPU        # Which group of nodes do you want to use? Use "GPU" for graphics card support
#SBATCh --time=7-00:00:00       # What is the max time you expect the job to finish by? DD-HH:MM:SS

# -- Resource Requirements -- #
#SBATCH --mem=128G                # How much memory do you need?
#SBATCH --ntasks-per-node=32     # How many CPU cores do you want to use per node (max 64)?
#SBATCH --nodes=1               # How many nodes do you need to use at once?
#SBATCH --gpus=1               # Do you require a graphics card? How many (up to 3 per node)? Remove the first "#" to activate.

# -- Email Support -- #
#SBATCH --mail-type=END         # What notifications should be emailed about? (Options: NONE, ALL, BEGIN, END, FAIL, QUEUE)

# ---- YOUR SCRIPT ---- #
cd ../..
source $(conda info --base)/etc/profile.d/conda.sh
module load python-libs
conda activate daformer_2 # activates the virtual environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python run_experiments.py --config configs/daformer/{name}.py
"""

    # Content of the test bash script
    test_script_content = f"""#!/bin/bash

# ---- SLURM SETTINGS ---- #

# -- Job Specific -- #
#SBATCH --job-name="gta2cs"   # What is your job called?
#SBATCH --output=%j-output.txt  # Output file - Use %j to inject job id, like %j-output.txt
#SBATCH --error=%j-error.txt    # Error file - Use %j to inject job id, like %j-error.txt

#SBATCH --partition=GPU        # Which group of nodes do you want to use? Use "GPU" for graphics card support
#SBATCh --time=7-00:00:00       # What is the max time you expect the job to finish by? DD-HH:MM:SS

# -- Resource Requirements -- #
#SBATCH --mem=128G                # How much memory do you need?
#SBATCH --ntasks-per-node=32     # How many CPU cores do you want to use per node (max 64)?
#SBATCH --nodes=1               # How many nodes do you need to use at once?
#SBATCH --gpus=1               # Do you require a graphics card? How many (up to 3 per node)? Remove the first "#" to activate.

# -- Email Support -- #
#SBATCH --mail-type=END         # What notifications should be emailed about? (Options: NONE, ALL, BEGIN, END, FAIL, QUEUE)

# ---- YOUR SCRIPT ---- #
cd ../..
source $(conda info --base)/etc/profile.d/conda.sh
module load python-libs
conda activate daformer_2 # activates the virtual environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Add the name of the folder in
sh test.sh work_dirs/local-basic/
"""

    # Create the train script file
    train_script_path = os.path.join(name, f"{name}_train.sh")
    with open(train_script_path, 'w') as train_script_file:
        train_script_file.write(train_script_content)

    # Create the test script file
    test_script_path = os.path.join(name, f"{name}_test.sh")
    with open(test_script_path, 'w') as test_script_file:
        test_script_file.write(test_script_content)

    print(f"Folder '{name}' and bash scripts created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate folder and bash scripts.")
    parser.add_argument('--name', type=str, required=True, help='Name for the folder and scripts.')
    
    args = parser.parse_args()
    create_folder_and_files(args.name)
