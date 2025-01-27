import subprocess
import itertools
import time

# Define parameters
runs = [2]  # Example run IDs
weights = [0.5]  # Example weight values
clusters = [0, 1, 2, 3, 4]  # Cluster indices

# Path to the Python script
script_path = "scripts/train_single_objective_ppo.py"
# Conda environment to activate
conda_env = "affect-envs"

# Iterate over all combinations of parameters
for run, weight, cluster in itertools.product(runs, weights, clusters):
    # Command to activate Conda and execute the script
    command = (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {conda_env} && "
        f"python {script_path} --run={run} --weight={weight} --cluster={cluster}"
    )

    # Open a new terminal window for each command
    subprocess.Popen([
        "gnome-terminal",
        "--",
        "bash", "-c", f"{command};"
    ])

    time.sleep(1)