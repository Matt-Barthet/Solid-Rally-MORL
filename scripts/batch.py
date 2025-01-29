import os
import subprocess
import itertools
import time
import platform

# Define parameters
runs = [4]  # Example run IDs
weights = [0.5]  # Example weight values
clusters = [0, 1, 2, 3, 4]  # Cluster indices

cwd = os.getcwd()
script_path = "train_single_objective_ppo.py"
conda_env = "affect-envs"
system = platform.system()

# Iterate over all combinations of parameters
for run, weight, cluster in itertools.product(runs, weights, clusters):
    # Command to activate Conda and execute the script
    command = (
        f"cd {cwd} && conda activate {conda_env} && "
        f"python {script_path} --run={run} --weight={weight} --cluster={cluster}"
    )

    if system == "Linux":
        # Open a new terminal window for each command
        subprocess.Popen([
            "gnome-terminal",
            "--",
            "bash", "-c", f"{command};"
        ])
    elif system == "Windows":
        # Use start command for Windows
        subprocess.Popen(
            ["cmd.exe", "/K", f"conda activate {conda_env} && {command}"], shell=True
        )
    elif system == "Darwin":  # macOS
        # Use Terminal on macOS
        subprocess.Popen(
            ["osascript", "-e", f'tell app "Terminal" to do script "{command}"']
        )
    time.sleep(1)