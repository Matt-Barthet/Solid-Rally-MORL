# Solid Rally MORL
This repository contains the code to run the Solid Rally game environment for reinforcement learning tasks. The builds are included in the `builds` directory for Windows, Macintosh and Linux, and should be automatically selected based on the system when running the training script. 

## Repository Overview
The code can be found in the `scripts` directory. For implementing a new controller, you can use code similar to the `train_single_objective_ppo.py` script, which defines a solid rally environment and the agent architecture being trained. Importantly, this script defines the `target_arousal`, which we test 1 (maximize arousal) and 0 (minimize arousal), and the behavior reward which is simply maximizing in-game score.  The `graphics` variable can be set to false to run in headless mode for better efficiency. The `batch.py` script can be used to deploy several experiments of an agent's training script using the given arguments in parallel for better utilization of compuational resources.

## Setup 
The `requirements.txt` contains all packages and versions needed to run this code. The builds are included in the repository so there is no longer need to download them separately.

Create conda environment.
```bash
conda create -n affect-envs python==3.9
```
Activate the environment
```bash
conda activate affect-envs
```
Downgrade `pip` and `setuptools`:
```bash
python -m pip install setuptools==69.5.1 pip==24.0
```
Install dependencies
```bash
pip install -r requirements.txt
```