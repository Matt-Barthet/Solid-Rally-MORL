import numpy as np
import sys
from affectively_environments.envs.pirates import PiratesEnvironment
from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from examples.Agents.DQN.Rainbow_DQN import RainbowAgent, train


def main(run, weight, env_type):
    np.set_printoptions(suppress=True, precision=6)

    preference_task = True
    classification_task = False

    if env_type.lower() == "pirates":
        env = PiratesEnvironment(id_number=run, weight=weight, graphics=True, logging=True,
                                 path="../Builds/MS_Pirates/platform.exe", log_prefix="DQN/")
    elif env_type.lower() == "solid":
        env = SolidEnvironmentGameObs(id_number=run, weight=weight, graphics=True, logging=True,
                                      path="../Builds/MS_Solid/platform.exe", log_prefix="DQN/")
    else:
        raise ValueError("Invalid environment type. Choose 'pirates' or 'solid'.")

    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    agent = RainbowAgent(env.observation_space.shape[0], env.action_space.nvec.tolist())
    num_episodes = 16638
    batch_size = 64
    update_target_every = 600
    train(agent, env, num_episodes, batch_size, update_target_every, name=f"run{run}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <run-number> <weight> <environment>")
        sys.exit(1)

    try:
        run = int(sys.argv[1])
        weight = float(sys.argv[2])
        env_type = sys.argv[3]
    except ValueError as e:
        print(f"Error in argument parsing: {e}")
        sys.exit(1)

    main(run, weight, env_type)

