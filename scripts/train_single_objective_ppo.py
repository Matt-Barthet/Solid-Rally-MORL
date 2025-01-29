import argparse
import numpy as np
from solid_rally import SolidRallyEnvironment
from stable_baselines3 import PPO
import platform


class SolidRallyRallySingleObjective(SolidRallyEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix="", cluster=0):
        super().__init__(id_number=id_number, graphics=graphics, path=path, logging=logging, log_prefix=log_prefix, cluster=cluster, weight=weight)
        self.weight = weight
        self.arousals = []

    def reward_behavior(self):
        # Using reward from the environment as behavior reward (i.e. optimize env score)
        return self.current_reward


    def reward_affect(self):
        mean_arousal = np.sum(self.arousals) / len(self.arousals) # Arousal range [0, 1]
        target_arousal = 1
        return 1 - (target_arousal - mean_arousal) # reward similarity to target arousal


    def step(self, action):
        state, env_score_change, arousal, d, info = super().step(action)
        self.arousals.append(arousal)

        # Only assign rewards if the agent behaves correctly (passes through a checkpoint).
        if env_score_change > 0:
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)
            self.arousals.clear()
        else:
            final_reward = 0

        self.reset_condition()
        return state, final_reward, d, info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    args = parser.parse_args()

    run = args.run
    weight = args.weight
    cluster = args.cluster

    system = platform.system()
    if system == "Darwin":
        game_path = "../solid_rally/macos/racing.app"
    elif system == "Linux":
        game_path = "../solid_rally/linux/racing.x86_64"
    else:
        game_path = "../solid_rally/windows/racing.exe"


    env = SolidRallyRallySingleObjective(id_number=run, weight=weight, graphics=False, logging=True,
                                         path=game_path, log_prefix="ppo/", cluster=cluster)

    cluster_names = ["All Players", "Intermediates", "Beginners", "Excited_Experts", "Unexcited_Experts"]
    model = PPO("MlpPolicy", env=env, tensorboard_log=f"./Tensorboard/ppo/Affectively_Log_{cluster_names[cluster]}_{weight}λ_Run{run}", device='cpu')
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(f"./Agents/ppo/ppo_solid_rally_cluster{cluster}_{weight}λ_{run}")
