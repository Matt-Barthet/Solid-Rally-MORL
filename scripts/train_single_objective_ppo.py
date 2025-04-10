import argparse
import numpy as np
from solid_rally import SolidRallyEnvironment
from stable_baselines3 import PPO
import platform


class SolidRallyRallySingleObjective(SolidRallyEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix="", cluster=0, target_arousal=1, periodic_ra=False):
        super().__init__(id_number=id_number, graphics=graphics, path=path, logging=logging, log_prefix=log_prefix, cluster=cluster, weight=weight, target_arousal=target_arousal)
        self.weight = weight
        self.score_change = False
        self.periodic_ra = True if periodic_ra == 1 else False

    def reward_behavior(self):
        # Using reward from the environment as behavior reward (i.e. optimize env score)
        r_b = 1 if self.score_change else 0
        self.score_change = False
        self.best_rb = np.max([r_b, self.best_rb])
        self.cumulative_rb += r_b
        return r_b


    def reward_affect(self):
        # Reward similarity of mean arousal this period to target arousal (0 = minimize, 1 = maximize)
        mean_arousal = np.mean(self.period_arousal_trace) if len(self.period_arousal_trace) > 0 else 0 # Arousal range [0, 1]
        r_a = 1 - np.abs(self.target_arousal - mean_arousal)
        self.best_ra = np.max([self.best_ra, r_a])
        self.cumulative_ra += r_a
        self.period_arousal_trace.clear()
        return r_a


    def step(self, action):
        state, env_score, arousal, d, info = super().step(action)
        change_in_score = (self.current_score - self.previous_score)
        if not self.score_change:
            self.score_change = change_in_score > 0

        if self.periodic_ra and len(self.period_arousal_trace) != 0:
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)
        elif not self.periodic_ra and change_in_score != 0:
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)
        else:
            final_reward = 0

        self.cumulative_rl += final_reward
        self.best_rl = np.max([self.best_rl, final_reward])
        self.reset_condition()
        return state, final_reward, d, info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--periodic_ra", type=float, required=True, help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    args = parser.parse_args()

    run = args.run
    weight = args.weight
    cluster = args.cluster
    target_arousal = args.target_arousal
    periodic_ra = args.periodic_ra

    system = platform.system()
    if system == "Darwin":
        game_path = "../solid_rally/macos/racing.app"
    elif system == "Linux":
        game_path = "../solid_rally/linux/racing.x86_64"
    else:
        game_path = "../solid_rally/windows/racing.exe"

    env = SolidRallyRallySingleObjective(id_number=run, weight=weight, graphics=False, logging=True, path=game_path, log_prefix="ppo/", cluster=cluster, target_arousal=target_arousal, periodic_ra=periodic_ra)
    cluster_names = ["All Players", "Intermediates", "Beginners", "Excited_Experts", "Unexcited_Experts"]
    model = PPO("MlpPolicy", env=env, tensorboard_log=f"../results/tensorboard/ppo/Affectively_Log_{cluster_names[cluster]}_{weight}λ_target{target_arousal}_Run{run}", device='cpu')
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(f"../results/agents/ppo/ppo_solid_rally_cluster{cluster}_{weight}λ_target{target_arousal}_run{run}.zip")
