import numpy as np
from solid_rally import SolidRallyEnvironment
from stable_baselines3 import PPO


class SolidRallyRallySingleObjective(SolidRallyEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix="", cluster=0):
        super().__init__(id_number=id_number, graphics=graphics, path=path, logging=logging, log_prefix=log_prefix, cluster=cluster, weight=weight)
        self.weight = weight
        self.arousals = []


    def reward_behavior(self):
        return self.current_reward
        # if (self.episode_length / 15)  % 1 == 0:
        #     print(f"REWARD - Ep length {self.episode_length} - Target Score {self.model.target_behavior[int(self.episode_length/15)-1]}")
        # Imitation reward for behavior (R_B)
        # return 0

    def reward_affect(self):
        # Imitation reward for affect (R_E)
        # if (self.episode_length / 15)  % 1 == 0:
        #     print(f"REWARD - Ep length {self.episode_length} - Target Arousal {self.model.target_arousal[int(self.episode_length/15)-1]}")
        return np.sum(self.arousals) / len(self.arousals)


    def step(self, action):
        state, env_score_change, arousal, d, info = super().step(action)
        self.arousals.append(arousal)

        # Only assign rewards if the agent behaves correctly.
        if env_score_change > 0:
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)
            self.arousals.clear()
        else:
            final_reward = 0

        self.reset_condition()
        return state, final_reward, d, info


if __name__ == "__main__":
    run = 1
    weight = 1
    cluster = 4
    env = SolidRallyRallySingleObjective(id_number=run, weight=weight, graphics=False, logging=True,
                                         path="solid_rally/Racing_Linux.x86", log_prefix="ppo/", cluster=cluster)
    env.targetSignal = np.ones

    cluster_names = ["All Players", "Intermediates", "Beginners", "Excited_Experts", "Unexcited_Experts"]
    model = PPO("MlpPolicy", env=env, tensorboard_log=f"./Tensorboard/ppo/Affectively_Log_{cluster_names[cluster]}_{weight}λ_Run{run}", device='cpu')
    model.learn(total_timesteps=10000000, progress_bar=True)
    model.save(f"./Agents/ppo/ppo_solid_rally_cluster{cluster}_{weight}λ_{run}")
