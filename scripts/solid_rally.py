import uuid
from abc import ABC

import gym
import numpy as np
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from tensorboardX import SummaryWriter

from custom_logging import backup
from sidechannels import AffectivelySideChannel
from surrogatemodel import KNNSurrogateModel

class TensorBoardCallback:
    def __init__(self, log_dir, environment):
        self.log_dir = log_dir
        self.environment = environment
        backup(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.episode = 0
        self.best_arousal_reward = 0

    def on_episode_end(self):
        mean_arousal = np.mean(self.environment.arousal_trace)
        mean_arousal = mean_arousal if not np.isnan(mean_arousal) else 0
        self.best_arousal_reward = np.max([self.best_arousal_reward, mean_arousal])
        self.writer.add_scalar('reward/best_r_a', self.best_arousal_reward, self.episode)
        self.writer.add_scalar('reward/best_score', self.environment.best_score, self.episode)
        self.writer.add_scalar('reward/best_r_b', self.environment.best_cumulative_reward, self.episode)
        self.writer.add_scalar('reward/episode_mean_arousal', np.mean(self.environment.arousal_trace), self.episode)
        self.episode += 1
        self.writer.flush()


class SolidRallyEnvironment(gym.Env, ABC):

    def __init__(self, id_number, graphics, path, capture_fps=5, time_scale=1, args=None,
                 logging=True, log_prefix="", cluster=0, weight=0):

        super(SolidRallyEnvironment, self).__init__()
        if args is None:
            args = []
        socket_id = uuid.uuid4()

        args += [f"-socketID", str(socket_id), "-frameBuffer", f"{False}"]

        self.game_obs = []

        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=capture_fps, time_scale=time_scale)
        self.customSideChannel = AffectivelySideChannel(socket_id)
        self.env = self.load_environment(path, id_number, graphics, args)

        self.env = UnityToGymWrapper(self.env, allow_multiple_obs=True)

        self.action_space, self.action_size = self.env.action_space, self.env.action_space.shape
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)

        self.model = KNNSurrogateModel(5, "Solid", cluster=cluster)
        self.scaler = self.model.scaler

        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.arousal_trace = []

        self.current_score, self.current_reward, self.cumulative_reward = 0, 0, 0
        self.best_reward, self.best_score, self.best_cumulative_reward = 0, 0, 0
        self.previous_score, self.episode_length, self.arousal_count = 0, 0, 0

        cluster_names = ["All Players", "Intermediates", "Beginners", "Excited_Experts", "Unexcited_Experts"]
        self.callback = TensorBoardCallback(f'./Tensorboard/{log_prefix}ppo-{cluster_names[cluster]}-{weight}λ-run{id_number}', self) if logging else None
        self.create_and_send_message("[Save States]:Seed")

    def construct_state(self, state):
        game_obs = state[0]
        self.game_obs = self.tuple_to_vector(game_obs)
        return self.game_obs

    def calculate_reward(self):
        self.best_score = np.max([self.current_score, self.best_score])
        self.current_reward = (self.current_score - self.previous_score)
        self.cumulative_reward += self.current_reward
        self.best_cumulative_reward = self.current_reward if self.current_reward > self.best_cumulative_reward else self.best_cumulative_reward

    def reset_condition(self):
        if self.episode_length > 600:
            self.episode_length = 0
            self.reset()

    def reset(self, **kwargs):
        if self.callback is not None and len(self.arousal_trace) > 0:
            self.callback.on_episode_end()

        self.episode_length = 0
        self.current_reward, self.current_score, self.cumulative_reward, self.previous_score = 0, 0, 0, 0
        self.previous_surrogate, self.current_surrogate = np.empty(0), np.empty(0)
        self.arousal_trace = []
        state = self.construct_state(self.env.reset())
        return state

    def generate_arousal(self):
        arousal = 0
        self.current_surrogate = np.array(self.customSideChannel.arousal_vector.copy(), dtype=np.float32)
        # print(self.current_surrogate)
        if self.current_surrogate.size != 0:

            # try:
            scaled_obs = np.array(self.scaler.transform(self.current_surrogate.reshape(1, -1))[0])

            # if np.abs(np.max(scaled_obs)) > 1:
            #     print(np.max(scaled_obs), np.argmax(scaled_obs))
            # except:
            #     scaled_obs = np.zeros((len(self.previous_surrogate)))
            #
            if self.previous_surrogate.size == 0:
                self.previous_surrogate = np.zeros(len(self.current_surrogate))

            previous_scaler = np.array(self.scaler.transform(self.previous_surrogate.reshape(1, -1))[0])
            unclipped_tensor = np.array(list(previous_scaler) + list(scaled_obs))

            tensor = torch.Tensor(np.clip(unclipped_tensor, 0, 1))
            self.previous_surrogate = previous_scaler

            arousal = self.model(tensor)[0]
            if not np.isnan(arousal):
                self.arousal_trace.append(arousal)
            self.previous_surrogate = self.current_surrogate.copy()
        return arousal

    def step(self, action):
        self.episode_length += 1
        self.arousal_count += 1

        self.previous_score = self.current_score
        send_arousal = 0
        if self.arousal_count % 14 == 0:  # Request the surrogate vector 2 ticks in advanced due to delay
            # print("Ask for arousal")
            send_arousal = 1

        arousal = 0
        if self.arousal_count % 15 == 0:  # Read the surrogate vector on the 15th tick
            # print("Generating Arousal")
            arousal = self.generate_arousal()
            self.arousal_count = 0
            # print(arousal)

        transformed_action = [action[0] - 1, action[1] - 1, send_arousal, 0, 0]
        state, env_score, done, info = self.env.step(transformed_action)
        state = self.construct_state(state)
        self.current_score = env_score
        self.calculate_reward()

        return state, env_score, arousal, done, info

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def load_environment(self, path, identifier, graphics, args):
        try:
            return UnityEnvironment(f"{path}",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics,
                                   additional_args=args)
        except:
            print("Checking next ID!")
            return self.load_environment(path, identifier + 1, graphics, args)


    @staticmethod
    def tuple_to_vector(s):
        obs = []
        for i in range(len(s)):
            obs.append(s[i])
        return obs