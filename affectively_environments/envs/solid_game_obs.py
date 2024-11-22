import numpy as np

from .solid import SolidEnvironment


class SolidEnvironmentGameObs(SolidEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix="", discretize=False):
        self.discretize = discretize
        self.estimated_position = [0, 0, 0]
        super().__init__(id_number=id_number, graphics=graphics,
                         obs={"low": -np.inf, "high": np.inf, "shape": (54,), "type": np.float32},
                         path=path, weight=weight, logging=logging, frame_buffer=False, log_prefix=log_prefix)

    def construct_state(self, state):
        if self.frameBuffer:
            game_obs = state[1]
        else:
            game_obs = state[0]
        if self.discretize:
            game_obs = self.discretize_observations(game_obs)
        self.game_obs = self.tuple_to_vector(game_obs)
        return self.game_obs

    def discretize_observations(self, game_obs):

        position_delta = game_obs[0:3]
        position = np.array(self.estimated_position) + position_delta
        self.estimated_position = position

        position_discrete = np.round(self.estimated_position / 40)
        position_discrete[0] = 0 if position_discrete[0] == -0 else position_discrete[0]
        position_discrete[1] = 0 if position_discrete[1] == -0 else position_discrete[1]
        position_discrete[2] = 0 if position_discrete[2] == -0 else position_discrete[2]

        velocity = game_obs[3:6]
        velocity_discrete = np.round(np.linalg.norm(velocity) / 30)

        score = game_obs[47]
        if score < 8:
            score_bin = 0
        elif score < 16:
            score_bin = 1
        else:
            score_bin = 2

        is_off_road = game_obs[48]
        is_in_loop_zone = game_obs[49]

        discrete_obs = (
            list(position_discrete) +
            [
                velocity_discrete,
                score_bin,
                is_off_road,
                is_in_loop_zone,
            ]
        )

        return discrete_obs
