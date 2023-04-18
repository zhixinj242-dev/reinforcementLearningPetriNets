import gym.spaces
import numpy as np

from .petri_net import JunctionPetriNetEnv, LanePetriNetTuple


class PetriNetEnvArray(JunctionPetriNetEnv):

    def __init__(self, net=None, max_num_tokens: int = 1, max_num_cars_per_lane: int = 50,
                 lanes: [LanePetriNetTuple] = None, success_action_reward: float = 5.0,
                 success_car_drive_reward: float = 5.0, max_steps: int = 100) -> None:
        super().__init__(net=net, max_num_tokens=max_num_tokens, max_num_cars_per_lane=max_num_cars_per_lane,
                         lanes=lanes, success_action_reward=success_action_reward,
                         success_car_drive_reward=success_car_drive_reward, max_steps=max_steps)

        lower_bound = np.zeros(len(self.observation_space.keys()), dtype=np.float32)
        upper_bound = np.zeros(len(self.observation_space.keys()), dtype=np.float32)
        for idx, dict in enumerate(self.observation_space):
            upper_bound[idx] = self.observation_space[dict].high
        self.observation_space = gym.spaces.Box(low=lower_bound, high=upper_bound, shape=lower_bound.shape, dtype=np.float32)

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape)
        idx = 0
        for place in self.net.place():
            obs[idx] = len(place.tokens)
            idx = idx + 1
        for lane in self.lanes:
            obs[idx] = len(lane.vehicles)
            idx = idx + 1

        return obs
