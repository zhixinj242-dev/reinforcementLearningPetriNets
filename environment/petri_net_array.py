import gym.spaces

from .petri_net import JunctionPetriNetEnv, LanePetriNetTuple


class PetriNetEnvArray(JunctionPetriNetEnv):

    def __init__(self, net=None, max_num_tokens: int = 1, max_num_cars_per_lane: int = 50,
                 lanes: [LanePetriNetTuple] = None, success_action_reward: float = 5.0,
                 success_car_drive_reward: float = 5.0, max_steps: int = 100) -> None:
        super().__init__(net=net, max_num_tokens=max_num_tokens, max_num_cars_per_lane=max_num_cars_per_lane,
                         lanes=lanes, success_action_reward=success_action_reward,
                         success_car_drive_reward=success_car_drive_reward, max_steps=max_steps)

        vector_dimensionality = []
        for dict in self.observation_space:
            for key in self.observation_space[dict]:
                vector_dimensionality.append(self.observation_space[dict][key].n)

        self.observation_space = gym.spaces.MultiDiscrete(vector_dimensionality)

    def _get_obs(self):
        obs = []
        for place in self.net.place():
            obs.append(len(place.tokens))
        for lane in self.lanes:
            obs.append(len(lane.vehicles))

        return obs
