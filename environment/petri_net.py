import gymnasium as gym
import numpy as np


class LanePetriNetTuple(object):
    def __init__(self, lane: str, place: str, probability: float = 1.0/8.0):
        self.name = lane
        self.place = place
        self.probability = probability
        self.num_vehicles = 0

    def reset(self):
        self.num_vehicles = 0


class JunctionPetriNetEnv(gym.Env):

    def __init__(self, net=None, max_num_tokens: int = 1, max_num_cars_per_lane: int = 50,
                 lanes: [LanePetriNetTuple] = None, success_action_reward: float = 5.0,
                 success_car_drive_reward: float = 5.0) -> None:
        super().__init__()

        self._net_backup = net.copy()
        self.net = net
        self.max_number_cars_per_lane = max_num_cars_per_lane
        self.success_action_reward = success_action_reward
        self.success_car_drive_reward = success_car_drive_reward

        assert lanes is None or len(lanes.keys()) == 8
        if lanes is None:
            self.lanes = [
                LanePetriNetTuple(lane='north_front', place='GreenSN'),
                LanePetriNetTuple(lane='south_front', place='GreenSN'),
                LanePetriNetTuple(lane='north_left', place='GreenSWNE'),
                LanePetriNetTuple(lane='south_left', place='GreenSWNE'),
                LanePetriNetTuple(lane='west_front', place='GreenWE'),
                LanePetriNetTuple(lane='east_front', place='GreenWE'),
                LanePetriNetTuple(lane='west_left', place='GreenWNES'),
                LanePetriNetTuple(lane='east_left', place='GreenWNES'),
            ]
        else:
            self.lanes = lanes

        print("Places:")
        for p in self.net.place():
            print("\tname: {}, tokens: {}".format(p.name, len(p.tokens.items())))
        print("Transitions:")
        for t in self.net.transition():
            print("\t{}".format(t.name))

        print("Marking:")
        print(self.net.get_marking())

        # actions are all possible net transitions [RtoGwe, RtoGsn, ... (tue nichts state)]
        self.action_space = gym.spaces.Discrete(len(self.net.transition()) + 1)
        self.actions_to_transitions = [t.name for t in self.net.transition()] + ["None"]

        # observation space:
        # {
        #   'net': {
        #       'RedSN': Tuple(Discrete(1))
        #       '
        #   },
        #   'vehicle_obs': {
        #       '0': Discrete(100),
        #       ...
        # }
        net_dict = {}
        vehicle_obs_dict = {}
        for place in self.net.place():
            net_dict[place.name] = gym.spaces.Discrete(max_num_tokens + 1)

        for lane in self.lanes:
            vehicle_obs_dict["{}".format(lane.name)] = gym.spaces.Discrete(max_num_cars_per_lane)

        self.observation_space = gym.spaces.Dict({
            'net': gym.spaces.Dict(net_dict),
            'vehicle_obs': gym.spaces.Dict(vehicle_obs_dict),
        })

    # 1 step is 3 sec (1 car per step and lane)
    def step(self, action) -> ({}, float, bool, {}):
        success = self._do_action(action)
        cars_driven = self._do_driving()
        reward = self._calculate_reward(success, cars_driven)
        observation = self._get_obs()
        terminated = self._terminated()
        info = self._info()

        return observation, reward, terminated, info

    def reset(self, seed) -> None:
        super().reset(seed=seed)

        self.net = self._net_backup.copy()
        for lane in self.lanes:
            lane.reset()

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    def flatten_observation(self, observation):
        return gym.spaces.flatten(self.observation_space, observation)

    def _calculate_reward(self, success, cars_driven) -> float:
        reward = self.success_action_reward if success else 0
        reward += cars_driven * self.success_car_drive_reward
        return reward

    def _do_action(self, action) -> bool:
        action = self.actions_to_transitions[action]
        if action in [t.name for t in self.net.transition()]:
            if len(self.net.transition(action).modes()) > 0:
                self.net.transition(action).fire(self.net.transition(action).modes()[0])
            else:
                # petri net constrains limit the action space
                return False
        elif action is "None":
            return True
        else:
            print("Undefined action space. Cannot be chosen.")
            return False

        return True

    def _do_driving(self) -> int:
        return 0

    def _get_obs(self) -> {}:
        net_dict = {}
        for place in self.net.place():
            net_dict[place.name] = len(place.tokens)
        lane_dict = {}
        for lane in self.lanes:
            lane_dict[lane.name] = lane.num_vehicles

        return {
            'net': net_dict,
            'vehicle_obs': lane_dict
        }

    def _info(self) -> {}:
        return self._get_obs()

    def _terminated(self) -> bool:
        # cars exceeded maximum number of waiting cars
        cars_exceeded = [lane.num_vehicles >= self.max_number_cars_per_lane for lane in self.lanes]

        # max execution rounds reached

        return np.any(cars_exceeded)
