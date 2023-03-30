import random

import gymnasium as gym
import numpy as np


class Vehicle(object):
    def __init__(self):
        self.time_steps = 0

    def increase_time(self):
        self.time_steps = self.time_steps + 1


class LanePetriNetTuple(object):
    def __init__(self, lane: str, place: str, probability: float = 1.0/8.0, min_cars_pc: int = 0,
                 max_cars_pc: int = 1, car_driving_speed: int = 4):
        self.name = lane
        self.place = place
        self.probability = probability
        self.min_cars_pc = min_cars_pc
        self.max_cars_pc = max_cars_pc
        self.car_driving_speed = car_driving_speed

        self.vehicles = []

    def reset(self):
        self.vehicles = []

    def add_vehicles(self):
        num_cars = random.randint(self.min_cars_pc, self.max_cars_pc)
        for i in range(num_cars):
            self.vehicles.append(Vehicle())

    def increase_time(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].increase_time()

    def drive_vehicles(self) -> int:
        num_cars_drivable = len(self.vehicles)
        if num_cars_drivable < self.car_driving_speed:
            self.vehicles = []
            return num_cars_drivable
        else:
            vehicles = len(self.vehicles)
            for car_num in range(self.car_driving_speed):
                self.vehicles.remove(self._get_longest_waiting_vehicle())
            return vehicles - len(self.vehicles)

    def _get_longest_waiting_vehicle(self):
        if len(self.vehicles) == 0:
            return None
        max_vehicle = self.vehicles[0]
        for i in range(len(self.vehicles)):
            if self.vehicles[i].time_steps > max_vehicle.time_steps:
                max_vehicle = self.vehicles[i]
        return max_vehicle


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

        # print("Places:")
        # for p in self.net.place():
        #     print("\tname: {}, tokens: {}".format(p.name, len(p.tokens.items())))
        # print("Transitions:")
        # for t in self.net.transition():
        #     print("\t{}".format(t.name))
        #
        # print("Marking:")
        # print(self.net.get_marking())

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

    # 1 step is 12 sec (4 car per step and lane)
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
        active_places = self._active_places()
        vehicles_driven = 0
        for i in range(len(self.lanes)):
            if self.lanes[i].place in active_places:
                vehicles_driven = vehicles_driven + self.lanes[i].drive_vehicles()
                print("{}: {}".format(self.lanes[i].place, vehicles_driven))

            self.lanes[i].add_vehicles()

        return vehicles_driven

    def _get_obs(self) -> {}:
        net_dict = {}
        for place in self.net.place():
            net_dict[place.name] = len(place.tokens)
        lane_dict = {}
        for lane in self.lanes:
            lane_dict[lane.name] = len(lane.vehicles)

        return {
            'net': net_dict,
            'vehicle_obs': lane_dict
        }

    def _active_places(self):
        active_places = []
        for place in self.net.place():
            if len(place.tokens.items()) > 0:
                active_places.append(place.name)
        return active_places

    def _info(self) -> {}:
        return self._get_obs()

    def _terminated(self) -> bool:
        # cars exceeded maximum number of waiting cars
        cars_exceeded = [len(lane.vehicles) >= self.max_number_cars_per_lane for lane in self.lanes]

        # max execution rounds reached

        return np.any(cars_exceeded)
