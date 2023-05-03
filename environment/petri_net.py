import json

import gymnasium as gym
import numpy as np
import time


class Vehicle(object):
    def __init__(self):
        self.time_steps = 0

    def increase_time(self):
        self.time_steps = self.time_steps + 1


class LanePetriNetTuple(object):
    def __init__(self, lane: str, place: str, poisson_mu_arriving: int = 1, poisson_mu_departing: int = 8):
        self.name = lane
        self.place = place
        self.arriving_mu = poisson_mu_arriving
        self.departing_mu = poisson_mu_departing

        self.vehicles = []

    def reset(self):
        self.vehicles = []

    def add_vehicles(self):
        num_cars = np.random.poisson(self.arriving_mu)
        for i in range(num_cars):
            self.vehicles.append(Vehicle())

    def increase_time(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i].increase_time()

    def max_time(self):
        return max([v.time_steps for v in self.vehicles]) if len(self.vehicles) > 0 else 0

    def drive_vehicles(self) -> int:
        num_cars_drivable = len(self.vehicles)
        cars_driving = np.random.poisson(self.departing_mu)
        if num_cars_drivable < cars_driving:
            self.vehicles = []
            return num_cars_drivable
        else:
            vehicles = len(self.vehicles)
            for car_num in range(cars_driving):
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
    metadata = {"render_modes": ["human", "file"], "render_fps": 4}

    def __init__(self, render_mode=None, net=None, reward_function = None,
                 max_num_tokens: int = 1, max_num_cars_per_lane: int = 50,
                 lanes: [LanePetriNetTuple] = None, success_action_reward: float = 5.0,
                 success_car_drive_reward: float = 5.0, max_steps: int = 10000,
                 transitions_to_obs: bool = True, places_to_obs: bool = False) -> None:
        super().__init__()

        self._net_backup = net.copy()
        self.net = net
        self.max_number_cars_per_lane = max_num_cars_per_lane
        self.success_action_reward = success_action_reward
        self.success_car_drive_reward = success_car_drive_reward
        self.steps = 0
        self.max_steps = max_steps
        self.reward_function = reward_function
        self.transitions_to_obs = transitions_to_obs
        self.places_to_obs = places_to_obs

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.output = []

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
        observation_dict = {}
        if self.transitions_to_obs:
            for t in self.net.transition():
                observation_dict["act-{}".format(t.name)] = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float64)
            observation_dict["act-nothing"] = gym.spaces.Box(low=0.0, high=1.0, dtype=np.float64)
        if self.places_to_obs:
            for place in self.net.place():
                observation_dict["net-{}".format(place.name)] = gym.spaces.Box(low=0.0, high=float(max_num_tokens),
                                                                               dtype=np.float64)

        for lane in self.lanes:
            observation_dict["vehicle_obs-{}-n".format(lane.name)] = gym.spaces.Box(low=0.0,
                                                                                    high=float(max_num_cars_per_lane),
                                                                                    dtype=np.float64)
            observation_dict["vehicle_obs-{}-t".format(lane.name)] = gym.spaces.Box(low=0.0,
                                                                                    high=self.max_steps,
                                                                                    dtype=np.float64)

        self.observation_space = gym.spaces.Dict(observation_dict)

    # 1 step is 12 sec (4 car per step and lane)
    def step(self, action) -> ({}, float, bool, {}, {}):
        self.steps = self.steps + 1
        previous_obs = self._get_obs()
        success = self._do_action(action)
        _ = self._do_driving()
        observation = self._get_obs()
        if self.reward_function:
            reward = self.reward_function(previous_obs, observation, success, self.steps)
        else:
            reward = self._calculate_reward(previous_obs, observation, success, self.steps)
        terminated = self._terminated()
        truncated = False
        info = self._info()

        return observation, reward, terminated, truncated, info

    def reset(self, seed: int = None) -> None:
        super().reset(seed=seed)

        self.net = self._net_backup.copy()
        for lane in self.lanes:
            lane.reset()
        self.steps = 0

        return self._get_obs(), self._get_obs()

    def render(self) -> None:
        if self.render_mode == "human":
            print("t-{}, obs: {}".format(self.steps, self._get_obs()))
        if self.render_mode == "file":
            self.output.append("t-{}, obs: {}".format(self.steps, self._get_obs()))

    def close(self) -> None:
        if self.render_mode == "file":
            timestr = time.strftime("%Y%m%d-%H%M%S")
            with open("{}-output.txt".format(timestr), 'w') as filehandle:
                json.dump(self.output, filehandle)

    @staticmethod
    def flatten_observation(observation):
        flattened_obs = []
        for k in observation.keys():
            flattened_obs.append(observation[k])
        return flattened_obs

    def _calculate_reward(self, prev_obs, obs, success, timesteps) -> float:
        cars_driven = 0
        for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs")]:
            cars_driven = cars_driven + prev_obs[i][0]-obs[i][0]
        reward = self.success_action_reward if success else 0
        reward = reward + cars_driven * self.success_car_drive_reward if cars_driven > 0 else 0
        return reward

    def _do_action(self, action) -> bool:
        action = self.actions_to_transitions[action]
        if action in [t.name for t in self.net.transition()]:
            if len(self.net.transition(action).modes()) > 0:
                self.net.transition(action).fire(self.net.transition(action).modes()[0])
            else:
                # petri net constrains limit the action space
                return False
        elif action == "None":
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

            self.lanes[i].increase_time()
            self.lanes[i].add_vehicles()

        return vehicles_driven

    def _get_obs(self):
        obs = {}
        if self.transitions_to_obs:
            for t in self.net.transition():
                obs["act-{}".format(t.name)] = np.array(1.0 if len(t.modes()) > 0 else 0.0)
            obs["act-nothing"] = 1.0
        if self.places_to_obs:
            for place in self.net.place():
                obs["net-{}".format(place.name)] = np.array([len(place.tokens)])
        for lane in self.lanes:
            obs["vehicle_obs-{}-n".format(lane.name)] = np.array([len(lane.vehicles)])
            obs["vehicle_obs-{}-t".format(lane.name)] = np.array([lane.max_time()])

        return obs

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

        return np.any(cars_exceeded) or self.max_steps <= self.steps
