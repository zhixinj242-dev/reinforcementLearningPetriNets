import json

import gymnasium as gym
import numpy as np
import time

"""
【文件角色】：十字路口 Petri 网环境的核心实现。
这里定义了路口的物理规则（车辆如何移动、如何排队）和逻辑规则（红绿灯如何与 Petri 网绑定）。
"""

class Vehicle(object):
    """
    【类功能】：模拟路口的一辆“车”。
    【属性】：time_steps 记录了这辆车在路口等了多久。
    """
    def __init__(self):
        self.time_steps = 0

    def increase_time(self):
        """每过一个时间步，等待时间就加 1"""
        self.time_steps = self.time_steps + 1


class LanePetriNetTuple(object):
    """
    【类功能】：车道与 Petri 网库所的“绑定关系户”。
    它把物理上的“车道”和逻辑上的“Petri 网节点（Place）”串联起来。
    """
    def __init__(self, lane: str, place: str, poisson_mu_arriving: int = 1, poisson_mu_departing: int = 8):
        """
        【参数解释】：
        - lane: 车道名称（如 'north_front'）。
        - place: 绑定的 Petri 网库所名称（如 'GreenSN'，表示这个库所有 Token 时，该车道是绿灯）。
        - poisson_mu_arriving: 【环境参数】泊松分布均值，决定平均每步新来几辆车。
        - poisson_mu_departing: 【环境参数】泊松分布均值，决定绿灯时平均每步走几辆车。
        """
        self.name = lane
        self.place = place
        self.arriving_mu = poisson_mu_arriving
        self.departing_mu = poisson_mu_departing

        self.vehicles = [] # 当前车道排队中的车辆列表

    def reset(self):
        self.vehicles = []

    def add_vehicles(self):
        """模拟车辆随机到达过程"""
        num_cars = np.random.poisson(self.arriving_mu)
        for i in range(num_cars):
            self.vehicles.append(Vehicle())

    def increase_time(self):
        """车道上所有还在排队的车辆等待时间 +1"""
        for i in range(len(self.vehicles)):
            self.vehicles[i].increase_time()

    def max_time(self):
        """获取当前车道等得最久的那辆车的时间"""
        return max([v.time_steps for v in self.vehicles]) if len(self.vehicles) > 0 else 0

    def drive_vehicles(self) -> (int, int):
        """
        【功能】：绿灯亮起，模拟车辆通过路口。
        【返回】：(驶离的车辆数, 这些车的总等待时间列表)
        """
        num_cars_drivable = len(self.vehicles)
        cars_driving = np.random.poisson(self.departing_mu) # 随机决定这波绿灯能走多少辆
        if num_cars_drivable < cars_driving:
            waiting_times = [v.time_steps for v in self.vehicles]
            driven = len(self.vehicles)
            self.vehicles = []
            return driven, waiting_times
        else:
            vehicles_before = len(self.vehicles)
            waiting_times = []
            for car_num in range(cars_driving):
                car = self._get_longest_waiting_vehicle() # 优先让等得最久的走
                waiting_times.append(car.time_steps)
                self.vehicles.remove(car)
            return vehicles_before - len(self.vehicles), waiting_times

    def _get_longest_waiting_vehicle(self):
        if len(self.vehicles) == 0:
            return None
        max_vehicle = self.vehicles[0]
        for v in self.vehicles:
            if v.time_steps > max_vehicle.time_steps:
                max_vehicle = v
        return max_vehicle


class JunctionPetriNetEnv(gym.Env):
    """
    【核心类】：十字路口 Petri 网环境。
    它是路口的“上帝”，掌握着所有的路况、灯态和奖惩。
    """
    metadata = {"render_modes": ["human", "file"], "render_fps": 4}

    def __init__(self, render_mode=None, net=None, reward_function = None,
                 max_num_tokens: int = 1, max_num_cars_per_lane: int = 50,
                 lanes: [LanePetriNetTuple] = None, success_action_reward: float = 5.0,
                 success_car_drive_reward: float = 5.0, max_steps: int = 1000,
                 transitions_to_obs: bool = True, places_to_obs: bool = False, log_manager=None) -> None:
        """
        【初始化】：创建世界。
        
        【可改参数】：
        1. max_num_cars_per_lane: 车道容量（默认50）。改小了路口极易堵死导致模拟结束。
        2. success_action_reward: 动作合法奖励（默认5.0）。
        3. success_car_drive_reward: 通行效率奖励（默认5.0）。
        """
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
        self.log_manager = log_manager

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.output = []

        # 设置默认的车道配置及其与 Petri 网库所的映射关系
        if lanes is None:
            self.lanes = [
                LanePetriNetTuple(lane='north_front', place='GreenSN', poisson_mu_arriving=2, poisson_mu_departing=13),
                LanePetriNetTuple(lane='south_front', place='GreenSN', poisson_mu_arriving=2, poisson_mu_departing=13),
                LanePetriNetTuple(lane='north_left', place='GreenSWNE', poisson_mu_arriving=1, poisson_mu_departing=8),
                LanePetriNetTuple(lane='south_left', place='GreenSWNE', poisson_mu_arriving=1, poisson_mu_departing=8),
                LanePetriNetTuple(lane='west_front', place='GreenWE', poisson_mu_arriving=3, poisson_mu_departing=13),
                LanePetriNetTuple(lane='east_front', place='GreenWE', poisson_mu_arriving=3, poisson_mu_departing=13),
                LanePetriNetTuple(lane='west_left', place='GreenWNES', poisson_mu_arriving=1, poisson_mu_departing=8),
                LanePetriNetTuple(lane='east_left', place='GreenWNES', poisson_mu_arriving=1, poisson_mu_departing=8),
            ]
        else:
            self.lanes = lanes

        # AI 的动作空间：Petri 网的所有变迁 + 一个"什么都不做"的动作
        # 确保变迁顺序与掩码顺序一致（按名称排序）
        transitions = sorted(self.net.transition(), key=lambda t: t.name)
        self.actions_to_transitions = [t.name for t in transitions] + ["None"]
        self.action_space = gym.spaces.Discrete(len(self.actions_to_transitions))

        # 构建观测空间（告诉 AI 它能看到什么）
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

    def __del__(self):
        # 关闭日志文件
        if hasattr(self, "log_manager") and self.log_manager:
            self.log_manager.close()

    def step(self, action) -> ({}, float, bool, {}, {}):
        """
        【函数功能】：时间步推行。AI每做一个决策（动作），这个函数就运行一次。
        它负责：1.执行动作 2.模拟车流 3.计算奖惩 4.返回新情况。
        """
        self.steps = self.steps + 1
        previous_obs = self._get_obs() # 记录动作前的路口情况
        
        # 添加调试信息：记录动作选择时的掩码状态
        action_name = self.actions_to_transitions[action] if action < len(self.actions_to_transitions) else "Unknown"
        # 安全获取动作掩码值
        if action_name == "None":
            action_mask = 1.0
        else:
            mask_val = previous_obs.get(f"act-{action_name}", np.array([0.0]))
            # 确保mask_val是可索引的
            if mask_val.ndim == 0:
                action_mask = float(mask_val)
            else:
                action_mask = float(mask_val[0])
        
        # 1. 执行AI的决策：比如AI想切灯，这里会检查Petri网准不准切
        success = self._do_action(action)
        
        # 2. 模拟车辆移动：绿灯方向的车开走，红灯方向继续等，并随机新来一些车
        cars_driven, waiting_times = self._do_driving()
        
        # 3. 获取动作后的新情况
        observation = self._get_obs()
        
        # 4. 计算奖励：做得好给正分，做得差扣分
        if self.reward_function:
            reward = self.reward_function(previous_obs, observation, success, self.steps, waiting_times)
        else:
            reward = self._calculate_reward(previous_obs, observation, success, self.steps, waiting_times)
        
        # 5. 检查是否结束（比如堵死了或时间到了）
        terminated = self._terminated()
        truncated = False
        info = self._info(success, cars_driven, waiting_times) # 额外的信息包，方便调试

        # 记录环境接受情况、合法动作等信息
        # 使用previous_obs（动作前的状态）而不是observation（动作后的状态）
        # 注意：environment_legal_actions必须基于previous_obs计算，而不是当前状态
        legal_actions_from_previous_state = []
        
        # 【关键修复】：按照actions_to_transitions的顺序构建合法动作列表，确保与智能体一致
        for action_name in self.actions_to_transitions:
            if action_name == "None":
                # "None"动作总是合法的
                legal_actions_from_previous_state.append(action_name)
            else:
                # 检查该动作的掩码值
                mask_key = f"act-{action_name}"
                if mask_key in previous_obs:
                    mask_val = previous_obs[mask_key]
                    if mask_val.ndim == 0:
                        mask_value = float(mask_val)
                    else:
                        mask_value = float(mask_val[0])
                    
                    # 如果掩码为1.0，则该动作合法
                    if mask_value == 1.0:
                        legal_actions_from_previous_state.append(action_name)
        
        environment_info = {
            "environment_acceptance": success,
            "environment_legal_actions": legal_actions_from_previous_state,
            "environment_legal_state": previous_obs  # 使用动作前的状态
        }

        # 将环境信息添加到info中，以便代理记录
        info.update(environment_info)
        
        # 环境信息已经在代理的record_transition方法中记录，这里不再单独记录
        # if self.log_manager:
        #     # 构建环境日志信息
        #     env_log_info = {
        #         "environment_acceptance": success,
        #         "environment_legal_actions": [t.name for t in self.net.transition() if len(t.modes()) > 0] + ["None"],
        #         "environment_legal_state": self._convert_numpy_to_list(self._get_obs())
        #     }
        #     # 记录环境信息到日志
        #     self.log_manager.log_step(self.steps, env_log_info)

        return observation, reward, terminated, truncated, info

    def _convert_numpy_to_list(self, obj):
        """
        将包含NumPy数组的对象转换为包含Python列表的对象，以便JSON序列化
        
        Args:
            obj: 要转换的对象
            
        Returns:
            转换后的对象
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    def reset(self, seed: int = None) -> tuple:
        """重置环境，回到最初的状态"""
        super().reset(seed=seed)

        self.net = self._net_backup.copy()
        for lane in self.lanes:
            lane.reset()
        self.steps = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def render(self) -> None:
        """渲染环境（打印日志或保存到文件）"""
        if self.render_mode == "human":
            print("t-{}, obs: {}".format(self.steps, self._get_obs()))
        if self.render_mode == "file":
            self.output.append("t-{}, obs: {}".format(self.steps, self._get_obs()))

    def close(self) -> None:
        """关闭环境，保存最后的输出结果"""
        if self.render_mode == "file":
            filename = "traffic_animation.png"
            with open("traffic_output.txt", 'w') as filehandle:
                json.dump(self.output, filehandle)

    def flatten_observation(self, observation):
        """将字典格式的观测打平成一个列表，确保掩码顺序与actions_to_transitions一致"""
        flattened_obs = []
        
        # 提取所有动作掩码键
        mask_keys = [k for k in observation.keys() if k.startswith("act-")]
        
        # 提取动作名称（去掉"act-"前缀）
        action_names = [k.replace("act-", "") for k in mask_keys]
        
        # 【关键修复】：按照字母顺序排序，与actions_to_transitions一致
        # 排除"nothing"，只保留实际动作
        actions = sorted([name for name in action_names if name != "nothing"])
        
        # 按照字母顺序添加掩码
        for action in actions:
            key = f"act-{action}"
            if key in observation:
                flattened_obs.append(observation[key])
        
        # 最后添加"act-nothing"（始终在最后）
        if "act-nothing" in observation:
            flattened_obs.append(observation["act-nothing"])
          
        # 【关键修复】：按照self.lanes的顺序添加车辆观测，确保与_get_obs一致
        # 先添加库所观测（按字母顺序）
        place_keys = sorted([k for k in observation.keys() if k.startswith("net-")])
        for k in place_keys:
            flattened_obs.append(observation[k])
        
        # 再添加车辆观测（按照self.lanes的顺序）
        for lane in self.lanes:
            key_n = f"vehicle_obs-{lane.name}-n"
            key_t = f"vehicle_obs-{lane.name}-t"
            if key_n in observation:
                flattened_obs.append(observation[key_n])
            if key_t in observation:
                flattened_obs.append(observation[key_t])
        
        return flattened_obs

    def _calculate_reward(self, prev_obs, obs, success, timesteps, waiting_times) -> float:
        """默认的奖励计算逻辑"""
        cars_driven = 0
        for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs")]:
            cars_driven = cars_driven + prev_obs[i][0]-obs[i][0]
        reward = self.success_action_reward if success else 0
        reward = reward + cars_driven * self.success_car_drive_reward if cars_driven > 0 else 0
        return reward

    def _do_action(self, action) -> bool:
        """执行 Petri 网变迁。如果变迁不符合规则（没有 Mode），则返回 False。"""
        action = self.actions_to_transitions[action]
        if action in [t.name for t in self.net.transition()]:
            # 记录动作执行前的mode数量
            transition = self.net.transition(action)
            modes_before = len(transition.modes())
            
            if modes_before > 0:
                # 记录执行的mode
                mode = transition.modes()[0]
                transition.fire(mode)
            else:
                # petri net constrains limit the action space
                return False
        elif action == "None":
            return True
        else:
            return False

        return True

    def _do_driving(self) -> (int, int):
        """核心仿真逻辑：找出当前哪些库所有 Token（即哪些路口是绿灯），让对应的车道跑起来。"""
        active_places = self._active_places()
        vehicles_driven = 0
        waiting_times = []
        for i in range(len(self.lanes)):
            # 如果该车道绑定的库所有 Token，说明现在是绿灯，可以通行
            if self.lanes[i].place in active_places:
                driven, wait_time = self.lanes[i].drive_vehicles()
                vehicles_driven = vehicles_driven + driven
                waiting_times.extend(wait_time)

            # 无论红绿灯，车道里的车等待时间都增加，并可能有新车进入
            self.lanes[i].increase_time()
            self.lanes[i].add_vehicles()

        return vehicles_driven, waiting_times

    def _get_obs(self):
        """获取当前路口的快照（包括灯态掩码、库所 Token 数、各车道车辆数和最长等待时间）。"""
        obs = {}
        if self.transitions_to_obs:
            # 按照actions_to_transitions的顺序添加掩码，确保顺序一致
            # 注意：actions_to_transitions包含"None"，但掩码中"act-nothing"是单独的
            for i, action_name in enumerate(self.actions_to_transitions[:-1]):  # 排除最后一个"None"
                # 确保返回的是数组，而不是标量
                transition = self.net.transition(action_name)
                obs["act-{}".format(action_name)] = np.array([1.0 if len(transition.modes()) > 0 else 0.0])
            # 添加"nothing"动作的掩码（始终为1.0）
            obs["act-nothing"] = np.array([1.0])
        if self.places_to_obs:
            for place in self.net.place():
                obs["net-{}".format(place.name)] = np.array([len(place.tokens)])
        for lane in self.lanes:
            obs["vehicle_obs-{}-n".format(lane.name)] = np.array([len(lane.vehicles)])
            obs["vehicle_obs-{}-t".format(lane.name)] = np.array([lane.max_time()])

        return obs

    def _active_places(self):
        """找出当前哪些库所有 Token（黑点）。"""
        active_places = []
        for place in self.net.place():
            if len(place.tokens.items()) > 0:
                active_places.append(place.name)
        return active_places

    def _info(self, success, cars_driven, waiting_times) -> {}:
        """返回辅助信息包。"""
        inf = self._get_obs()
        inf["success"] = success
        inf["num_cars_driven"] = cars_driven
        inf["waiting_times"] = waiting_times
        return inf

    def _terminated(self) -> bool:
        """检查路口是否崩溃（任何一条车道满了）或时间到。"""
        # cars exceeded maximum number of waiting cars
        cars_exceeded = [len(lane.vehicles) >= self.max_number_cars_per_lane for lane in self.lanes]

        return np.any(cars_exceeded) or self.max_steps <= self.steps
