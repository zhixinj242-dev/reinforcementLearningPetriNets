import gymnasium.spaces as spaces
import numpy as np

from .petri_net import JunctionPetriNetEnv, LanePetriNetTuple

"""
【文件角色】：数据适配器（翻译官）。
它是 `JunctionPetriNetEnv` 的派生类（亲儿子）。
【核心作用】：把老爸产出的复杂“字典格式”观测，压扁成一串纯数字的“长列表（Array）”。
【适用场景】：专门为了兼容那些只认长列表输入、不认字典的强化学习算法库（如 Stable Baselines3）。
"""

class PetriNetEnvArray(JunctionPetriNetEnv):

    def __init__(self, render_mode=None, net=None, max_num_tokens: int = 1, max_num_cars_per_lane: int = 50,
                 lanes: [LanePetriNetTuple] = None, success_action_reward: float = 5.0,
                 success_car_drive_reward: float = 5.0, max_steps: int = 300) -> None:
        super().__init__(render_mode=render_mode, net=net, max_num_tokens=max_num_tokens,
                         max_num_cars_per_lane=max_num_cars_per_lane,
                         lanes=lanes, success_action_reward=success_action_reward,
                         success_car_drive_reward=success_car_drive_reward, max_steps=max_steps)

        # 重新定义观测空间为 Box（连续向量空间）
        lower_bound = np.zeros(len(self.observation_space.keys()), dtype=np.float32)
        upper_bound = np.zeros(len(self.observation_space.keys()), dtype=np.float32)
        for idx, dict in enumerate(self.observation_space):
            upper_bound[idx] = self.observation_space[dict].high
        self.observation_space = spaces.Box(low=lower_bound, high=upper_bound, shape=lower_bound.shape, dtype=np.float32)

    def _get_obs(self):
        """
        【函数功能】：观测展平逻辑。
        【重写细节】：不再返回字典，而是把所有数据按顺序排好，拼成一个一维向量。
        排序顺序：1.所有库所的Token数 -> 2.每条车道的车辆数 -> 3.每条车道的最长等待时间。
        """
        obs = np.zeros(self.observation_space.shape)
        idx = 0
        # 1. 填入灯态信息（库所中的 Token 数）
        for place in self.net.place():
            obs[idx] = len(place.tokens)
            idx = idx + 1
        # 2. 填入交通流信息（各车道的排队情况）
        for lane in self.lanes:
            obs[idx] = len(lane.vehicles)      # 该车道排了几个车
            obs[idx+1] = lane.max_time()       # 等得最惨的人等了多久
            idx = idx + 2

        return obs

    def _calculate_reward(self, prev_obs, obs, success) -> float:
        """针对展平后观测的奖励计算（通过索引访问数据）"""
        cars_driven = 0
        # 假设前 9 位是灯态，从第 10 位开始是车辆数据
        for i in range(9, len(prev_obs)):
            cars_driven = cars_driven + prev_obs[i]-obs[i]
        reward = self.success_action_reward if success else 0
        reward = reward + cars_driven * self.success_car_drive_reward if cars_driven > 0 else 0
        return reward
