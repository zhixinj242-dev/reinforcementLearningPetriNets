"""
【文件角色】：AI 的“奖金制度”仓库。
这里定义了各种不同的奖励函数（Reward Functions）。AI 的学习目标就是最大化这个函数的得分。
通过切换不同的奖励函数，你可以训练出性格截然不同的 AI（例如：有的 AI 拼命追求通行量，有的 AI 则更在意公平性）。
"""

def constraint_driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    """
    【函数功能】：基于“等待时间”和“成功约束”的奖惩逻辑。
    【逻辑细节】：
    1. 成功执行动作给 +200 分（鼓励守法）。
    2. 总等待时间越长，扣分越多（总容量 400 减去当前总等待时间）。
    3. 运行步数（时间）越长，给一点点加分（鼓励系统不崩溃）。
    """
    waiting_times = 0 if len(waiting_times) == 0 else sum(waiting_times)
    r = 0
    r = r + 200 if success else 0
    r = r + ((50*8) - waiting_times) * 2
    r = r + timestep * 0.5
    return r


def driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    """与上面类似，但去掉了对“成功执行动作”的奖励（即不强制守法，只看交通效率）"""
    waiting_times = 0 if len(waiting_times) == 0 else sum(waiting_times)
    r = 0
    r = r + ((50*8) - waiting_times) * 2
    r = r + timestep * 0.5
    return r


def constraint_avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    """基于平均等待时间的奖励"""
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 0
    r = r + 200 if success else 0
    r = r + ((50*8)-waiting_times)
    r = r + timestep * 0.5
    return r


def avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    """不带约束惩罚的平均等待时间奖励"""
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 0
    r = r + ((50*8)-waiting_times)
    r = r + timestep * 0.5
    return r


def constraint_timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    """最简单的生存奖励：只要没堵死，活得越久分越高（守法还有额外奖）"""
    r = 0
    r = r + 200 if success else 0
    r = r + timestep * 0.5
    return r


def timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    """纯生存奖励"""
    r = 0
    r = r + timestep * 0.5
    return r


def constraint_cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    """侧重于通行效率：每走一辆车给 4 分，再加上生存时长奖"""
    cars_driven = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    r = 0
    r = r + 200 if success else 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + timestep * 0.5
    return r


def cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    """不带约束惩罚的通行效率奖励"""
    cars_driven = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    r = 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + timestep * 0.5
    return r


# --- 【超参数】：通过修改下面的乘数，你可以一键改变 AI 的性格 ---
success_multiplier = 1          # 守法奖金倍数
car_driven_multiplier = 1       # 通行量奖金倍数
waiting_time_multiplier = 1     # 等待时间惩罚倍数（越大 AI 越不能忍受排队）
max_waiting_time_multiplier = 1 # 针对最惨车辆的惩罚倍数
timestep_multiplier = 1         # 生存时长奖励倍数


def dynamic_reward(prev_obs, obs, success, timestep, waiting_times) -> float:
    """
    【核心函数】：动态综合奖励。
    【特点】：它综合了上述所有因素，并且可以通过上面的“乘数”来灵活调参。
    这是最推荐使用的奖励函数，因为它最全面。
    """
    cars_driven = 0
    waiting_times = []
    # 统计这一步走了多少车
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    # 统计每条车道剩下的车等了多久
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times.append(obs[i][0])
    
    r = 0
    r = r + 200 * success_multiplier if success else 0 # 守法奖
    r = r + (cars_driven + 5) * car_driven_multiplier if cars_driven > 0 else 0 # 通行奖
    r = r + ((50 * 8) - sum(waiting_times)) * waiting_time_multiplier # 整体排队惩罚
    r = r + (50 - max(waiting_times)) * max_waiting_time_multiplier # 最长等待惩罚（公平性）
    r = r + timestep * timestep_multiplier # 生存时长奖
    return r


def base_reward(prev_obs, obs, success, timestep, waiting_times) -> float:
    """一个标准的基础奖励配置"""
    cars_driven = 0
    waiting_times = []
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times.append(obs[i][0])
    r = 0
    r = r + 200 if success else 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + ((50*8)-sum(waiting_times))
    r = r + (50-max(waiting_times))
    r = r + timestep * 0.5
    return r

def discounted_reward(prev_obs, obs, success, timestep, waiting_times) -> float:
    """
    【特殊函数】：带衰减因子的奖励。
    它把分数缩小了 20 倍，目的是为了让神经网络学习起来更稳定，防止奖励值过大导致网络梯度爆炸。
    """
    discount_factor = 20
    cars_driven = 0
    waiting_times = []
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times.append(obs[i][0])
    r = 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + ((50 * 8) - sum(waiting_times)) * 0.1
    r = r + (50 - max(waiting_times)) * 0.5
    r = r + timestep * 2
    return r/discount_factor


def reward_without_time(prev_obs, obs, success, timestep, waiting_times) -> float:
    """极简奖惩：不看时间，只看走了多少车和有没有犯规"""
    cars_driven = 0
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 20 if success else 0
    r = r + (cars_driven + 5) * 2 if cars_driven > 0 else 0
    return r
