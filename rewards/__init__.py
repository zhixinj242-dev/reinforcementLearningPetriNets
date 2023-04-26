

def cars_driving_and_waiting_times(prev_obs, obs, success, timestep) -> float:
    cars_driven = 0
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 0
    r = r + (cars_driven + 5) * 2 if cars_driven > 0 else 0
    r = r - waiting_times * 0.01
    return r

def waiting_times(prev_obs, obs, success, timestep) -> float:
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = waiting_times * -0.1
    return r


def base_reward(prev_obs, obs, success, timestep) -> float:
    cars_driven = 0
    waiting_times = []
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times.append(obs[i][0])
    r = 0
    r = r + 20 if success else 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + ((50*8)-sum(waiting_times)) * 0.1
    r = r + (50-max(waiting_times)) * 0.5
    r = r + timestep * 2
    return r


def reward_without_time(prev_obs, obs, success, timestep) -> float:
    cars_driven = 0
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 20 if success else 0
    r = r + (cars_driven + 5) * 2 if cars_driven > 0 else 0
    return r
