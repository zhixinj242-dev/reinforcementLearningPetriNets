

def constraint_driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    waiting_times = 0 if len(waiting_times) == 0 else sum(waiting_times)
    r = 0
    r = r + 200 if success else 0
    r = r + ((50*8) - waiting_times) * 2
    r = r + timestep * 0.5
    return r


def driven_waiting_times_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    waiting_times = 0 if len(waiting_times) == 0 else sum(waiting_times)
    r = 0
    r = r + ((50*8) - waiting_times) * 2
    r = r + timestep * 0.5
    return r


def constraint_avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 0
    r = r + 200 if success else 0
    r = r + ((50*8)-waiting_times)
    r = r + timestep * 0.5
    return r


def avg_waiting_times_and_timesteps(prev_obs, obs, success, timestep, waiting_times) -> float:
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 0
    r = r + ((50*8)-waiting_times)
    r = r + timestep * 0.5
    return r


def constraint_timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    r = 0
    r = r + 200 if success else 0
    r = r + timestep * 0.5
    return r


def timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    r = 0
    r = r + timestep * 0.5
    return r


def constraint_cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    cars_driven = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    r = 0
    r = r + 200 if success else 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + timestep * 0.5
    return r


def cars_driven_timestep(prev_obs, obs, success, timestep, waiting_times) -> float:
    cars_driven = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    r = 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + timestep * 0.5
    return r


success_multiplier = 1
car_driven_multiplier = 1
waiting_time_multiplier = 1
max_waiting_time_multiplier = 1
timestep_multiplier = 1


def dynamic_reward(prev_obs, obs, success, timestep, waiting_times) -> float:
    cars_driven = 0
    waiting_times = []
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times.append(obs[i][0])
    r = 0
    r = r + 200 * success_multiplier if success else 0
    r = r + (cars_driven + 5) * car_driven_multiplier if cars_driven > 0 else 0
    r = r + ((50 * 8) - sum(waiting_times)) * waiting_time_multiplier
    r = r + (50 - max(waiting_times)) * max_waiting_time_multiplier
    r = r + timestep * timestep_multiplier
    return r


def base_reward(prev_obs, obs, success, timestep, waiting_times) -> float:
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
    discount_factor = 20
    cars_driven = 0
    waiting_times = []
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times.append(obs[i][0])
    r = 0
    #r = r + 20 if success else 0
    r = r + (cars_driven + 5) * 4 if cars_driven > 0 else 0
    r = r + ((50 * 8) - sum(waiting_times)) * 0.1
    r = r + (50 - max(waiting_times)) * 0.5
    r = r + timestep * 2
    return r/discount_factor


def reward_without_time(prev_obs, obs, success, timestep, waiting_times) -> float:
    cars_driven = 0
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 20 if success else 0
    r = r + (cars_driven + 5) * 2 if cars_driven > 0 else 0
    return r
