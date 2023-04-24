

def base_reward(prev_obs, obs, success) -> float:
    cars_driven = 0
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 1.0 if success else 0
    r = r + cars_driven * 0.2 if cars_driven > 0 else 0
    r = r - waiting_times * 0.05
    return r