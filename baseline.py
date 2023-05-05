import torch
from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import get_petri_net, Parser
import numpy as np

def main():
    env = JunctionPetriNetEnv(render_mode="human", reward_function=base_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO), transitions_to_obs=True, places_to_obs=False)
    env.reset()

    terminated = False
    # action_sequence = []
    action_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # action_sequence = [0, 8, 3, 1, 8, 2, 4, 8, 5, 6, 8, 7]
    action_sequence = [0, 3, 1, 2, 4, 5, 6, 7]
    action_sequence = [1, 2, 0, 3, 6, 7, 4, 5]
    action_sequence = [4, 5, 6, 7, 1, 8, 2, 0, 8, 3]
    action_sequence = [4, 5, 6, 7, 1, 8, 2, 0, 8, 3]

    env.reset()
    t = 0
    while not terminated:
        if len(action_sequence) == 0:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = action_sequence[t % len(action_sequence)]

        obs, reward, terminated, truncated, info = env.step(action)
        t = t + 1
        print(reward)
        env.render()

    iterations = 100
    total_timesteps = []
    constraints_broken = []
    average_waiting_times = []
    min_waiting_times = []
    max_waiting_times = []
    for i in range(iterations):
        env.reset()
        c = 0
        num_cars_driven = 0
        sum_waiting_times = 0
        min_waiting_time = 10000
        max_waiting_time = 0
        t = 0
        terminated = False
        while not terminated:
            if len(action_sequence) == 0:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = action_sequence[t % len(action_sequence)]
            obs, reward, terminated, _, inf = env.step(action)
            if not inf["success"]:
                c = c + 1
            num_cars_driven = num_cars_driven + inf["num_cars_driven"]
            sum_waiting_times = sum_waiting_times + sum(inf["waiting_times"])
            if len(inf["waiting_times"]) > 0:
                min_waiting_time = min([min_waiting_time, min(inf["waiting_times"])])
                max_waiting_time = max([max_waiting_time, max(inf["waiting_times"])])
            t = t + 1
        print("Iteration {} finished".format(i))
        total_timesteps.append(t)
        constraints_broken.append(c)
        average_waiting_times.append(float(sum_waiting_times) / float(num_cars_driven))
        if min_waiting_time == 10000:
            continue
        min_waiting_times.append(min_waiting_time)
        max_waiting_times.append(max_waiting_time)

    print(total_timesteps)
    print(float(sum(total_timesteps)) / float(iterations))
    print(average_waiting_times)
    print("Min. constraints broken: {}, {}%".format(min(constraints_broken),
                                                    min(constraints_broken)/float(max(total_timesteps))*100))
    print("Max. constraints broken: {}, {}%".format(max(constraints_broken),
                                                    max(constraints_broken)/float(max(total_timesteps))*100))
    print("Avg. constraints broken: {}, {}%".format(sum(constraints_broken)/len(constraints_broken),
                                                    sum(constraints_broken)/len(constraints_broken) /
                                                    float(max(total_timesteps))*100))
    print("Avg. waiting time: {}".format(float(sum(average_waiting_times)) / float(len(average_waiting_times))))
    print("Min. waiting time: {}".format(float(sum(min_waiting_times)) / float(len(min_waiting_times))))
    print("Max. waiting time: {}".format(float(sum(max_waiting_times)) / float(len(max_waiting_times))))


if __name__ == "__main__":
    main()
