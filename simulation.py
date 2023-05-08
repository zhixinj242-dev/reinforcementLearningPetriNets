import torch
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
import pandas as pd

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import get_petri_net, Parser

params = [
    (0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.5, 0.0),
    (1.0, 0.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0, 1.5, 0.0),
    (1.5, 0.0, 1.0, 0.0, 0.0),
    (1.5, 0.0, 1.0, 1.5, 0.0),
    (2.0, 0.0, 1.0, 0.0, 0.0),
    (2.0, 0.0, 1.0, 1.5, 0.0)
]

df = pd.DataFrame(columns=['s', 'c', 'w', 'mw', 't', 'min_t_frame', 'avg_t_frame', 'max_t_frame', 'min_waiting_time',
                           'avg_waiting_time', 'max_waiting_time', 'min_c_broken', 'avg_c_broken', 'max_c_broken'])


def main(p):
    path = "lido-run-events/agent_s{}c{}w{}mw{}t{}.pt".format(p[0], p[1], p[2], p[3], p[4])
    print(path)
    env = JunctionPetriNetEnv(render_mode="human", reward_function=base_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO), transitions_to_obs=True, places_to_obs=False)
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    agent = get_dqn_model(env, memory=None, cfg=DQN_DEFAULT_CONFIG.copy())

    agent.load(path)
    agent.set_mode("eval")
    agent.set_running_mode("eval")

    terminated = False
    obs, _ = env.reset()
    #t = 0
    #while not terminated:
    #    action = torch.argmax(agent.q_network.act({"states": obs}, role="q_network")[0], dim=1, keepdim=True)

    #    obs, reward, terminated, truncated, info = env.step(action)
    #    print(reward)
    #    env.render()

    iterations = 200
    total_timesteps = []
    constraints_broken = []
    average_waiting_times = []
    min_waiting_times = []
    max_waiting_times = []
    for i in range(iterations):
        obs, _ = env.reset()
        c = 0
        num_cars_driven = 0
        sum_waiting_times = 0
        min_waiting_time = 10000
        max_waiting_time = 0
        t = 0
        terminated = False
        while not terminated:
            action = torch.argmax(agent.q_network.act({"states": obs}, role="q_network")[0], dim=1, keepdim=True)
            obs, reward, terminated, _, inf = env.step(action)
            if not inf["success"]:
                c = c + 1
            num_cars_driven = num_cars_driven + inf["num_cars_driven"]
            sum_waiting_times = sum_waiting_times + sum(inf["waiting_times"])
            if len(inf["waiting_times"]) > 0:
                min_waiting_time = min([min_waiting_time, min(inf["waiting_times"])])
                max_waiting_time = max([max_waiting_time, max(inf["waiting_times"])])
            t = t + 1
        # print("Iteration {} finished".format(i))
        total_timesteps.append(t)
        constraints_broken.append(c)
        average_waiting_times.append(float(sum_waiting_times) / float(num_cars_driven))
        min_waiting_times.append(min_waiting_time)
        max_waiting_times.append(max_waiting_time)

    avg_timesteps = float(sum(total_timesteps)) / float(iterations)
    min_timesteps = min(total_timesteps)
    max_timesteps = max(total_timesteps)
    print(total_timesteps)
    print(avg_timesteps)
    print(average_waiting_times)
    min_constraints_broken = min(constraints_broken)/float(max(total_timesteps))*100
    print("Min. constraints broken: {}%".format(min_constraints_broken))
    avg_constraints_broken = sum(constraints_broken)/len(constraints_broken) / float(max(total_timesteps))*100
    print("Avg. constraints broken: {}%".format(avg_constraints_broken))
    max_constraints_broken = max(constraints_broken)/float(max(total_timesteps))*100
    print("Max. constraints broken: {}%".format(max_constraints_broken))
    min_waiting_time = float(sum(min_waiting_times)) / float(len(min_waiting_times))
    print("Min. waiting time: {}".format(min_waiting_time))
    avg_waiting_time = float(sum(average_waiting_times)) / float(len(average_waiting_times))
    print("Avg. waiting time: {}".format(avg_waiting_time))
    max_waiting_time = float(sum(max_waiting_times)) / float(len(max_waiting_times))
    print("Max. waiting time: {}".format(max_waiting_time))

    # row = {
    #     's': p[0], 'c': p[1], 'w': p[2], 'mw': p[3], 't': p[4],
    #     'min_t_frame': min_timesteps,
    #     'avg_t_frame': avg_timesteps,
    #     'max_t_frame': max_timesteps,
    #     'min_waiting_time': min_waiting_time,
    #     'avg_waiting_time': avg_waiting_time,
    #     'max_waiting_time': max_waiting_time,
    #     'min_c_broken': min_constraints_broken,
    #     'avg_c_broken': avg_constraints_broken,
    #     'max_c_broken': max_constraints_broken
    # }
    # df.append(row, ignore_index=True)
    df.loc[len(df.index)] = [p[0], p[1], p[2], p[3], p[4], min_timesteps, avg_timesteps, max_timesteps,
                             min_waiting_time, avg_waiting_time, max_waiting_time, min_constraints_broken,
                             avg_constraints_broken, max_constraints_broken]


if __name__ == "__main__":
    for p in params:
        main(p)

    df.to_csv("simulation-results.csv")
