import torch
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import get_petri_net, Parser

agent_path = "runs/train_202304271635_exp-400000t-0.04e_200000lrate_200000randtsteps/checkpoints/best_agent.pt"


def main():
    env = JunctionPetriNetEnv(render_mode="human", reward_function=base_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO), transitions_to_obs=False, places_to_obs=True)
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    agent = get_dqn_model(env, memory=None, cfg=DQN_DEFAULT_CONFIG.copy())

    agent.load(agent_path)
    agent.set_mode("eval")
    agent.set_running_mode("eval")

    terminated = False
    obs, _ = env.reset()
    t = 0
    while not terminated:
        action = torch.argmax(agent.q_network.act({"states": obs}, role="q_network")[0], dim=1, keepdim=True)

        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        env.render()

    iterations = 1000
    total_timesteps = []
    for i in range(iterations):
        obs, _ = env.reset()
        t = 0
        terminated = False
        while not terminated:
            action = torch.argmax(agent.q_network.act({"states": obs}, role="q_network")[0], dim=1, keepdim=True)
            obs, reward, terminated, _, _ = env.step(action)
            t = t + 1
        total_timesteps.append(t)

    print(total_timesteps)
    print(float(sum(total_timesteps)) / float(iterations))


if __name__ == "__main__":
    main()
