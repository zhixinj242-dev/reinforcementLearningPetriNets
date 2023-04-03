from environment import JunctionPetriNetEnv
from agents import QLearningAgent
from utils import get_petri_net, Parser
import random


def main():
    print('Starting training...')
    env = JunctionPetriNetEnv(net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))

    print("Action space: {}".format(env.action_space))
    print("Observation space: {}".format(env.observation_space))
    T = 1000 #total timesteps T

    agent = QLearningAgent(env, T)
    obs = env.reset()

    for i in range(T):
        action, _states = agent.get_action(obs)
        obs, reward, terminated, _ = env.step(action)
        print(reward)
        if terminated:
            obs = env.reset()


if __name__ == '__main__':
    main()
