<<<<<<< HEAD
from environment import JunctionPetriNetEnv
from agents import QLearningAgent
from utils import get_petri_net, Parser
=======
from environment import PetriNetEnvArray
from utils.petri_net import get_petri_net, Parser
>>>>>>> b30d46ef95d43d30eb166accd991bd088c5bc1cb
import random


def main():
    print('Starting training...')
    env = PetriNetEnvArray(net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))

    print("Action space: {}".format(env.action_space))
    print("Observation space: {}".format(env.observation_space))
    T = 1000 #total timesteps T

<<<<<<< HEAD
    agent = QLearningAgent(env, T)
    obs = env.reset()

    for i in range(T):
        action, _states = agent.get_action(obs)
        obs, reward, terminated, _ = env.step(action)
        print(reward)
        if terminated:
            obs = env.reset()
=======
    for i in range(200):
        observation, reward, terminated, _ = env.step(random.randint(0, 8))

        print("{}, {}, {}".format(terminated, reward, observation))
>>>>>>> b30d46ef95d43d30eb166accd991bd088c5bc1cb


if __name__ == '__main__':
    main()
