from environment import PetriNetEnvArray
from utils.petri_net import get_petri_net, Parser
import random


def main():
    print('Starting training...')
    env = PetriNetEnvArray(net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))

    print("Action space: {}".format(env.action_space))
    print("Observation space: {}".format(env.observation_space))

    for i in range(200):
        observation, reward, terminated, _ = env.step(random.randint(0, 8))

        print("{}, {}, {}".format(terminated, reward, observation))


if __name__ == '__main__':
    main()
