from environment import JunctionPetriNetEnv
from utils.petri_net import get_petri_net, Parser
import random


def main():
    print('Starting training...')
    env = JunctionPetriNetEnv(net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))

    print("Action space: {}".format(env.action_space))
    print("Observation space: {}".format(env.observation_space))

    for i in range(12):
        observation, reward, terminated, _ = env.step(random.randint(0, 8))

        print(env.flatten_observation(observation))
        # print(observation)


if __name__ == '__main__':
    main()
