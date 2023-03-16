from environment import PetriNetEnv
from utils.petri_net import get_petri_net, Parser


def main():
	print('Starting training...')
	env = PetriNetEnv(net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))
	
	print("Action space: {}".format(env.action_space))
	print("Observation space: {}".format(env.observation_space))


if __name__ == '__main__':
    main()
