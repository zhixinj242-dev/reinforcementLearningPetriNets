import gymnasium as gym


class PetriNetEnv(gym.Env):

    def __init__(self, net=None, num_lanes: int = 8, simulate_cars: bool = True) -> None:
        super().__init__()

        self.net = net
        self.num_lanes = num_lanes
        self.simulate_cars = simulate_cars

        print("Places:")
        for p in self.net.place():
            print("\tname: {}, tokens: {}".format(p.name, len(p.tokens.items())))
        print("Transitions:")
        for t in self.net.transition():
            print("\t{}".format(t.name))

        print("Marking:")
        print(self.net.get_marking())

        # actions are all possible net transitions [RtoGwe, RtoGsn, ...]
        self.action_space = gym.spaces.Discrete(len(self.net.transition()))
        # observation space:
        # {
        #   'net': {
        #       'RedSN': Tuple(Discrete(1))
        #       '
        #   },
        #   'vehicle_obs': {
        #       '0': Discrete(100),
        #       ...
        # }
        self.observation_space = gym.spaces.Discrete(len(self.net.place()) + num_lanes)

    def step(self, action):
        pass

    def reset(self, **kwargs):
        pass

    def render(self):
        pass

    def close(self):
        pass
