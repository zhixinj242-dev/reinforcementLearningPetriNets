#import gymnasium as gym
from stable_baselines3 import DQN

class QLearningAgent(object):

    def __init__(self, env, T):
        self.model = DQN("MultiInputPolicy", env, verbose=1,)
        self.model.learn(total_timesteps=T, log_interval=10)
        #model.save("dqn_rlpn")

    def get_action(self, obs):
        action, states = self.model.predict(obs, deterministic=True)
        return action, states