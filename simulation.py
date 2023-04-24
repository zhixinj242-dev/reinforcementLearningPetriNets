from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
from rewards import base_reward
from utils.petri_net import get_petri_net, Parser

agent_path = "runs/checkpoints/best_agent.pt"


def main():
    env = JunctionPetriNetEnv(render_mode="human", reward_function=base_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    agent = get_dqn_model(env, memory=None, cfg=DQN_DEFAULT_CONFIG.copy())

    terminated = False
    obs, _ = env.reset()
    while not terminated:
        action, _, _ = agent.act(obs, 0, 0)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()