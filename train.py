import argparse

from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from datetime import datetime

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
import rewards
from utils.petri_net import get_petri_net, Parser


def generate_parsed_arguments():
    parser = argparse.ArgumentParser(prog="RLPN-Train")
    parser.add_argument("-t", "--train", action='store_true', default=False)
    parser.add_argument("-e", "--eval", action='store_true', default=False)
    parser.add_argument("-p", "--path", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-exp-t", "--exploration-timesteps", type=int, default=600000)
    parser.add_argument("-exp-e", "--exploration-final-epsilon", type=float, default=0.04)
    parser.add_argument("-learn-s", "--learning-starts", type=int, default=20000)
    parser.add_argument("-rand-t", "--random-timesteps", type=int, default=10000)

    return parser.parse_args()


def main():
    parser = generate_parsed_arguments()

    env = JunctionPetriNetEnv(reward_function=rewards.discounted_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    memory = RandomMemory(memory_size=500000, num_envs=env.num_envs)

    state = "{}{}".format("train" if parser.train else "", "eval" if parser.eval else "")

    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = parser.batch_size
    cfg["exploration"]["timesteps"] = parser.exploration_timesteps
    cfg["exploration"]["final_epsilon"] = parser.exploration_final_epsilon
    cfg["learning_starts"] = parser.learning_starts
    cfg["random_timesteps"] = parser.random_timesteps
    cfg["experiment"]["checkpoint_interval"] = 100000
    cfg["experiment"]["write_interval"] = 10000
    cfg["experiment"]["experiment_name"] = "{}_{}_exp-{}t-{}e_{}lrate_{}randtsteps"\
        .format(state, datetime.now().strftime("%Y%m%d%H%M"), cfg["exploration"]["timesteps"],
                cfg["exploration"]["final_epsilon"], cfg["learning_starts"], cfg["random_timesteps"])

    dqn_agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=True)
    if parser.path is not None:
        dqn_agent.load(parser.path)

    cfg_trainer = {
        "timesteps": 7000000,
        "headless": True
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=dqn_agent)

    if parser.train:
        trainer.train()
    if parser.eval:
        trainer.eval()


if __name__ == '__main__':
    main()
