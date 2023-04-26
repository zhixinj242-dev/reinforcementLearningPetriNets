from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from datetime import datetime

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
import rewards
from utils.petri_net import get_petri_net, Parser


def main():
    env = JunctionPetriNetEnv(reward_function=rewards.base_reward,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    memory = RandomMemory(memory_size=100000, num_envs=env.num_envs)

    cfg = DQN_DEFAULT_CONFIG.copy()
    cfg["batch_size"] = 64
    cfg["exploration"]["timesteps"] = 600000
    cfg["exploration"]["final_epsilon"] = 0.04
    cfg["learning_starts"] = 20000
    cfg["random_timesteps"] = 10000
    cfg["experiment"]["checkpoint_interval"] = 100000
    cfg["experiment"]["write_interval"] = 10000
    cfg["experiment"]["experiment_name"] = "{}_exp-{}t-{}e_{}lrate_{}randtsteps"\
        .format(datetime.now().strftime("%Y%m%d%H%M"), cfg["exploration"]["timesteps"],
                cfg["exploration"]["final_epsilon"], cfg["learning_starts"], cfg["random_timesteps"])

    dqn_agent = get_dqn_model(env=env, memory=memory, cfg=cfg)
    # dqn_agent.load("runs/23-04-26_13-38-36-989612_DQN/checkpoints/best_agent.pt")

    cfg_trainer = {
        "timesteps": 3000000,
        "headless": True
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=dqn_agent)

    trainer.train()
    # trainer.eval()


if __name__ == '__main__':
    main()
