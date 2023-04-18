from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from agents.dqn import get_dqn_model
from environment import PetriNetEnvArray
from utils.petri_net import get_petri_net, Parser


def main():
    env = PetriNetEnvArray(net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO))
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    device = env.device
    memory = RandomMemory(memory_size=100000, num_envs=env.num_envs)

    cfg = DQN_DEFAULT_CONFIG.copy()

    dqn_agent = get_dqn_model(env=env, device=device, memory=memory, cfg=cfg)

    cfg_trainer = {
        "timesteps": 300000,
        "headless": True
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=dqn_agent)

    trainer.train()
    trainer.eval()


if __name__ == '__main__':
    main()
