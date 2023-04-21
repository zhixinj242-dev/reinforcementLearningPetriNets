from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
from utils.petri_net import get_petri_net, Parser


def reward(prev_obs, obs, success) -> float:
    cars_driven = 0
    waiting_times = 0
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("n")]:
        cars_driven = cars_driven + prev_obs[i][0] - obs[i][0]
    for i in [key for key in prev_obs.keys() if key.startswith("vehicle_obs") and key.endswith("t")]:
        waiting_times = waiting_times + obs[i][0]
    r = 1.0 if success else 0
    r = r + cars_driven * 0.2 if cars_driven > 0 else 0
    r = r - waiting_times * 0.05
    return r


def main():
    env = JunctionPetriNetEnv(reward_function=reward, net=get_petri_net('data/traffic-scenario.PNPRO',
                                                                        type=Parser.PNPRO))
    env.reset()
    env = wrap_env(env, wrapper="gymnasium")
    device = env.device
    memory = RandomMemory(memory_size=100000, num_envs=env.num_envs)

    cfg = DQN_DEFAULT_CONFIG.copy()

    dqn_agent = get_dqn_model(env=env, device=device, memory=memory, cfg=cfg)

    cfg_trainer = {
        "timesteps": 600000,
        "headless": True
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=dqn_agent)

    trainer.train()
    trainer.eval()


if __name__ == '__main__':
    main()
