#!/usr/bin/env python3
"""
简单的奖励跟踪脚本
为每个参数组合生成奖励曲线
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from agents.dqn import get_dqn_model
from environment import JunctionPetriNetEnv
import rewards
from utils.petri_net import get_petri_net, Parser
from utils.log_manager import LogManager


class SimpleRewardTracker:
    """简单的奖励跟踪器"""
    
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def step(self, reward):
        """记录一步的奖励"""
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
    def end_episode(self):
        """结束一个episode，记录总奖励和长度"""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def save_reward_plot(self, timestep, final=False):
        """保存奖励曲线"""
        plt.figure(figsize=(12, 8))
        
        # 原始奖励曲线
        plt.plot(self.episode_rewards, alpha=0.7, color='lightblue', label='Episode奖励')
        plt.title(f'{"最终" if final else f"Episode {timestep}"} - {self.exp_name} 奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存文件
        if final:
            filename = f'experiments/results/final_reward_{self.exp_name}.png'
        else:
            filename = f'experiments/results/reward_{self.exp_name}_ep{timestep}.png'
            
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'奖励曲线已保存到: {filename}')


def train_with_rewards(args):
    """带奖励跟踪的训练函数"""
    
    # 构建奖励函数
    reward_params = {
        'success': args.m_success,
        'cars_driven': args.m_cars_driven,
        'waiting_time': args.m_waiting_time,
        'max_waiting_time': args.m_max_waiting_time,
        'timestep': args.m_timestep,
    }
    
    if args.constrained:
        reward_function = rewards.constrained_reward
        algorithm_name = "CDQN"
    else:
        reward_function = rewards.discounted_reward
        algorithm_name = "DQN"
    
    # 设置日志管理器
    log_manager = LogManager(algorithm_name, reward_params, log_dir="experiments/logs")
    
    # 构建环境
    env = JunctionPetriNetEnv(
        reward_function=reward_function,
        net=get_petri_net("data/traffic-scenario.PNPRO", type=Parser.PNPRO),
        transitions_to_obs=True,
        places_to_obs=False,
    )
    env = wrap_env(env, wrapper="gymnasium")
    
    # 构建智能体
    memory = RandomMemory(memory_size=500000, num_envs=env.num_envs)
    cfg = DQN_DEFAULT_CONFIG.copy()
    
    agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=args.constrained)
    
    # 创建奖励跟踪器
    reward_tracker = SimpleRewardTracker(args.exp_name)
    
    # 创建 skrl 训练器
    trainer = SequentialTrainer(cfg, env, agent)
    
    print(f"开始训练 {algorithm_name}，参数组合: {reward_params}")
    
    # 训练循环
    for timestep in range(args.timesteps):
        # 执行 skrl 训练步骤
        step_info = trainer.train(timestep, 1)
        
        # 获取环境反馈
        if hasattr(step_info, 'infos') and step_info['infos']:
            info = step_info['infos'][0]  # 假设单环境
            reward = info.get('reward', 0)
            
            # 记录奖励
            reward_tracker.step(reward)
            
            # 检查是否episode结束
            if hasattr(info, 'terminated') and info['terminated']:
                reward_tracker.end_episode()
        
        # 每500步保存一次奖励曲线
        if timestep > 0 and timestep % 500 == 0:
            reward_tracker.save_reward_plot(timestep)
    
    # 训练结束后保存最终奖励曲线
    reward_tracker.save_reward_plot(args.timesteps, final=True)
    
    # 保存最终模型
    final_model_path = f"models/{args.exp_name}_final.pt"
    agent.save(final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(prog="simple_reward_tracker")
    
    # 基本参数
    parser.add_argument("--exp-name", type=str, required=True, help="实验名称")
    parser.add_argument("--constrained", action='store_true', help="使用CDQN")
    
    # 奖励函数参数
    parser.add_argument("--m-success", type=float, default=1.0)
    parser.add_argument("--m-cars-driven", type=float, default=1.0)
    parser.add_argument("--m-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-max-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-timestep", type=float, default=1.0)
    
    # 训练参数
    parser.add_argument("--timesteps", type=int, default=3000, help="训练步数")
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments/logs", exist_ok=True)
    
    # 开始训练
    train_with_rewards(args)


if __name__ == "__main__":
    main()