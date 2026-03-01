#!/usr/bin/env python3
"""
带奖励曲线的训练脚本
为每个参数组合生成训练过程和奖励曲线
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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


class RewardTracker:
    """奖励跟踪器，记录每个episode的奖励变化"""
    
    def __init__(self):
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
        
    def get_smoothed_rewards(self, window_size=100):
        """获取平滑的奖励曲线"""
        if len(self.episode_rewards) < window_size:
            return self.episode_rewards
        
        smoothed_rewards = []
        for i in range(len(self.episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_avg = np.mean(self.episode_rewards[start_idx:end_idx])
            smoothed_rewards.append(window_avg)
        
        return smoothed_rewards
    
    def plot_rewards(self, save_path, title="训练奖励曲线"):
        """绘制并保存奖励曲线"""
        plt.figure(figsize=(12, 8))
        
        # 原始奖励曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, alpha=0.3, color='lightblue', label='原始奖励')
        plt.title('原始奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.grid(True)
        
        # 平滑奖励曲线
        plt.subplot(2, 1, 2)
        smoothed_rewards = self.get_smoothed_rewards()
        plt.plot(smoothed_rewards, color='darkblue', linewidth=2, label='平滑奖励(窗口=100)')
        plt.title('平滑奖励曲线')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"奖励曲线已保存到: {save_path}")


class CustomTrainer:
    """自定义训练器，集成奖励跟踪"""
    
    def __init__(self, cfg, env, agent, reward_tracker, log_manager):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.reward_tracker = reward_tracker
        self.log_manager = log_manager
        
        # 创建 skrl 训练器
        self.trainer = SequentialTrainer(cfg, env, agent)
        
    def train_step(self, timestep, timesteps):
        """训练步骤，集成奖励跟踪"""
        # 执行 skrl 训练步骤
        step_info = self.trainer.train(timestep, timesteps)
        
        # 获取环境反馈
        if hasattr(step_info, 'infos') and step_info['infos']:
            info = step_info['infos'][0]  # 假设单环境
            reward = info.get('reward', 0)
            
            # 记录奖励
            self.reward_tracker.step(reward)
            
            # 检查是否episode结束
            if hasattr(info, 'terminated') and info['terminated']:
                self.reward_tracker.end_episode()
                
                # 记录到日志
                log_data = {
                    'episode': len(self.reward_tracker.episode_rewards),
                    'total_reward': self.reward_tracker.episode_rewards[-1],
                    'episode_length': self.reward_tracker.episode_lengths[-1],
                    'average_reward': np.mean(self.reward_tracker.episode_rewards[-100:]),
                }
                
                self.log_manager.log_step(timestep, {
                    'reward_tracking': log_data
                })
        
        return step_info
    
    def train(self, timesteps):
        """完整训练过程"""
        print(f"开始训练 {timesteps} 步...")
        
        for timestep in range(timesteps):
            self.train_step(timestep, 1)
            
            # 定期保存奖励曲线
            if timestep > 0 and timestep % 500 == 0:
                reward_plot_path = f"experiments/results/reward_plot_ep{timestep}.png"
                self.reward_tracker.plot_rewards(reward_plot_path, 
                    title=f"训练奖励曲线 (Episode {timestep})")
        
        # 训练结束后保存最终奖励曲线
        final_reward_path = "experiments/results/final_reward_curve.png"
        self.reward_tracker.plot_rewards(final_reward_path, 
            title="最终训练奖励曲线")
        
        print(f"训练完成！最终奖励曲线已保存到: {final_reward_path}")


def create_experiment_directory(exp_name):
    """创建实验目录"""
    exp_dir = f"experiments/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    return exp_dir


def main():
    parser = argparse.ArgumentParser(prog="train_with_rewards")
    
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
    parser.add_argument("--save-interval", type=int, default=500, help="保存奖励曲线间隔")
    
    args = parser.parse_args()
    
    # 创建实验目录
    exp_dir = create_experiment_directory(args.exp_name)
    
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
    log_manager = LogManager(algorithm_name, reward_params, log_dir=f"{exp_dir}/logs")
    
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
    cfg["experiment"]["directory"] = exp_dir
    cfg["experiment"]["experiment_name"] = args.exp_name
    cfg["experiment"]["write_interval"] = args.save_interval
    
    agent = get_dqn_model(env=env, memory=memory, cfg=cfg, constrained=args.constrained)
    
    # 创建奖励跟踪器
    reward_tracker = RewardTracker()
    
    # 创建自定义训练器
    trainer = CustomTrainer(cfg, env, agent, reward_tracker, log_manager)
    
    # 开始训练
    print(f"开始训练 {algorithm_name}，参数组合: {reward_params}")
    trainer.train(args.timesteps)
    
    # 保存最终模型
    final_model_path = f"models/{args.exp_name}_final.pt"
    agent.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")


if __name__ == "__main__":
    main()