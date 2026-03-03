import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv

class RewardTracker:
    """
    优化的奖励曲线跟踪器，提供高效的奖励数据记录和分析
    """
    
    def __init__(self, algorithm_type, reward_params, output_dir="experiments/results", 
                 window_size=100, save_interval=10):
        """
        初始化奖励跟踪器
        
        Args:
            algorithm_type: 算法类型，如 "CDQN" 或 "DQN"
            reward_params: 奖励函数参数，字典格式
            output_dir: 输出目录
            window_size: 滑动窗口大小，用于计算移动平均
            save_interval: 保存间隔，每多少个episode保存一次数据
        """
        self.algorithm_type = algorithm_type
        self.reward_params = reward_params
        self.output_dir = output_dir
        self.window_size = window_size
        self.save_interval = save_interval
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        params_str = "_".join([f"{k}{v}" for k, v in sorted(reward_params.items())])
        self.base_filename = f"{self.algorithm_type}_{params_str}_{timestamp}"
        
        # 数据存储
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_violations = []
        self.step_rewards = []
        self.moving_avg_rewards = deque(maxlen=window_size)
        
        # 统计信息
        self.best_episode_reward = float('-inf')
        self.worst_episode_reward = float('inf')
        self.current_episode = 0
        self.current_step = 0
        
        # CSV文件
        self.csv_file = open(os.path.join(output_dir, f"{self.base_filename}_rewards.csv"), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Episode', 'TotalReward', 'AverageStepReward', 'EpisodeLength', 'Violations', 'MovingAvg'])
        
        # 初始化图表
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle(f'{algorithm_type} 奖励曲线 - 参数: {params_str}', fontsize=16)
        
    def log_step(self, reward):
        """
        记录单步奖励
        
        Args:
            reward: 单步奖励值
        """
        self.step_rewards.append(reward)
        self.current_step += 1
    
    def log_episode_end(self, episode, total_reward, episode_length, violations=0):
        """
        记录episode结束信息
        
        Args:
            episode: episode编号
            total_reward: 总奖励
            episode_length: episode长度
            violations: 违规次数
        """
        self.current_episode = episode
        
        # 计算平均步奖励
        avg_step_reward = total_reward / episode_length if episode_length > 0 else 0
        
        # 记录数据
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.episode_violations.append(violations)
        self.moving_avg_rewards.append(total_reward)
        
        # 更新最佳/最差奖励
        if total_reward > self.best_episode_reward:
            self.best_episode_reward = total_reward
        if total_reward < self.worst_episode_reward:
            self.worst_episode_reward = total_reward
        
        # 计算移动平均
        moving_avg = np.mean(self.moving_avg_rewards) if self.moving_avg_rewards else total_reward
        
        # 写入CSV
        self.csv_writer.writerow([episode, total_reward, avg_step_reward, episode_length, violations, moving_avg])
        self.csv_file.flush()
        
        # 定期更新图表和保存数据
        if episode % self.save_interval == 0:
            self.update_plot()
            self.save_data()
    
    def update_plot(self):
        """更新奖励曲线图"""
        if not self.episode_rewards:
            return
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 清除之前的图
        self.ax1.clear()
        self.ax2.clear()
        
        # 上图：原始奖励和移动平均
        self.ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.7, label='Episode奖励')
        
        # 计算并绘制移动平均
        if len(self.episode_rewards) >= self.window_size:
            moving_averages = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - self.window_size + 1)
                moving_averages.append(np.mean(self.episode_rewards[start_idx:i+1]))
            
            self.ax1.plot(episodes, moving_averages, 'r-', linewidth=2, label=f'{self.window_size}步移动平均')
        
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('奖励')
        self.ax1.set_title('Episode奖励曲线')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # 下图：违规次数
        self.ax2.bar(episodes, self.episode_violations, alpha=0.7, color='orange', label='违规次数')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('违规次数')
        self.ax2.set_title('Episode违规次数')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # 调整布局并保存
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{self.base_filename}_rewards_plot.png")
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        print(f"奖励曲线已更新并保存到: {plot_path}")
    
    def save_data(self):
        """保存数据到JSON文件"""
        data = {
            "algorithm_type": self.algorithm_type,
            "reward_params": self.reward_params,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_violations": self.episode_violations,
            "statistics": {
                "total_episodes": len(self.episode_rewards),
                "best_episode_reward": self.best_episode_reward,
                "worst_episode_reward": self.worst_episode_reward,
                "average_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
                "average_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
                "total_violations": sum(self.episode_violations),
                "episodes_with_violations": sum(1 for v in self.episode_violations if v > 0),
                "violation_rate": sum(self.episode_violations) / sum(self.episode_lengths) if sum(self.episode_lengths) > 0 else 0
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        json_path = os.path.join(self.output_dir, f"{self.base_filename}_data.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"奖励数据已保存到: {json_path}")
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.episode_rewards:
            return {}
        
        recent_episodes = min(100, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_episodes:]
        
        return {
            "total_episodes": len(self.episode_rewards),
            "best_episode_reward": self.best_episode_reward,
            "worst_episode_reward": self.worst_episode_reward,
            "average_reward": np.mean(self.episode_rewards),
            "recent_average_reward": np.mean(recent_rewards),
            "average_episode_length": np.mean(self.episode_lengths),
            "total_violations": sum(self.episode_violations),
            "episodes_with_violations": sum(1 for v in self.episode_violations if v > 0),
            "violation_rate": sum(self.episode_violations) / sum(self.episode_lengths) if sum(self.episode_lengths) > 0 else 0,
            "recent_violation_rate": sum(self.episode_violations[-recent_episodes:]) / sum(self.episode_lengths[-recent_episodes:]) if sum(self.episode_lengths[-recent_episodes:]) > 0 else 0
        }
    
    def close(self):
        """关闭跟踪器，保存最终数据"""
        # 最终更新图表
        self.update_plot()
        
        # 保存最终数据
        self.save_data()
        
        # 关闭CSV文件
        if not self.csv_file.closed:
            self.csv_file.close()
        
        # 输出统计信息
        stats = self.get_statistics()
        print(f"\n=== 奖励统计 ===")
        print(f"总episode数: {stats['total_episodes']}")
        print(f"最佳episode奖励: {stats['best_episode_reward']:.2f}")
        print(f"最差episode奖励: {stats['worst_episode_reward']:.2f}")
        print(f"平均奖励: {stats['average_reward']:.2f}")
        print(f"最近100个episode平均奖励: {stats['recent_average_reward']:.2f}")
        print(f"平均episode长度: {stats['average_episode_length']:.2f}")
        print(f"总违规次数: {stats['total_violations']}")
        print(f"有违规的episode数: {stats['episodes_with_violations']}")
        print(f"违规率: {stats['violation_rate']:.4f}")
        print(f"最近违规率: {stats['recent_violation_rate']:.4f}")
    
    def get_csv_file_path(self):
        """获取CSV文件路径"""
        return os.path.join(self.output_dir, f"{self.base_filename}_rewards.csv")