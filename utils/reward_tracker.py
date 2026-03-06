"""
【文件角色】：奖励曲线生成器。
在训练过程中实时生成奖励曲线，只记录违规动作，大幅减少日志量。
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
from collections import deque


class RewardTracker:
    """奖励曲线跟踪器，实时生成奖励曲线"""
    
    def __init__(self, output_dir="plots", algorithm_type="DQN", reward_params=None, window_size=100):
        """初始化奖励跟踪器
        
        Args:
            output_dir: 输出目录
            algorithm_type: 算法类型 (DQN/CDQN)
            reward_params: 奖励参数
            window_size: 移动平均窗口大小
        """
        self.output_dir = output_dir
        self.algorithm_type = algorithm_type
        self.reward_params = reward_params or {}
        self.window_size = window_size
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成基础文件名
        reward_str = "_".join([f"{k}{v}" for k, v in self.reward_params.items()])
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.base_filename = f"{algorithm_type}_{reward_str}_{timestamp}"
        
        # 数据存储
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_violations = []
        
        # 设置图形
        plt.style.use('default')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle(f'{algorithm_type} 奖励曲线 - 参数: {reward_str}', fontsize=14)
        
        print(f"[RewardTracker] 输出目录: {output_dir}")
        print(f"[RewardTracker] 基础文件名: {self.base_filename}")
    
    def update_episode(self, episode_reward, episode_length, episode_violations):
        """更新episode数据
        
        Args:
            episode_reward: episode总奖励
            episode_length: episode长度
            episode_violations: episode违规次数
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_violations.append(episode_violations)
        
        # 每10个episode更新一次图形
        if len(self.episode_rewards) % 10 == 0:
            self.update_plot()
    
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
        
        print(f"[RewardTracker] 奖励曲线已更新并保存到: {plot_path}")
    
    def save_final_plot(self):
        """保存最终奖励曲线"""
        self.update_plot()
        
        # 保存数据为CSV
        csv_path = os.path.join(self.output_dir, f"{self.base_filename}_rewards_data.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("episode,reward,length,violations\n")
            for i, (reward, length, violations) in enumerate(zip(self.episode_rewards, self.episode_lengths, self.episode_violations)):
                f.write(f"{i+1},{reward},{length},{violations}\n")
        
        print(f"[RewardTracker] 最终数据已保存到: {csv_path}")
        
        # 保存统计信息
        stats = {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "total_violations": sum(self.episode_violations),
            "violation_rate": sum(self.episode_violations) / sum(self.episode_lengths) if sum(self.episode_lengths) > 0 else 0
        }
        
        stats_path = os.path.join(self.output_dir, f"{self.base_filename}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"[RewardTracker] 统计信息已保存到: {stats_path}")
        print(f"[RewardTracker] 统计: {stats}")
    
    def close(self):
        """关闭图形"""
        self.save_final_plot()
        plt.close(self.fig)