"""
【文件角色】：独立的奖励记录器。
专门记录奖励信息，生成滑动平均曲线，不依赖违规日志。
"""
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class RewardLogger:
    """独立的奖励记录器，专门记录奖励信息"""
    
    def __init__(self, log_dir="reward_logs", algorithm_type="DQN", reward_params=None, window_size=100):
        """初始化奖励记录器
        
        Args:
            log_dir: 日志目录
            algorithm_type: 算法类型 (DQN/CDQN)
            reward_params: 奖励参数
            window_size: 滑动窗口大小
        """
        self.log_dir = log_dir
        self.algorithm_type = algorithm_type
        self.reward_params = reward_params or {}
        self.window_size = window_size
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        reward_str = "_".join([f"{k}{v}" for k, v in self.reward_params.items()])
        self.base_filename = f"{algorithm_type}_{reward_str}_{timestamp}"
        
        # 数据存储
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_violations = []
        self.step_rewards = []
        self.recent_rewards = deque(maxlen=window_size)
        
        # 打开奖励日志文件
        self.reward_log_file = open(os.path.join(log_dir, f"{self.base_filename}_rewards.jsonl"), 'w', encoding='utf-8')
        
        print(f"[RewardLogger] 奖励日志文件: {self.reward_log_file.name}")
    
    def log_step(self, step, reward, info):
        """记录单步奖励
        
        Args:
            step: 当前步数
            reward: 单步奖励
            info: 环境信息
        """
        self.step_rewards.append(reward)
        self.recent_rewards.append(reward)
        
        # 记录到日志文件
        log_entry = {
            "step": step,
            "reward": reward,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "environment_acceptance": info.get("environment_acceptance", True)
        }
        
        self.reward_log_file.write(json.dumps(log_entry, ensure_ascii=False))
        self.reward_log_file.write("\n")
        self.reward_log_file.flush()
    
    def log_episode(self, episode, total_reward, length, violations):
        """记录episode结束信息
        
        Args:
            episode: episode编号
            total_reward: episode总奖励
            length: episode长度
            violations: 违规次数
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_violations.append(violations)
        
        # 计算滑动平均
        if len(self.episode_rewards) >= self.window_size:
            recent_rewards = self.episode_rewards[-self.window_size:]
            moving_avg = sum(recent_rewards) / len(recent_rewards)
        else:
            moving_avg = sum(self.episode_rewards) / len(self.episode_rewards)
        
        # 记录episode日志
        episode_log = {
            "episode": episode,
            "total_reward": total_reward,
            "length": length,
            "violations": violations,
            "moving_avg": moving_avg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        self.reward_log_file.write(json.dumps(episode_log, ensure_ascii=False))
        self.reward_log_file.write("\n")
        self.reward_log_file.flush()
        
        # 每50个episode更新一次图表
        if episode % 50 == 0:
            self.update_plot(episode)
        
        # 每100个episode输出一次统计信息
        if episode % 100 == 0:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
            total_violations = sum(self.episode_violations)
            violation_rate = total_violations / sum(self.episode_lengths) if sum(self.episode_lengths) > 0 else 0
            
            print(f"[RewardLogger] Episode {episode}: 平均奖励={avg_reward:.2f}, 平均长度={avg_length:.2f}, 违规率={violation_rate:.4f}")
    
    def update_plot(self, current_episode=None):
        """
        更新奖励曲线图
        
        Args:
            current_episode: 当前episode编号
        """
        if not self.episode_rewards:
            return
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 子图1: 原始奖励
        plt.subplot(2, 2, 1)
        plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.7, label='Episode奖励')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.title('Episode奖励曲线')
        plt.legend()
        plt.grid(True)
        
        # 子图2: 滑动平均
        plt.subplot(2, 2, 2)
        if len(self.episode_rewards) >= self.window_size:
            moving_averages = []
            for i in range(len(self.episode_rewards)):
                start_idx = max(0, i - self.window_size + 1)
                moving_averages.append(np.mean(self.episode_rewards[start_idx:i+1]))
            
            plt.plot(episodes, moving_averages, 'r-', linewidth=2, label=f'{self.window_size}步滑动平均')
        else:
            # 如果数据不足，使用全部数据计算平均
            moving_avg = sum(self.episode_rewards) / len(self.episode_rewards)
            plt.axhline(y=moving_avg, color='r', linestyle='-', label=f'平均奖励: {moving_avg:.2f}')
        
        plt.xlabel('Episode')
        plt.ylabel('平均奖励')
        plt.title('滑动平均奖励')
        plt.legend()
        plt.grid(True)
        
        # 子图3: Episode长度
        plt.subplot(2, 2, 3)
        plt.plot(episodes, self.episode_lengths, 'g-', alpha=0.7, label='Episode长度')
        plt.xlabel('Episode')
        plt.ylabel('步数')
        plt.title('Episode长度')
        plt.legend()
        plt.grid(True)
        
        # 子图4: 违规次数
        plt.subplot(2, 2, 4)
        plt.bar(episodes, self.episode_violations, alpha=0.7, color='orange', label='违规次数')
        plt.xlabel('Episode')
        plt.ylabel('违规次数')
        plt.title('Episode违规次数')
        plt.legend()
        plt.grid(True)
        
        # 添加总标题
        reward_str = "_".join([f"{k}{v}" for k, v in self.reward_params.items()])
        plt.suptitle(f'{self.algorithm_type} 奖励曲线 - 参数: {reward_str}', fontsize=16)
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = os.path.join(self.log_dir, f"{self.base_filename}_rewards_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 只在特定条件下输出信息，减少终端输出
        if current_episode and current_episode % 100 == 0:
            print(f"[RewardLogger] 奖励曲线已更新: {plot_path}")
        
        # 输出当前统计信息
        if current_episode:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            avg_length = sum(self.episode_lengths) / len(self.episode_lengths)
            total_violations = sum(self.episode_violations)
            violation_rate = total_violations / sum(self.episode_lengths) if sum(self.episode_lengths) > 0 else 0
            
            print(f"[RewardLogger] Episode {current_episode}: 平均奖励={avg_reward:.2f}, 平均长度={avg_length:.2f}, 违规率={violation_rate:.4f}")
    
    def get_current_stats(self):
        """获取当前统计信息"""
        if not self.episode_rewards:
            return {}
        
        return {
            "episodes": len(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "max_reward": max(self.episode_rewards),
            "min_reward": min(self.episode_rewards),
            "avg_length": sum(self.episode_lengths) / len(self.episode_lengths),
            "total_violations": sum(self.episode_violations),
            "violation_rate": sum(self.episode_violations) / sum(self.episode_lengths) if sum(self.episode_lengths) > 0 else 0
        }
    
    def save_final_data(self):
        """保存最终数据"""
        # 保存CSV数据
        csv_path = os.path.join(self.log_dir, f"{self.base_filename}_rewards_data.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("episode,reward,length,violations\n")
            for i, (reward, length, violations) in enumerate(zip(self.episode_rewards, self.episode_lengths, self.episode_violations)):
                f.write(f"{i+1},{reward},{length},{violations}\n")
        
        # 保存统计信息
        stats = self.get_current_stats()
        stats_path = os.path.join(self.log_dir, f"{self.base_filename}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"[RewardLogger] 最终数据已保存: {csv_path}")
        print(f"[RewardLogger] 统计信息已保存: {stats_path}")
        print(f"[RewardLogger] 最终统计: {stats}")
    
    def close(self):
        """关闭日志文件"""
        self.save_final_data()
        if self.reward_log_file:
            self.reward_log_file.close()
            print(f"[RewardLogger] 奖励日志已关闭: {self.reward_log_file.name}")