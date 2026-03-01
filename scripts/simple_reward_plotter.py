#!/usr/bin/env python3
"""
极简奖励曲线生成器
直接从训练日志中提取奖励数据并绘图
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import glob

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def extract_rewards_from_log(log_file):
    """从日志文件中提取奖励数据"""
    rewards = []
    episodes = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配日志中的奖励信息
                if 'episode' in line.lower() and 'reward' in line.lower():
                    # 简单匹配，根据实际日志格式调整
                    match = re.search(r'episode.*?(\d+).*?reward.*?(-?\d+\.?\d*)', line, re.IGNORECASE)
                    if match:
                        episode = int(match.group(1))
                        reward = float(match.group(2))
                        episodes.append(episode)
                        rewards.append(reward)
    except FileNotFoundError:
        print(f"日志文件未找到: {log_file}")
    
    return episodes, rewards

def plot_reward_curve(rewards, title="奖励曲线", save_path=None):
    """绘制简单的奖励曲线"""
    if not rewards:
        print("没有奖励数据可绘制")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, 'b-', alpha=0.7, linewidth=1)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.grid(True, alpha=0.3)
    
    # 添加平滑线
    if len(rewards) > 10:
        window = min(50, len(rewards) // 4)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2, label=f'平滑(窗口={window})')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"奖励曲线已保存到: {save_path}")
    
    plt.show()

def plot_all_curves():
    """绘制所有可用的奖励曲线"""
    # 查找所有日志文件
    log_files = glob.glob("experiments/logs/*.log")
    
    if not log_files:
        print("未找到日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    # 创建对比图
    plt.figure(figsize=(15, 10))
    
    for i, log_file in enumerate(log_files[:8]):  # 最多显示8个
        episodes, rewards = extract_rewards_from_log(log_file)
        
        if rewards:
            exp_name = os.path.basename(log_file).replace('.log', '')
            plt.plot(rewards, alpha=0.7, label=exp_name)
    
    plt.title('所有实验的奖励曲线对比')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiments/results/all_rewards_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("对比图已保存到: experiments/results/all_rewards_comparison.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="简单奖励曲线生成器")
    parser.add_argument("--log", type=str, help="单个日志文件路径")
    parser.add_argument("--all", action='store_true', help="绘制所有奖励曲线")
    parser.add_argument("--title", type=str, default="奖励曲线", help="图表标题")
    parser.add_argument("--save", type=str, help="保存路径")
    
    args = parser.parse_args()
    
    if args.all:
        plot_all_curves()
    elif args.log:
        episodes, rewards = extract_rewards_from_log(args.log)
        plot_reward_curve(rewards, args.title, args.save)
    else:
        print("用法:")
        print("  python simple_reward_plotter.py --log experiments/logs/some_log.log")
        print("  python simple_reward_plotter.py --all")

if __name__ == "__main__":
    main()