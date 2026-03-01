#!/usr/bin/env python3
"""
最简单的奖励曲线生成器 - 不需要matplotlib
直接输出CSV数据，可以用Excel或其他工具绘图
"""

import os
import sys
import csv

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def save_rewards_to_csv(rewards, filename="rewards.csv"):
    """保存奖励数据到CSV文件"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
        for i, reward in enumerate(rewards):
            writer.writerow([i+1, reward])
    print(f"奖励数据已保存到: {filename}")
    print("可以用Excel打开并绘制图表")

def create_simple_logger():
    """创建简单的奖励记录器"""
    
    logger_code = '''
# 在训练脚本中添加的简单代码
import csv

# 在训练开始前
episode_rewards = []
csv_file = open('training_rewards.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Episode', 'Reward'])

# 在每个episode结束时
episode_rewards.append(total_reward)
writer.writerow([len(episode_rewards), total_reward])
csv_file.flush()

# 训练结束后
csv_file.close()
print("奖励数据已保存到: training_rewards.csv")
'''
    
    print("=== 最简单的奖励记录方法 ===")
    print()
    print("只需要在训练脚本中添加以下代码：")
    print(logger_code)
    print()
    print("然后可以用Excel、Google Sheets或任何绘图工具打开CSV文件绘图。")
    print()
    print("或者使用Python的pandas和matplotlib：")
    print("import pandas as pd")
    print("import matplotlib.pyplot as plt")
    print("df = pd.read_csv('training_rewards.csv')")
    print("plt.plot(df['Episode'], df['Reward'])")
    print("plt.savefig('reward_curve.png')")

if __name__ == "__main__":
    create_simple_logger()
    
    # 示例：生成一些测试数据
    test_rewards = [10, 15, 12, 18, 25, 30, 28, 35, 40, 38, 45, 50]
    save_rewards_to_csv(test_rewards, "example_rewards.csv")