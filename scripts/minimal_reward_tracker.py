#!/usr/bin/env python3
"""
最简单的奖励曲线生成器
直接修改训练脚本，添加几行代码即可
"""

import os
import sys
import matplotlib.pyplot as plt

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def add_simple_reward_tracking():
    """为训练脚本添加最简单的奖励跟踪"""
    
    print("=== 最简单的奖励曲线实现 ===")
    print()
    print("只需要在训练脚本中添加以下几行代码：")
    print()
    print("# 在训练脚本开头添加")
    print("import matplotlib.pyplot as plt")
    print("episode_rewards = []")
    print()
    print("# 在训练循环中添加")
    print("if episode_done:")
    print("    episode_rewards.append(total_reward)")
    print("    if len(episode_rewards) % 100 == 0:")
    print("        plt.plot(episode_rewards)")
    print("        plt.savefig(f'reward_curve_{len(episode_rewards)}.png')")
    print("        plt.close()")
    print()
    print("# 训练结束后添加")
    print("plt.plot(episode_rewards)")
    print("plt.savefig('final_reward_curve.png')")
    print("plt.close()")
    print()
    print("就这么简单！不需要复杂的类和函数。")
    print()
    print("或者使用现成的简单脚本：")
    print("python scripts/simple_reward_plotter.py --all")

def create_ultra_simple_plotter():
    """创建极简版绘图脚本"""
    
    script = """#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os

def plot_rewards(data, title="奖励曲线"):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.grid(True)
    plt.savefig('reward_curve.png')
    print("奖励曲线已保存到: reward_curve.png")

# 示例数据
if __name__ == "__main__":
    # 这里替换成你的实际奖励数据
    rewards = [10, 15, 12, 18, 25, 30, 28, 35, 40, 38, 45, 50]
    plot_rewards(rewards)
"""
    
    with open('scripts/ultra_simple_plotter.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("已创建极简版绘图脚本: scripts/ultra_simple_plotter.py")
    print("用法:")
    print("1. 修改脚本中的 rewards 列表为你的实际数据")
    print("2. 运行 python scripts/ultra_simple_plotter.py")

if __name__ == "__main__":
    add_simple_reward_tracking()
    create_ultra_simple_plotter()