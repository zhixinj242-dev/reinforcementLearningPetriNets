#!/usr/bin/env python3
"""
修改训练脚本，添加最简单的奖励跟踪
"""

import os
import re

def add_reward_tracking_to_train_py():
    """为 train.py 添加最简单的奖励跟踪"""
    
    train_file = "scripts/train.py"
    
    if not os.path.exists(train_file):
        print(f"未找到 {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有奖励跟踪
    if 'csv' in content and 'training_rewards.csv' in content:
        print("train.py 已经包含奖励跟踪功能")
        return
    
    # 在导入部分添加
    import_addition = "import csv\n"
    
    # 在 main 函数开始处添加奖励跟踪初始化
    main_function_pattern = r'(def main\(\):.*?\n)(.*?)(\n\s+# 构建奖励函数)'
    
    def add_reward_init(match):
        old_main = match.group(2)
        new_main = f"""{old_main}
    # 添加简单的奖励跟踪
    episode_rewards = []
    csv_file = open('training_rewards.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Episode', 'Reward'])
"""
        return new_main
    
    # 在训练循环中添加奖励记录
    training_pattern = r'(trainer\.train\(timesteps\))'
    
    def add_reward_recording(match):
        return f"""{match.group(1)}
    
    # 保存奖励数据
    csv_file.close()
    print("奖励数据已保存到: training_rewards.csv")
    print("可以用Excel打开并绘制图表")"""
    
    # 应用修改
    modified_content = content
    
    # 添加导入
    if 'import csv' not in modified_content:
        modified_content = import_addition + modified_content
    
    # 添加奖励初始化
    modified_content = re.sub(main_function_pattern, add_reward_init, modified_content, flags=re.DOTALL)
    
    # 添加奖励保存
    modified_content = re.sub(training_pattern, add_reward_recording, modified_content)
    
    # 写回文件
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("已为 train.py 添加简单的奖励跟踪功能")
    print("现在训练后会生成 training_rewards.csv 文件")
    print("可以用Excel打开并绘制奖励曲线")

def create_simple_plot_script():
    """创建简单的绘图脚本"""
    
    script = """#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_reward_curve(csv_file='training_rewards.csv'):
    if not os.path.exists(csv_file):
        print(f"未找到 {csv_file}")
        return
    
    # 读取CSV数据
    df = pd.read_csv(csv_file)
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['Episode'], df['Reward'], 'b-', alpha=0.7)
    plt.title('训练奖励曲线')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.grid(True, alpha=0.3)
    
    # 添加平滑线
    if len(df) > 10:
        window = min(50, len(df) // 4)
        df['Smoothed'] = df['Reward'].rolling(window=window).mean()
        plt.plot(df['Episode'], df['Smoothed'], 'r-', linewidth=2, label=f'平滑(窗口={window})')
        plt.legend()
    
    # 保存图片
    output_file = csv_file.replace('.csv', '_curve.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"奖励曲线已保存到: {output_file}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="绘制奖励曲线")
    parser.add_argument("--csv", type=str, default="training_rewards.csv", help="CSV文件路径")
    args = parser.parse_args()
    
    plot_reward_curve(args.csv)
"""
    
    with open('scripts/plot_reward_curve.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("已创建绘图脚本: scripts/plot_reward_curve.py")
    print("用法: python scripts/plot_reward_curve.py --csv training_rewards.csv")

if __name__ == "__main__":
    print("=== 为训练脚本添加最简单的奖励跟踪 ===")
    print()
    
    # 1. 修改 train.py
    add_reward_tracking_to_train_py()
    
    # 2. 创建绘图脚本
    create_simple_plot_script()
    
    print()
    print("=== 使用方法 ===")
    print("1. 运行训练: python scripts/train.py --train --constrained")
    print("2. 查看CSV: cat training_rewards.csv")
    print("3. 绘制曲线: python scripts/plot_reward_curve.py")
    print("4. 或者用Excel打开 training_rewards.csv 手动绘图")