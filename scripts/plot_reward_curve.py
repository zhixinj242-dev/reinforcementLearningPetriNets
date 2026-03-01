#!/usr/bin/env python3
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
