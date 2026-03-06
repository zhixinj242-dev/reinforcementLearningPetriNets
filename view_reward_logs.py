#!/usr/bin/env python3
"""
【文件角色】：RewardLogger生成文件说明和查看工具。
解释奖励日志中各个文件的作用，并提供查看方法。
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def explain_reward_logs():
    """解释RewardLogger生成的各种文件"""
    print("=" * 60)
    print("RewardLogger 生成的文件说明")
    print("=" * 60)
    
    print("\n1. 奖励日志文件 (.jsonl)")
    print("   - 文件名格式: {algorithm}_{params}_{timestamp}_rewards.jsonl")
    print("   - 内容: 每行一个JSON对象，记录每步的奖励信息")
    print("   - 用途: 详细记录训练过程中的每一步奖励")
    print("   - 示例内容:")
    print("     {\"step\": 1, \"reward\": 5.0, \"timestamp\": \"2023-03-06 13:39:35\", \"environment_acceptance\": true}")
    print("     {\"episode\": 1, \"total_reward\": 45.0, \"length\": 30, \"violations\": 2, \"moving_avg\": 42.3}")
    
    print("\n2. 奖励曲线图 (.png)")
    print("   - 文件名格式: {algorithm}_{params}_{timestamp}_rewards_plot.png")
    print("   - 内容: 包含4个子图的奖励曲线")
    print("   - 子图1: 原始奖励 - 每个episode的总奖励")
    print("   - 子图2: 滑动平均 - {window_size}步滑动平均奖励")
    print("   - 子图3: Episode长度 - 每个episode的步数")
    print("   - 子图4: 违规次数 - 每个episode的违规次数")
    
    print("\n3. CSV数据文件 (.csv)")
    print("   - 文件名格式: {algorithm}_{params}_{timestamp}_rewards_data.csv")
    print("   - 内容: 表格形式的episode数据")
    print("   - 列: episode, reward, length, violations")
    print("   - 用途: 方便在Excel或其他工具中分析")
    
    print("\n4. 统计信息文件 (.json)")
    print("   - 文件名格式: {algorithm}_{params}_{timestamp}_stats.json")
    print("   - 内容: 训练完成的统计摘要")
    print("   - 包含: episodes, avg_reward, max_reward, min_reward, avg_length, total_violations, violation_rate")
    
    print("\n" + "=" * 60)


def view_reward_log_files(log_dir="reward_logs"):
    """查看reward_logs目录中的文件"""
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在")
        return
    
    print(f"\n{log_dir} 目录中的文件:")
    print("-" * 40)
    
    files = os.listdir(log_dir)
    for file in sorted(files):
        file_path = os.path.join(log_dir, file)
        size = os.path.getsize(file_path)
        print(f"{file} ({size} bytes)")
    
    print("-" * 40)
    
    # 查找最新的奖励日志文件
    jsonl_files = [f for f in files if f.endswith('_rewards.jsonl')]
    if jsonl_files:
        latest_file = sorted(jsonl_files)[-1]
        print(f"\n最新奖励日志文件: {latest_file}")
        
        # 显示前几行内容
        file_path = os.path.join(log_dir, latest_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:5]  # 只显示前5行
        
        print("\n前5行内容:")
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                print(f"  {i+1}: {json.dumps(data, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError:
                print(f"  {i+1}: {line.strip()}")
    
    # 查找最新的统计文件
    stats_files = [f for f in files if f.endswith('_stats.json')]
    if stats_files:
        latest_stats = sorted(stats_files)[-1]
        print(f"\n最新统计文件: {latest_stats}")
        
        file_path = os.path.join(log_dir, latest_stats)
        with open(file_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print("\n统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def view_csv_data(log_dir="reward_logs"):
    """查看CSV数据文件"""
    csv_files = [f for f in os.listdir(log_dir) if f.endswith('_rewards_data.csv')]
    if not csv_files:
        print("未找到CSV数据文件")
        return
    
    latest_csv = sorted(csv_files)[-1]
    file_path = os.path.join(log_dir, latest_csv)
    
    print(f"\nCSV数据文件: {latest_csv}")
    print("-" * 40)
    
    try:
        df = pd.read_csv(file_path)
        print(f"数据形状: {df.shape}")
        print("\n前5行:")
        print(df.head())
        
        print("\n基本统计:")
        print(df.describe())
    except Exception as e:
        print(f"读取CSV文件出错: {e}")


def show_plot_files(log_dir="reward_logs"):
    """显示生成的图像文件"""
    plot_files = [f for f in os.listdir(log_dir) if f.endswith('_rewards_plot.png')]
    if not plot_files:
        print("未找到奖励曲线图文件")
        return
    
    print(f"\n奖励曲线图文件:")
    print("-" * 40)
    for file in sorted(plot_files):
        file_path = os.path.join(log_dir, file)
        print(f"{file}")
        
        # 尝试显示图像
        try:
            img = plt.imread(file_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(file)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"显示图像出错: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="查看RewardLogger生成的文件")
    parser.add_argument("--log-dir", default="reward_logs", help="日志目录")
    parser.add_argument("--explain", action="store_true", help="解释文件格式")
    parser.add_argument("--view", action="store_true", help="查看文件内容")
    parser.add_argument("--csv", action="store_true", help="查看CSV数据")
    parser.add_argument("--plot", action="store_true", help="显示图像")
    
    args = parser.parse_args()
    
    if args.explain:
        explain_reward_logs()
    
    if args.view:
        view_reward_log_files(args.log_dir)
    
    if args.csv:
        view_csv_data(args.log_dir)
    
    if args.plot:
        show_plot_files(args.log_dir)
    
    if not any([args.explain, args.view, args.csv, args.plot]):
        # 默认显示所有信息
        explain_reward_logs()
        view_reward_log_files(args.log_dir)


if __name__ == "__main__":
    main()