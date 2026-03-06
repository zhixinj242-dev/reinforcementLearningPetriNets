#!/usr/bin/env python3
"""
【文件角色】：从现有日志生成奖励曲线的工具。
解析训练日志，提取奖励数据，生成可视化曲线。
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
from datetime import datetime
import pandas as pd


def parse_log_file(log_file):
    """解析单个日志文件，提取奖励数据
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        dict: 包含解析数据的字典
    """
    data = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'violations': []
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 检查是否是JSONL格式（OptimizedViolationLogger生成的）
            if content.strip().startswith('{') and '"step"' in content:
                # 这是违规日志，需要从环境信息中提取奖励
                lines = content.strip().split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            log_entry = json.loads(line)
                            # 从违规日志中提取信息
                            if 'step' in log_entry:
                                data['episodes'].append(log_entry['step'])
                                # 违规日志中没有直接奖励信息，使用违规次数作为指标
                                data['violations'].append(1)
                        except json.JSONDecodeError:
                            continue
            else:
                # 尝试解析标准日志格式
                lines = content.strip().split('\n')
                current_episode = 0
                current_reward = 0
                current_length = 0
                current_violations = 0
                
                for line in lines:
                    if 'Episode' in line and 'reward' in line:
                        # 尝试从日志行提取奖励信息
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'Episode' and i+1 < len(parts):
                                try:
                                    current_episode = int(parts[i+1])
                                except ValueError:
                                    pass
                            elif part == 'reward:' and i+1 < len(parts):
                                try:
                                    current_reward = float(parts[i+1])
                                    data['episodes'].append(current_episode)
                                    data['rewards'].append(current_reward)
                                except ValueError:
                                    pass
                    elif 'episode' in line and 'length' in line:
                        # 尝试提取episode长度
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'episode' and i+1 < len(parts):
                                try:
                                    current_length = int(parts[i+1])
                                    if len(data['lengths']) < len(data['episodes']):
                                        data['lengths'].append(current_length)
                                except ValueError:
                                    pass
                    elif 'violations' in line or '违规' in line:
                        # 尝试提取违规次数
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and int(part) > 0:
                                current_violations = int(part)
                                if len(data['violations']) < len(data['episodes']):
                                    data['violations'].append(current_violations)
                                break
    
    except Exception as e:
        print(f"解析日志文件 {log_file} 时出错: {e}")
    
    return data


def find_log_files(log_dir="detailed_logs"):
    """查找所有日志文件
    
    Args:
        log_dir: 日志目录
        
    Returns:
        list: 日志文件路径列表
    """
    log_files = []
    
    # 查找.log文件
    log_files.extend(glob.glob(os.path.join(log_dir, "*.log")))
    
    # 查找.jsonl文件（OptimizedViolationLogger生成的）
    log_files.extend(glob.glob(os.path.join(log_dir, "*.jsonl")))
    
    # 递归查找子目录中的日志文件
    log_files.extend(glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True))
    log_files.extend(glob.glob(os.path.join(log_dir, "**/*.jsonl"), recursive=True))
    
    return sorted(log_files)


def extract_params_from_filename(filename):
    """从文件名中提取参数
    
    Args:
        filename: 文件名
        
    Returns:
        dict: 参数字典
    """
    params = {}
    basename = os.path.basename(filename)
    
    # 提取算法类型
    if 'CDQN' in basename.upper():
        params['algorithm'] = 'CDQN'
    elif 'DQN' in basename.upper():
        params['algorithm'] = 'DQN'
    
    # 提取奖励参数
    import re
    # 匹配模式: success1.0_cars_driven1.0_waiting_time1.0_max_waiting_time1.0_timestep1.0
    pattern = r'success(\d+\.?\d*)_cars_driven(\d+\.?\d*)_waiting_time(\d+\.?\d*)_max_waiting_time(\d+\.?\d*)_timestep(\d+\.?\d*)'
    match = re.search(pattern, basename)
    if match:
        params['success'] = float(match.group(1))
        params['cars_driven'] = float(match.group(2))
        params['waiting_time'] = float(match.group(3))
        params['max_waiting_time'] = float(match.group(4))
        params['timestep'] = float(match.group(5))
    
    return params


def plot_rewards_comparison(log_files, output_dir="plots"):
    """绘制多个日志文件的奖励曲线对比
    
    Args:
        log_files: 日志文件路径列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('不同参数组合的奖励曲线对比', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for i, log_file in enumerate(log_files):
        data = parse_log_file(log_file)
        params = extract_params_from_filename(log_file)
        
        if not data['episodes']:
            print(f"警告: 无法从 {log_file} 提取有效数据")
            continue
        
        # 创建标签
        algorithm = params.get('algorithm', 'Unknown')
        success = params.get('success', 0)
        waiting_time = params.get('waiting_time', 0)
        label = f"{algorithm} s={success} w={waiting_time}"
        
        color = colors[i % len(colors)]
        
        # 绘制奖励曲线
        if data['rewards']:
            ax1.plot(data['episodes'], data['rewards'], label=label, color=color, alpha=0.7)
        
        # 绘制移动平均
        if data['rewards'] and len(data['rewards']) > 10:
            window_size = min(10, len(data['rewards']))
            moving_avg = np.convolve(data['rewards'], np.ones(window_size)/window_size, mode='valid')
            episodes_avg = data['episodes'][window_size-1:]
            ax2.plot(episodes_avg, moving_avg, label=label, color=color, linewidth=2)
        
        # 绘制episode长度
        if data['lengths']:
            ax3.plot(data['episodes'][:len(data['lengths'])], data['lengths'], label=label, color=color, alpha=0.7)
        
        # 绘制违规次数
        if data['violations']:
            ax4.plot(data['episodes'][:len(data['violations'])], data['violations'], label=label, color=color, alpha=0.7)
    
    # 设置子图标题和标签
    ax1.set_title('原始奖励')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('奖励')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('移动平均奖励')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('平均奖励')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_title('Episode长度')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('步数')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('违规次数')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('违规次数')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    output_file = os.path.join(output_dir, f"rewards_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"奖励曲线对比图已保存到: {output_file}")
    
    # 生成参数对比表
    create_params_table(log_files, output_dir)


def create_params_table(log_files, output_dir):
    """创建参数对比表
    
    Args:
        log_files: 日志文件路径列表
        output_dir: 输出目录
    """
    table_data = []
    
    for log_file in log_files:
        data = parse_log_file(log_file)
        params = extract_params_from_filename(log_file)
        
        if not data['episodes']:
            continue
        
        # 计算统计信息
        stats = {
            'algorithm': params.get('algorithm', 'Unknown'),
            'success': params.get('success', 0),
            'cars_driven': params.get('cars_driven', 0),
            'waiting_time': params.get('waiting_time', 0),
            'max_waiting_time': params.get('max_waiting_time', 0),
            'timestep': params.get('timestep', 0),
            'episodes': len(data['episodes']),
            'avg_reward': np.mean(data['rewards']) if data['rewards'] else 0,
            'max_reward': np.max(data['rewards']) if data['rewards'] else 0,
            'avg_length': np.mean(data['lengths']) if data['lengths'] else 0,
            'total_violations': sum(data['violations']) if data['violations'] else 0,
            'violation_rate': sum(data['violations']) / sum(data['lengths']) if data['violations'] and data['lengths'] else 0
        }
        
        table_data.append(stats)
    
    # 保存为CSV
    if table_data:
        df = pd.DataFrame(table_data)
        csv_file = os.path.join(output_dir, f"params_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_file, index=False)
        print(f"参数对比表已保存到: {csv_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从日志生成奖励曲线")
    parser.add_argument("--log-dir", default="detailed_logs", help="日志目录")
    parser.add_argument("--output-dir", default="plots", help="输出目录")
    parser.add_argument("--pattern", help="文件名模式，用于筛选特定日志")
    
    args = parser.parse_args()
    
    # 查找日志文件
    log_files = find_log_files(args.log_dir)
    
    # 根据模式筛选文件
    if args.pattern:
        log_files = [f for f in log_files if args.pattern in f]
    
    if not log_files:
        print(f"在 {args.log_dir} 中未找到日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件:")
    for f in log_files:
        print(f"  - {f}")
    
    # 绘制奖励曲线对比
    plot_rewards_comparison(log_files, args.output_dir)


if __name__ == "__main__":
    main()