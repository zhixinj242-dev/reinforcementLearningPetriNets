#!/usr/bin/env python3
"""
奖励曲线对比脚本
比较所有参数组合的奖励曲线
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

def extract_params_from_filename(filename):
    """从文件名中提取参数"""
    # 匹配格式: agent_sX.XcX.XwX.XmwX.XtX.X_algorithm
    pattern = r'agent_s([\d.]+)c([\d.]+)w([\d.]+)mw([\d.]+)t([\d.]+)_(\w+)'
    match = re.search(pattern, filename)
    
    if match:
        return {
            'success': float(match.group(1)),
            'cars_driven': float(match.group(2)),
            'waiting_time': float(match.group(3)),
            'max_waiting_time': float(match.group(4)),
            'timestep': float(match.group(5)),
            'algorithm': match.group(6)
        }
    return None

def create_reward_comparison():
    """创建奖励曲线对比图"""
    
    # 查找所有最终奖励曲线文件
    reward_files = glob.glob("experiments/results/final_reward_*.png")
    
    if not reward_files:
        print("未找到奖励曲线文件")
        return
    
    print(f"找到 {len(reward_files)} 个奖励曲线文件")
    
    # 解析参数并分类
    cdqn_files = []
    dqn_files = []
    
    for file in reward_files:
        params = extract_params_from_filename(os.path.basename(file))
        if params:
            if params['algorithm'] == 'cdqn':
                cdqn_files.append((file, params))
            else:
                dqn_files.append((file, params))
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('不同参数组合的奖励曲线对比', fontsize=16)
    
    # 1. CDQN 最好的4个奖励曲线
    cdqn_files.sort(key=lambda x: x[1]['success'], reverse=True)
    for i, (file, params) in enumerate(cdqn_files[:4]):
        try:
            img = Image.open(file)
            axes[0, 0].imshow(img)
            axes[0, 0].set_title(f'CDQN: s={params["success"]}, w={params["waiting_time"]}')
            axes[0, 0].axis('off')
            break
        except:
            continue
    
    # 2. DQN 最好的4个奖励曲线
    dqn_files.sort(key=lambda x: x[1]['success'], reverse=True)
    for i, (file, params) in enumerate(dqn_files[:4]):
        try:
            img = Image.open(file)
            axes[0, 1].imshow(img)
            axes[0, 1].set_title(f'DQN: s={params["success"]}, w={params["waiting_time"]}')
            axes[0, 1].axis('off')
            break
        except:
            continue
    
    # 3. 参数对比图
    cdqn_success = [p[1]['success'] for p in cdqn_files]
    dqn_success = [p[1]['success'] for p in dqn_files]
    
    axes[1, 0].scatter(range(len(cdqn_success)), cdqn_success, alpha=0.7, label='CDQN')
    axes[1, 0].scatter(range(len(dqn_success)), dqn_success, alpha=0.7, label='DQN')
    axes[1, 0].set_title('成功参数对比')
    axes[1, 0].set_xlabel('实验编号')
    axes[1, 0].set_ylabel('成功权重')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计信息
    avg_cdqn = np.mean(cdqn_success) if cdqn_success else 0
    avg_dqn = np.mean(dqn_success) if dqn_success else 0
    
    algorithms = ['CDQN', 'DQN']
    averages = [avg_cdqn, avg_dqn]
    
    axes[1, 1].bar(algorithms, averages, color=['blue', 'orange'], alpha=0.7)
    axes[1, 1].set_title('平均成功权重对比')
    axes[1, 1].set_ylabel('平均成功权重')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(averages):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiments/results/reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("奖励对比图已保存到: experiments/results/reward_comparison.png")
    
    # 生成统计报告
    generate_reward_report(cdqn_files, dqn_files)


def generate_reward_report(cdqn_files, dqn_files):
    """生成奖励统计报告"""
    
    report = []
    report.append("# 奖励曲线统计报告\n")
    
    # CDQN 统计
    report.append("## CDQN 统计")
    if cdqn_files:
        cdqn_success = [p[1]['success'] for p in cdqn_files]
        cdqn_waiting = [p[1]['waiting_time'] for p in cdqn_files]
        
        report.append(f"- 实验数量: {len(cdqn_files)}")
        report.append(f"- 平均成功权重: {np.mean(cdqn_success):.3f}")
        report.append(f"- 平均等待时间权重: {np.mean(cdqn_waiting):.3f}")
        report.append(f"- 最大成功权重: {np.max(cdqn_success):.3f}")
        report.append(f"- 最小成功权重: {np.min(cdqn_success):.3f}")
        
        # 找到最好的CDQN参数组合
        best_cdqn = max(cdqn_files, key=lambda x: x[1]['success'])
        report.append(f"- 最佳参数组合: {best_cdqn[1]}")
    else:
        report.append("- 无CDQN实验数据")
    
    report.append("")
    
    # DQN 统计
    report.append("## DQN 统计")
    if dqn_files:
        dqn_success = [p[1]['success'] for p in dqn_files]
        dqn_waiting = [p[1]['waiting_time'] for p in dqn_files]
        
        report.append(f"- 实验数量: {len(dqn_files)}")
        report.append(f"- 平均成功权重: {np.mean(dqn_success):.3f}")
        report.append(f"- 平均等待时间权重: {np.mean(dqn_waiting):.3f}")
        report.append(f"- 最大成功权重: {np.max(dqn_success):.3f}")
        report.append(f"- 最小成功权重: {np.min(dqn_success):.3f}")
        
        # 找到最好的DQN参数组合
        best_dqn = max(dqn_files, key=lambda x: x[1]['success'])
        report.append(f"- 最佳参数组合: {best_dqn[1]}")
    else:
        report.append("- 无DQN实验数据")
    
    report.append("")
    
    # 算法对比
    report.append("## 算法对比")
    if cdqn_files and dqn_files:
        cdqn_avg = np.mean([p[1]['success'] for p in cdqn_files])
        dqn_avg = np.mean([p[1]['success'] for p in dqn_files])
        
        report.append(f"- CDQN 平均成功权重: {cdqn_avg:.3f}")
        report.append(f"- DQN 平均成功权重: {dqn_avg:.3f}")
        
        if cdqn_avg > dqn_avg:
            report.append("- **CDQN 在此任务中表现更好**")
        elif dqn_avg > cdqn_avg:
            report.append("- **DQN 在此任务中表现更好**")
        else:
            report.append("- 两种算法表现相当")
    
    # 保存报告
    with open('experiments/results/reward_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("奖励统计报告已保存到: experiments/results/reward_report.md")


def main():
    print("开始生成奖励曲线对比...")
    
    # 确保结果目录存在
    os.makedirs("experiments/results", exist_ok=True)
    
    # 创建对比图
    create_reward_comparison()
    
    print("奖励曲线对比完成！")
    print("生成的文件:")
    print("- experiments/results/reward_comparison.png")
    print("- experiments/results/reward_report.md")


if __name__ == "__main__":
    main()