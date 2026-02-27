import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def smooth(scalars, weight=0.6):
    """平滑曲线，让趋势更明显"""
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    log_dir = "lido-run-events"
    output_file = "training_curves.png"
    
    # 查找所有实验目录
    # 结构通常是: lido-run-events/experiment_name/datetime/events.out.tfevents...
    # 或者直接在 lido-run-events/experiment_name/events.out.tfevents...
    
    # 获取所有包含 tfevents 的文件
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
    
    if not event_files:
        print(f"错误：在 {log_dir} 中未找到 TensorBoard 日志文件。请确保训练已运行。")
        return

    plt.figure(figsize=(12, 8))
    
    # 常见 tag 名称（skrl 可能用的名字）
    possible_tags = [
        "Reward / Total reward (mean)",
        "Reward / Mean reward",
        "Reward/Mean",
        "Total reward (mean)"
    ]

    print(f"找到 {len(event_files)} 个日志文件，开始解析...")

    has_data = False
    
    for event_file in event_files:
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # 确定实验名称（取父文件夹名）
            parent_dir = os.path.dirname(event_file)
            # 如果父文件夹是时间戳（常见情况），再往上一级
            if "202" in os.path.basename(parent_dir): # 简单判断是不是日期格式
                exp_name = os.path.basename(os.path.dirname(parent_dir))
            else:
                exp_name = os.path.basename(parent_dir)
            
            # 尝试找到奖励 tag
            found_tag = None
            for tag in possible_tags:
                if tag in ea.Tags()['scalars']:
                    found_tag = tag
                    break
            
            if not found_tag:
                # 打印所有 available tags 供调试
                # print(f"  [{exp_name}] 未找到奖励 tag。可用 tags: {ea.Tags()['scalars']}")
                continue
                
            # 提取数据
            events = ea.Scalars(found_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            if len(values) < 2:
                continue

            # 平滑处理
            smoothed_values = smooth(values, weight=0.8)
            
            # 区分 CDQN 和 DQN 的颜色/线型
            if "cdqn" in exp_name.lower():
                linestyle = '-'
                marker = None
                label_prefix = "CDQN"
            elif "dqn" in exp_name.lower():
                linestyle = '--'
                marker = None
                label_prefix = "DQN"
            else:
                linestyle = '-'
                marker = None
                label_prefix = "Agent"

            # 简化图例名称，提取参数
            # 假设名称格式 agent_s1.0c0.0..._cdqn
            try:
                short_name = exp_name.replace("agent_", "").replace("_cdqn", "").replace("_dqn", "")
            except:
                short_name = exp_name

            label = f"{label_prefix} {short_name}"
            
            plt.plot(steps, smoothed_values, label=label, linestyle=linestyle, alpha=0.7)
            has_data = True
            print(f"  已绘制: {exp_name} (步数: {len(steps)})")

        except Exception as e:
            print(f"  解析失败 {event_file}: {e}")

    if has_data:
        plt.title("Training Reward Curves (Smoothed)")
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"\n成功！图表已保存为: {output_file}")
    else:
        print("\n未找到任何有效的奖励数据，无法绘图。")

if __name__ == "__main__":
    main()
