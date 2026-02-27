import torch
from environment import JunctionPetriNetEnv
import rewards
from utils.petri_net import get_petri_net, Parser
import numpy as np
import pandas as pd
from datetime import datetime
import os

"""
【文件角色】：基准对照组（Baseline）。
它不使用 AI，而是使用"固定序列"或"随机动作"来控制红绿灯。
【存在意义】：为了证明你的 AI 到底有多聪明。你需要先看看"笨办法"能拿多少分，才能体现出 AI 拿的高分有价值。
"""

import argparse

def generate_parsed_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(prog="Baseline-Test")
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="是否保存帧图片用于合成视频")
    parser.add_argument("--render-steps", type=int, default=50,
                        help="演示运行的步数（默认50）")
                        
    # --- 奖励函数参数（与 train.py 保持一致）---
    parser.add_argument("--reward-function", type=str, default="discounted_reward")
    parser.add_argument("--m-success", type=float, default=1.0)
    parser.add_argument("--m-cars-driven", type=float, default=1.0)
    parser.add_argument("--m-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-max-waiting-time", type=float, default=1.0)
    parser.add_argument("--m-timestep", type=float, default=1.0)

    return parser.parse_args()

def run_baseline(env, baseline_name, action_sequence, iterations=100, render_first=True, render_steps=50, save_frames=False):
    """
    运行一个基线策略并返回统计结果
    
    Args:
        save_frames: 是否保存帧图片（用于合成视频）
    """
    print(f"\n{'='*60}")
    print(f"正在测试 {baseline_name}")
    print(f"{'='*60}")
    
    # 第一轮：单次演示
    if render_first:
        print(f"--- {baseline_name} 开始演示（运行 {render_steps} 步） ---")
        
        # 如果需要保存帧，创建目录并导入必要的库
        frame_dir = None
        if save_frames:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            frame_dir = f"baseline_frames_{baseline_name}"
            os.makedirs(frame_dir, exist_ok=True)
            print(f"帧图片将保存到: {frame_dir}/")
        
        env.reset()
        t = 0
        terminated = False
        
        print(f"\n{'步骤':<6} {'动作':<20} {'奖励':<10} {'成功':<8} {'通行车辆':<10} {'总等待':<10}")
        print("-" * 80)
        
        while not terminated and t < render_steps:
            action = action_sequence[t % len(action_sequence)]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 打印详细信息（文本输出）
            print(f"{t+1:<6} {env.actions_to_transitions[action]:<20} {reward:>8.1f}  "
                  f"{'✓' if info['success'] else '✗':<8} {info['num_cars_driven']:<10} "
                  f"{sum(info['waiting_times']):<10.1f}")
            
            # 保存帧图片
            if save_frames and frame_dir:
                save_traffic_frame(env, obs, t, frame_dir, baseline_name, 
                                 env.actions_to_transitions[action], reward)
            
            t = t + 1
        
        print(f"\n{baseline_name} 演示完成，运行了 {t} 步")
        if save_frames:
            print(f"\n帧图片已保存到: {frame_dir}/")
            print(f"合成视频命令：")
            print(f"  ffmpeg -framerate 2 -i {frame_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {baseline_name}.mp4\n")
    
    
    # 大规模统计测试
    print(f"--- {baseline_name} 开始 {iterations} 轮统计评估 ---")
    total_timesteps = []
    constraints_broken = []
    average_waiting_times = []
    min_waiting_times = []
    max_waiting_times = []
    
    for i in range(iterations):
        env.reset()
        c = 0  # 违规计数
        all_waiting_times = []  # 与 simulation 一致：本步通过的 + 结束时仍在队列的
        t = 0
        terminated = False

        while not terminated:
            action = action_sequence[t % len(action_sequence)]
            obs, reward, terminated, _, inf = env.step(action)

            if not inf["success"]:  # 如果尝试了非法切换（Petri 网拦截）
                c = c + 1
            all_waiting_times.extend(inf["waiting_times"])
            t = t + 1

        # 与 simulation 一致：结束时仍在队列的车，用其 time_steps 作为等待时间
        for lane in env.lanes:
            for vehicle in lane.vehicles:
                all_waiting_times.append(vehicle.time_steps)

        total_timesteps.append(t)
        constraints_broken.append(c)
        if len(all_waiting_times) > 0:
            average_waiting_times.append(sum(all_waiting_times) / len(all_waiting_times))
            min_waiting_times.append(min(all_waiting_times))
            max_waiting_times.append(max(all_waiting_times))
        else:
            average_waiting_times.append(0)
            min_waiting_times.append(0)
            max_waiting_times.append(0)
        
        # 每 10 轮打印一次进度
        if (i + 1) % 10 == 0:
            print(f"  {baseline_name}: 第 {i+1}/{iterations} 轮完成")
    
    # 计算并返回统计结果（与 simulation 统一：违规用百分比，便于合并到一张表对比）
    avg_t = sum(total_timesteps) / iterations
    max_steps = max(total_timesteps) if total_timesteps else 1
    avg_c_raw = sum(constraints_broken) / len(constraints_broken) if constraints_broken else 0
    avg_c_pct = avg_c_raw / max_steps * 100  # 违规率 %，与 simulation 一致

    results = {
        "name": baseline_name,
        "avg_timesteps": avg_t,
        "avg_constraints_broken": avg_c_pct,
        "avg_waiting_time": sum(average_waiting_times) / len(average_waiting_times) if average_waiting_times else 0,
        "min_waiting_time": sum(min_waiting_times) / len(min_waiting_times) if min_waiting_times else 0,
        "max_waiting_time": sum(max_waiting_times) / len(max_waiting_times) if max_waiting_times else 0,
    }

    return results


def save_traffic_frame(env, obs, step, output_dir, baseline_name, action_name, reward):
    """保存交通状态为图片帧"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # 使用偶数尺寸（libx264 要求宽高必须是偶数）
    fig, ax = plt.subplots(figsize=(10, 8))  # 10*80=800, 8*80=640 都是偶数
    
    # 绘制十字路口
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 绘制道路
    road_color = '#555555'
    road_width = 0.6
    # 南北向道路
    ax.add_patch(patches.Rectangle((-road_width/2, -2), road_width, 4, 
                                   facecolor=road_color, edgecolor='white'))
    # 东西向道路
    ax.add_patch(patches.Rectangle((-2, -road_width/2), 4, road_width, 
                                   facecolor=road_color, edgecolor='white'))
    
    # 提取车辆信息
    lanes_info = {
        'north': {'n': obs.get('vehicle_obs-north_front-n', [0])[0], 
                  't': obs.get('vehicle_obs-north_front-t', [0])[0]},
        'south': {'n': obs.get('vehicle_obs-south_front-n', [0])[0], 
                  't': obs.get('vehicle_obs-south_front-t', [0])[0]},
        'west': {'n': obs.get('vehicle_obs-west_front-n', [0])[0], 
                 't': obs.get('vehicle_obs-west_front-t', [0])[0]},
        'east': {'n': obs.get('vehicle_obs-east_front-n', [0])[0], 
                 't': obs.get('vehicle_obs-east_front-t', [0])[0]},
    }
    
    # 绘制车辆队列（简化为数字显示）
    text_size = 14
    ax.text(0, 1.5, f"N: {lanes_info['north']['n']}辆\n等待:{lanes_info['north']['t']}", 
           ha='center', va='center', fontsize=text_size, 
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(0, -1.5, f"S: {lanes_info['south']['n']}辆\n等待:{lanes_info['south']['t']}", 
           ha='center', va='center', fontsize=text_size,
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(-1.5, 0, f"W: {lanes_info['west']['n']}辆\n等待:{lanes_info['west']['t']}", 
           ha='center', va='center', fontsize=text_size,
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(1.5, 0, f"E: {lanes_info['east']['n']}辆\n等待:{lanes_info['east']['t']}", 
           ha='center', va='center', fontsize=text_size,
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # 显示信息
    info_text = f"{baseline_name} - 步骤 {step+1}\n动作: {action_name}\n奖励: {reward:.1f}"
    ax.text(0, -1.9, info_text, ha='center', va='top', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 保存（使用固定 DPI=100 确保尺寸是偶数：10*100=1000, 8*100=800）
    plt.savefig(f"{output_dir}/frame_{step:04d}.png", dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    args = generate_parsed_arguments()
    
    # 1. 配置奖励函数（逻辑与 train.py 一致）
    reward_fn_name = args.reward_function
    reward_fn = getattr(rewards, reward_fn_name, rewards.discounted_reward)
    
    if reward_fn_name == "dynamic_reward":
        rewards.success_multiplier = args.m_success
        rewards.car_driven_multiplier = args.m_cars_driven
        rewards.waiting_time_multiplier = args.m_waiting_time
        rewards.max_waiting_time_multiplier = args.m_max_waiting_time
        rewards.timestep_multiplier = args.m_timestep
        print(f"Using dynamic_reward: s={args.m_success}, w={args.m_waiting_time}, mw={args.m_max_waiting_time}")

    # 2. 创建环境
    env = JunctionPetriNetEnv(reward_function=reward_fn,
                              net=get_petri_net('data/traffic-scenario.PNPRO', type=Parser.PNPRO), 
                              transitions_to_obs=True, places_to_obs=False)
    env.reset()
    
    # === Baseline B1（根据图片公式）===
    # B1 = [R1oG1n, R2oG2n, R1oG1n+s, R2oG2n+s, R1oG1s+n, R2oG2s+n, ...]
    action_sequence_names_b1 = [
        "RtoGwnes", "GtoRwnes", "RtoGswne", "GtoRswne",
        "RtoGsn", "GtoRsn", "RtoGwe", "GtoRwe"
    ]
    
    # === Baseline B2（根据图片公式）===
    # B2 在第1、2个动作上加了 None（什么都不做）
    action_sequence_names_b2 = [
        "RtoGwnes", "GtoRwnes", "RtoGswne", "GtoRswne",
        "RtoGsn", "None", "GtoRsn", "RtoGwe",
        "None", "GtoRwe"
    ]
    
    # 将变迁名称转换为动作索引
    transitions_to_actions = {name: idx for idx, name in enumerate(env.actions_to_transitions)}
    action_sequence_b1 = [transitions_to_actions[name] for name in action_sequence_names_b1]
    action_sequence_b2 = [transitions_to_actions[name] for name in action_sequence_names_b2]
    
    print("\n" + "="*60)
    print("基线测试：B1 和 B2 对比")
    print("="*60)
    print(f"\nB1 动作序列: {action_sequence_names_b1}")
    print(f"B2 动作序列: {action_sequence_names_b2}")
    
    # 运行两个基线测试
    suffix = ""
    if reward_fn_name == "dynamic_reward":
        suffix = f"_s{args.m_success}c{args.m_cars_driven}w{args.m_waiting_time}mw{args.m_max_waiting_time}t{args.m_timestep}"

    name_b1 = f"B1{suffix}"
    name_b2 = f"B2{suffix}"

    results_b1 = run_baseline(env, name_b1, action_sequence_b1, iterations=100, render_first=True, 
                              render_steps=args.render_steps, save_frames=args.save_frames)
    results_b2 = run_baseline(env, name_b2, action_sequence_b2, iterations=100, render_first=True, 
                              render_steps=args.render_steps, save_frames=args.save_frames)
    
    # 打印对比结果
    print("\n" + "="*60)
    print("基准对照测试对比报告")
    print("="*60)
    print(f"{'指标':<25} {'B1':<15} {'B2':<15} {'差异':<15}")
    print("-" * 60)
    
    def print_comparison(metric_name, b1_val, b2_val, fmt=".2f"):
        diff = b2_val - b1_val
        diff_str = f"{diff:+{fmt}}" if isinstance(diff, (int, float)) else "N/A"
        print(f"{metric_name:<25} {b1_val:<15{fmt}} {b2_val:<15{fmt}} {diff_str:<15}")
    
    print_comparison("平均生存步数", results_b1["avg_timesteps"], results_b2["avg_timesteps"])
    print_comparison("平均违规率(%)", results_b1["avg_constraints_broken"], results_b2["avg_constraints_broken"])
    print_comparison("平均车辆等待时间", results_b1["avg_waiting_time"], results_b2["avg_waiting_time"])
    print_comparison("平均最小等待时间", results_b1["min_waiting_time"], results_b2["min_waiting_time"])
    print_comparison("平均最大等待时间", results_b1["max_waiting_time"], results_b2["max_waiting_time"])
    
    print("="*60)
    
    # 判断哪个更好
    if results_b1["avg_timesteps"] > results_b2["avg_timesteps"]:
        print("\n结论：B1 的生存时间更长（性能更好）")
    elif results_b1["avg_timesteps"] < results_b2["avg_timesteps"]:
        print("\n结论：B2 的生存时间更长（性能更好）")
    else:
        print("\n结论：B1 和 B2 性能相当")
    
    # 写入统一对比文件（与 GAIL、CDQN 等共用 method_comparison.csv）
    from utils.result_comparison import update_methods
    update_methods({name_b1: results_b1, name_b2: results_b2})


if __name__ == "__main__":
    main()
