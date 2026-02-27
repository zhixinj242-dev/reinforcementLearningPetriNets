#!/usr/bin/env python3
"""
实时监控训练/评估进度
用法：
  python monitor_progress.py              # 监控训练进度
  python monitor_progress.py --simulation # 监控 simulation.py 的输出
"""
import os
import sys
import time
import glob
import argparse


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_train_status(base_dir, exp_name):
    """获取训练实验状态"""
    exp_dir = os.path.join(base_dir, exp_name)
    
    # 检查目录是否存在
    if not os.path.isdir(exp_dir):
        return "未开始", 0, 0
    
    # 查找 checkpoints（多种可能路径）
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    checkpoints = []
    if os.path.isdir(ckpt_dir):
        checkpoints = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not checkpoints:
        # 容错：检查实验根目录
        checkpoints = glob.glob(os.path.join(exp_dir, "agent_*.pt"))
    
    if not checkpoints:
        return "训练中", 0, 0
    
    # 获取最新 checkpoint
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    filename = os.path.basename(latest_ckpt)
    
    # 解析步数（格式：agent_xxxxx.pt）
    try:
        step = int(filename.replace("agent_", "").replace(".pt", ""))
    except ValueError:
        step = 0
    
    # 检查是否完成（存在最终 .pt）
    final_pt = os.path.join(base_dir, f"{exp_name}.pt")
    if os.path.exists(final_pt):
        status = "✓ 完成"
    else:
        status = "训练中"
    
    return status, step, len(checkpoints)


def monitor_training():
    """监控训练进度"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "lido-run-events")
    
    try:
        while True:
            clear_screen()
            print("=" * 90)
            print("训练进度监控 (每 3 秒刷新)")
            print("=" * 90)
            print(f"{'ID':<4} {'实验名':<45} {'状态':<10} {'步数':<10} {'检查点'}")
            print("-" * 90)
            
            # 动态扫描所有实验
            exp_names = set()
            if os.path.isdir(base_dir):
                for name in os.listdir(base_dir):
                    path = os.path.join(base_dir, name)
                    if os.path.isdir(path) and name.startswith("agent_"):
                        exp_names.add(name)
                    elif os.path.isfile(path) and name.startswith("agent_") and name.endswith(".pt"):
                        exp_names.add(name[:-3])
            
            exp_list = sorted(exp_names)
            
            if not exp_list:
                print("  （暂无实验，请先运行 bash train_all.sh）")
            else:
                for i, exp_name in enumerate(exp_list, 1):
                    status, step, ckpts = get_train_status(base_dir, exp_name)
                    
                    if status == "✓ 完成":
                        color = "\033[92m"
                    elif status == "训练中":
                        color = "\033[93m"
                    else:
                        color = "\033[90m"
                    reset = "\033[0m"
                    
                    display_name = exp_name[:42] + ".." if len(exp_name) > 44 else exp_name
                    print(f"{color}P{i:02d}  {display_name:<45} {status:<10} {step:<10} {ckpts}{reset}")
            
            print("=" * 90)
            print(f"路径: {base_dir}")
            print("按 Ctrl+C 退出")
            
            time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n监控已停止")
        sys.exit(0)


def monitor_simulation():
    """监控 simulation 进度（读取 simulation-results.csv）"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "simulation-results.csv")
    
    try:
        import pandas as pd
        
        while True:
            clear_screen()
            print("=" * 90)
            print("Simulation 评估进度 (每 5 秒刷新)")
            print("=" * 90)
            
            if not os.path.exists(csv_path):
                print("  （simulation-results.csv 不存在，请先运行 python simulation.py）")
            else:
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) == 0:
                        print("  （CSV 为空，评估尚未开始）")
                    else:
                        print(f"{'模型':<10} {'参数(s/c/w/mw/t)':<20} {'平均步数':<12} {'违规率%':<10} {'平均等待':<10}")
                        print("-" * 90)
                        for _, row in df.iterrows():
                            model = row['model_type'].upper()
                            params = f"{row['s']}/{row['c']}/{row['w']}/{row['mw']}/{row['t']}"
                            avg_steps = f"{row['avg_t_frame']:.0f}"
                            avg_broken = f"{row['avg_c_broken']:.2f}"
                            avg_wait = f"{row['avg_waiting_time']:.2f}"
                            print(f"{model:<10} {params:<20} {avg_steps:<12} {avg_broken:<10} {avg_wait:<10}")
                        print("-" * 90)
                        print(f"已完成: {len(df)} / 16 个模型")
                except Exception as e:
                    print(f"  （读取 CSV 出错: {e}）")
            
            print("=" * 90)
            print("按 Ctrl+C 退出")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n监控已停止")
        sys.exit(0)
    except ImportError:
        print("错误：需要 pandas 库。请运行: pip install pandas")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Monitor Progress")
    parser.add_argument("--simulation", action="store_true", help="监控 simulation.py 进度")
    args = parser.parse_args()
    
    if args.simulation:
        monitor_simulation()
    else:
        monitor_training()
