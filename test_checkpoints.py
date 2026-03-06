#!/usr/bin/env python3
"""
测试checkpoint保存
"""
import os
import sys
import subprocess

def run_training_with_checkpoints():
    """运行短时间训练以测试checkpoint保存"""
    print("开始测试checkpoint保存...")
    
    # 运行短时间训练
    cmd = [
        "python", "train.py", 
        "--train", 
        "--constrained", 
        "--timesteps", "300",  # 短时间训练
        "--m-success", "1.0",
        "--m-cars-driven", "0.0",
        "--m-waiting-time", "1.0",
        "--m-max-waiting-time", "0.0",
        "--m-timestep", "0.0"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print("训练输出:")
        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("训练超时")
    except Exception as e:
        print(f"训练失败: {e}")

def check_checkpoints():
    """检查checkpoint文件"""
    print("\n检查checkpoint文件...")
    
    # 检查lido-run-events目录
    lido_dir = "lido-run-events"
    if os.path.exists(lido_dir):
        print(f"✓ {lido_dir} 目录存在")
        for root, dirs, files in os.walk(lido_dir):
            for file in files:
                if file.endswith('.pt'):
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    print(f"✓ 找到checkpoint: {file_path} ({size} bytes)")
    else:
        print(f"✗ {lido_dir} 目录不存在")
    
    # 检查当前目录
    current_dir_pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    if current_dir_pt_files:
        print(f"✓ 当前目录找到 {len(current_dir_pt_files)} 个.pt文件:")
        for file in current_dir_pt_files:
            size = os.path.getsize(file)
            print(f"  - {file} ({size} bytes)")
    else:
        print("✗ 当前目录没有.pt文件")

def main():
    """主函数"""
    run_training_with_checkpoints()
    check_checkpoints()

if __name__ == "__main__":
    main()