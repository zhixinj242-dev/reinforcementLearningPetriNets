#!/usr/bin/env python3
"""
【文件角色】：监控服务器上的训练进度。
通过SSH连接到服务器，查看训练日志、checkpoint文件和训练状态。
"""
import argparse
import subprocess
import sys
import time
import re


def run_ssh_command(server_ip, server_port, server_user, command):
    """运行SSH命令并返回输出"""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", 
        "-p", str(server_port), 
        f"{server_user}@{server_ip}",
        command
    ]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "命令执行超时", 1
    except Exception as e:
        return "", f"SSH连接错误: {e}", 1


def monitor_training(server_ip, server_port, server_user, project_path):
    """监控训练进度"""
    print(f"🔍 监控服务器 {server_user}@{server_ip}:{server_port}")
    print(f"📁 项目路径: {project_path}")
    print("=" * 60)
    
    # 1. 检查训练进程
    print("\n📋 1. 检查训练进程状态")
    print("-" * 40)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && pgrep -f train_all.sh -l'
    )
    
    if code == 0 and stdout.strip():
        print(f"✅ 训练进程正在运行:")
        print(stdout)
    else:
        print("❌ 未找到训练进程")
    
    # 2. 查看训练日志
    print("\n📋 2. 查看训练日志 (最后30行)")
    print("-" * 40)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && tail -n 30 training.log 2>/dev/null || echo "日志文件不存在"'
    )
    
    if stdout.strip():
        print(stdout)
    else:
        print("无法读取训练日志")
    
    # 3. 检查checkpoint文件
    print("\n📋 3. 检查checkpoint文件")
    print("-" * 40)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && find . -name "*.pt" -type f -exec ls -lh {{}} \\; 2>/dev/null | head -10'
    )
    
    if stdout.strip():
        print("找到的checkpoint文件:")
        print(stdout)
    else:
        print("未找到checkpoint文件")
    
    # 4. 检查训练进度
    print("\n📋 4. 检查训练进度")
    print("-" * 40)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && python monitor_progress.py 2>/dev/null || echo "无法获取进度信息"'
    )
    
    if stdout.strip():
        print(stdout)
    else:
        print("无法获取训练进度")
    
    # 5. 检查结果文件
    print("\n📋 5. 检查结果文件")
    print("-" * 40)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && ls -la *.csv 2>/dev/null || echo "未找到结果文件"'
    )
    
    if stdout.strip():
        print("结果文件:")
        print(stdout)
    else:
        print("未找到结果文件")


def continuous_monitor(server_ip, server_port, server_user, project_path, interval=60):
    """连续监控"""
    print(f"🔄 开始连续监控，每{interval}秒刷新一次")
    print("按 Ctrl+C 停止监控")
    print("=" * 60)
    
    try:
        while True:
            print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')} - 刷新监控信息")
            monitor_training(server_ip, server_port, server_user, project_path)
            print(f"\n⏳ 等待{interval}秒后刷新...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="监控服务器上的训练进度")
    parser.add_argument("--server-ip", required=True, help="服务器IP地址")
    parser.add_argument("--server-port", default=22, help="SSH端口")
    parser.add_argument("--server-user", required=True, help="服务器用户名")
    parser.add_argument("--project-path", default="/autodl-tmp/petri RL", help="项目路径")
    parser.add_argument("--continuous", action="store_true", help="连续监控模式")
    parser.add_argument("--interval", type=int, default=60, help="连续监控刷新间隔(秒)")
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitor(
            args.server_ip, args.server_port, 
            args.server_user, args.project_path, 
            args.interval
        )
    else:
        monitor_training(
            args.server_ip, args.server_port, 
            args.server_user, args.project_path
        )


if __name__ == "__main__":
    main()