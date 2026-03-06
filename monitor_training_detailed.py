#!/usr/bin/env python3
"""
【文件角色】：远程训练进度监控工具。
实时监控服务器上的训练进度，显示详细的训练信息。
"""
import argparse
import subprocess
import sys
import time
import re
from datetime import datetime


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


def parse_training_progress(log_content):
    """解析训练日志，提取进度信息"""
    progress_info = {
        'current_episode': None,
        'total_episodes': None,
        'current_step': None,
        'total_steps': None,
        'recent_rewards': [],
        'recent_violations': [],
        'error_count': 0,
        'last_update': None
    }
    
    lines = log_content.strip().split('\n')
    
    for line in lines[-50:]:  # 只分析最后50行
        # 提取episode信息
        episode_match = re.search(r'Episode\s+(\d+)', line, re.IGNORECASE)
        if episode_match:
            progress_info['current_episode'] = int(episode_match.group(1))
        
        # 提取步数信息
        step_match = re.search(r'(\d+)/(\d+)', line)
        if step_match:
            progress_info['current_step'] = int(step_match.group(1))
            progress_info['total_steps'] = int(step_match.group(2))
        
        # 提取奖励信息
        reward_match = re.search(r'reward[:\s=]+([-\d.]+)', line, re.IGNORECASE)
        if reward_match:
            try:
                progress_info['recent_rewards'].append(float(reward_match.group(1)))
                if len(progress_info['recent_rewards']) > 10:
                    progress_info['recent_rewards'].pop(0)
            except ValueError:
                pass
        
        # 提取违规信息
        violation_match = re.search(r'violat', line, re.IGNORECASE)
        if violation_match:
            progress_info['recent_violations'].append(line.strip())
            if len(progress_info['recent_violations']) > 5:
                progress_info['recent_violations'].pop(0)
        
        # 统计错误
        error_match = re.search(r'error|exception|traceback', line, re.IGNORECASE)
        if error_match:
            progress_info['error_count'] += 1
        
        # 记录最后更新时间
        progress_info['last_update'] = line.strip()
    
    return progress_info


def monitor_training_detailed(server_ip, server_port, server_user, project_path):
    """详细监控训练进度"""
    print(f"🔍 详细监控服务器 {server_user}@{server_ip}:{server_port}")
    print(f"📁 项目路径: {project_path}")
    print("=" * 70)
    
    # 1. 检查训练进程
    print("\n📋 1. 训练进程状态")
    print("-" * 50)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && pgrep -f train_all.sh -l'
    )
    
    if code == 0 and stdout.strip():
        print(f"✅ 训练进程正在运行:")
        print(stdout)
        
        # 提取PID
        pid_match = re.search(r'^\s*(\d+)', stdout)
        if pid_match:
            pid = pid_match.group(1)
            
            # 获取进程详细信息
            stdout2, stderr2, code2 = run_ssh_command(
                server_ip, server_port, server_user,
                f'cd "{project_path}" && ps -p {pid} -o pid,ppid,cmd,etime,pcpu,pmem'
            )
            if code2 == 0:
                print("进程详细信息:")
                print(stdout2)
    else:
        print("❌ 未找到训练进程")
    
    # 2. 查看训练日志
    print("\n📋 2. 训练日志分析")
    print("-" * 50)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && ls -t training_*.log 2>/dev/null | head -1'
    )
    
    if code == 0 and stdout.strip():
        log_file = stdout.strip()
        print(f"使用日志文件: {log_file}")
        
        # 获取日志内容
        stdout2, stderr2, code2 = run_ssh_command(
            server_ip, server_port, server_user,
            f'cd "{project_path}" && tail -n 100 {log_file}'
        )
        
        if code2 == 0 and stdout2.strip():
            # 解析进度
            progress = parse_training_progress(stdout2)
            
            # 显示进度信息
            if progress['current_episode']:
                print(f"当前Episode: {progress['current_episode']}")
            if progress['current_step'] and progress['total_steps']:
                percentage = (progress['current_step'] / progress['total_steps']) * 100
                print(f"训练进度: {progress['current_step']}/{progress['total_steps']} ({percentage:.1f}%)")
            
            if progress['recent_rewards']:
                avg_reward = sum(progress['recent_rewards']) / len(progress['recent_rewards'])
                print(f"最近平均奖励: {avg_reward:.2f}")
            
            if progress['recent_violations']:
                print(f"最近违规次数: {len(progress['recent_violations'])}")
            
            if progress['error_count'] > 0:
                print(f"错误计数: {progress['error_count']}")
            
            # 显示最后几行日志
            print("\n最新日志内容:")
            print("-" * 30)
            lines = stdout2.strip().split('\n')
            for line in lines[-10:]:
                print(line)
        else:
            print("无法读取日志内容")
    else:
        print("未找到训练日志文件")
    
    # 3. 检查checkpoint文件
    print("\n📋 3. Checkpoint文件")
    print("-" * 50)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && find . -name "*.pt" -type f -exec ls -lh {{}} \\; 2>/dev/null | sort -k6,7'
    )
    
    if stdout.strip():
        print("找到的checkpoint文件:")
        print(stdout)
    else:
        print("未找到checkpoint文件")
    
    # 4. 检查结果文件
    print("\n📋 4. 结果文件")
    print("-" * 50)
    stdout, stderr, code = run_ssh_command(
        server_ip, server_port, server_user,
        f'cd "{project_path}" && ls -la *.csv 2>/dev/null || echo "未找到结果文件"'
    )
    
    if stdout.strip():
        print("结果文件:")
        print(stdout)
    else:
        print("未找到结果文件")


def continuous_monitor_detailed(server_ip, server_port, server_user, project_path, interval=180):
    """连续详细监控"""
    print(f"🔄 开始连续详细监控，每{interval}秒刷新一次")
    print("按 Ctrl+C 停止监控")
    print("=" * 70)
    
    try:
        while True:
            print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 刷新监控信息")
            monitor_training_detailed(server_ip, server_port, server_user, project_path)
            print(f"\n⏳ 等待{interval}秒后刷新...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="详细监控服务器上的训练进度")
    parser.add_argument("--server-ip", required=True, help="服务器IP地址")
    parser.add_argument("--server-port", default=22, help="SSH端口")
    parser.add_argument("--server-user", required=True, help="服务器用户名")
    parser.add_argument("--project-path", default="/autodl-tmp/petri RL", help="项目路径")
    parser.add_argument("--continuous", action="store_true", help="连续监控模式")
    parser.add_argument("--interval", type=int, default=180, help="连续监控刷新间隔(秒)")
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitor_detailed(
            args.server_ip, args.server_port, 
            args.server_user, args.project_path, 
            args.interval
        )
    else:
        monitor_training_detailed(
            args.server_ip, args.server_port, 
            args.server_user, args.project_path
        )


if __name__ == "__main__":
    main()