#!/usr/bin/env python3
import os
import json
import glob

def analyze_violation_logs():
    """分析违规日志文件"""
    
    # 查找所有违规日志文件
    log_files = glob.glob("experiments/logs/violations_*.log")
    
    if not log_files:
        print("未找到违规日志文件")
        return
    
    print(f"找到 {len(log_files)} 个违规日志文件")
    print()
    
    total_violations = 0
    total_steps = 0
    
    for log_file in log_files:
        print(f"分析: {os.path.basename(log_file)}")
        
        violations = 0
        steps = 0
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('{') and '"violation_type"' in line:
                    violations += 1
                elif '"total_violations"' in line:
                    data = json.loads(line)
                    violations = data.get('total_violations', 0)
                    steps = data.get('total_steps', 0)
        
        print(f"  违规次数: {violations}")
        print(f"  总步数: {steps}")
        
        if steps > 0:
            violation_rate = violations / steps * 100
            print(f"  违规率: {violation_rate:.2f}%")
        
        total_violations += violations
        total_steps += steps
        print()
    
    # 总体统计
    if total_steps > 0:
        overall_rate = total_violations / total_steps * 100
        print("=== 总体统计 ===")
        print(f"总违规次数: {total_violations}")
        print(f"总步数: {total_steps}")
        print(f"总体违规率: {overall_rate:.2f}%")

if __name__ == "__main__":
    analyze_violation_logs()