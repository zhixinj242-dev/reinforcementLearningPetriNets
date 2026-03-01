#!/usr/bin/env python3
"""
修改训练脚本，使用简化的违规日志记录器
"""

import os
import re

def modify_train_to_use_violation_logger():
    """修改 train.py 使用违规日志记录器"""
    
    train_file = "scripts/train.py"
    
    if not os.path.exists(train_file):
        print(f"未找到 {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修改
    if 'ViolationLogger' in content:
        print("train.py 已经使用违规日志记录器")
        return
    
    # 替换 LogManager 导入为 ViolationLogger
    content = content.replace(
        'from utils.log_manager import LogManager',
        'from utils.violation_logger import ViolationLogger'
    )
    
    # 替换 LogManager 实例化为 ViolationLogger
    content = content.replace(
        'log_manager = LogManager(algorithm_name, reward_params, log_dir="experiments/logs")',
        'violation_logger = ViolationLogger(algorithm_name, reward_params, log_dir="experiments/logs")'
    )
    
    # 写回文件
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("已修改 train.py 使用违规日志记录器")
    print("现在只会记录违规动作，大大减少日志量")

def create_violation_tracking_example():
    """创建违规跟踪示例"""
    
    example = """#!/usr/bin/env python3
# 违规跟踪使用示例

from utils.violation_logger import ViolationLogger

# 创建违规日志记录器
violation_logger = ViolationLogger("CDQN", {
    'success': 1.0,
    'waiting_time': 1.0,
    'max_waiting_time': 1.5
})

# 在训练循环中使用
for episode in range(num_episodes):
    episode_violations = 0
    
    for step in range(max_steps):
        # 执行动作
        action, reward, done, info = env.step(action)
        
        # 检查是否有违规
        if info.get('violation', False):
            episode_violations += 1
            violation_logger.log_violation(step, episode, {
                'violation_type': info.get('violation_type', 'unknown'),
                'action': action,
                'state': info.get('state', 'unknown'),
                'reason': info.get('violation_reason', 'unknown')
            })
        
        if done:
            break
    
    # 记录episode结束
    violation_logger.log_episode_end(episode, total_reward, step, episode_violations)

# 关闭日志记录器
violation_logger.close()
"""
    
    with open('scripts/violation_tracking_example.py', 'w', encoding='utf-8') as f:
        f.write(example)
    
    print("已创建违规跟踪示例: scripts/violation_tracking_example.py")

if __name__ == "__main__":
    print("=== 简化日志记录 - 只记录违规动作 ===")
    print()
    
    # 1. 修改训练脚本
    modify_train_to_use_violation_logger()
    
    # 2. 创建示例
    create_violation_tracking_example()
    
    print()
    print("=== 使用方法 ===")
    print("1. 运行训练: python scripts/train.py --train --constrained")
    print("2. 查看违规日志: ls experiments/logs/violations_*.log")
    print("3. 分析违规统计: python scripts/analyze_violations.py")
    print()
    print("现在日志文件会小很多，只包含违规相关信息！")