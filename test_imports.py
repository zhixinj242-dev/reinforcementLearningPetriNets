#!/usr/bin/env python3
"""
测试重构后的导入路径
"""

import sys
import os

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Python 路径:")
for path in sys.path:
    print(f"  {path}")

print("\n测试导入...")

try:
    from agents import constrained_dqn
    print("✓ 成功导入 agents.constrained_dqn")
except ImportError as e:
    print(f"✗ 导入 agents.constrained_dqn 失败: {e}")

try:
    from environment import petri_net
    print("✓ 成功导入 environment.petri_net")
except ImportError as e:
    print(f"✗ 导入 environment.petri_net 失败: {e}")

try:
    from utils import log_manager
    print("✓ 成功导入 utils.log_manager")
except ImportError as e:
    print(f"✗ 导入 utils.log_manager 失败: {e}")

print("\n重构验证完成！")