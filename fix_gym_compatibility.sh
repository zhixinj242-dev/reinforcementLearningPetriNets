#!/bin/bash

# 临时修复Gym兼容性问题的脚本

echo "正在修复Gym兼容性问题..."

# 1. 卸载可能冲突的gym包
echo "步骤1: 卸载gym包..."
pip uninstall -y gym

# 2. 重新安装gymnasium
echo "步骤2: 重新安装gymnasium..."
pip install gymnasium

# 3. 安装gymnasium兼容包
echo "步骤3: 安装gymnasium兼容包..."
pip install gymnasium[atari,accept-rom-license]

# 4. 创建gym别名
echo "步骤4: 创建gym别名..."
python -c "
import sys
import os
# 创建一个gym模块，指向gymnasium
import gymnasium
sys.modules['gym'] = gymnasium
print('gym别名已创建，指向gymnasium')
"

echo "修复完成！现在可以运行训练了。"
echo ""
echo "运行以下命令开始训练："
echo "python train.py --train --constrained --timesteps 1000"