#!/bin/bash

# 最终工作版本的训练脚本
# 自动设置环境并运行训练

echo "=== 强化学习训练启动脚本（最终工作版本） ==="
echo ""

# 获取当前目录
CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"

# 设置 PYTHONPATH
export PYTHONPATH=$CURRENT_DIR/src:$PYTHONPATH
echo "PYTHONPATH 已设置为: $PYTHONPATH"

# 检查模块是否可以导入
echo "检查模块导入..."
python -c "
import sys
import os

# 获取脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath('train_working.py'))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

try:
    from agents.dqn import get_dqn_model
    print('✓ agents.dqn 导入成功')
except ImportError as e:
    print(f'✗ agents.dqn 导入失败: {e}')

try:
    from environment import JunctionPetriNetEnv
    print('✓ environment 导入成功')
except ImportError as e:
    print(f'✗ environment 导入失败: {e}')

try:
    from utils.parser_pnpro import PNProParser
    print('✓ utils.parser_pnpro 导入成功')
except ImportError as e:
    print(f'✗ utils.parser_pnpro 导入失败: {e}')

try:
    from utils.violation_logger import ViolationLogger
    print('✓ utils.violation_logger 导入成功')
except ImportError as e:
    print(f'✗ utils.violation_logger 导入失败: {e}')

try:
    from utils.entities import PetriPlace, PetriTransition, PetriArc
    print('✓ utils.entities 导入成功')
except ImportError as e:
    print(f'✗ utils.entities 导入失败: {e}')

try:
    import rewards
    print('✓ rewards 导入成功')
    print('✓ 可用的奖励函数:')
    print(f'  - constraint_driven_waiting_times_timesteps')
    print(f'  - driven_waiting_times_timesteps')
    print(f'  - constraint_avg_waiting_times_and_timesteps')
    print(f'  - constraint_timestep')
    print(f'  - timestep')
    print(f'  - constraint_cars_driven_timestep')
    print(f'  - cars_driven_timestep')
    print(f'  - dynamic_reward')
    print(f'  - base_reward')
    print(f'  - discounted_reward')
    print(f'  - reward_without_time')
except ImportError as e:
    print(f'✗ rewards 导入失败: {e}')

# 测试自定义的 get_petri_net 函数
try:
    from train_working import get_petri_net
    print('✓ 自定义 get_petri_net 函数导入成功')
except ImportError as e:
    print(f'✗ 自定义 get_petri_net 函数导入失败: {e}')
"

echo ""
echo "如果所有模块导入成功，可以开始训练"
echo ""

# 定义测试参数
SUCCESS=1.0
CARS_DRIVEN=0.0
WAITING_TIME=1.0
MAX_WAITING_TIME=1.5
TIMESTEP=0.0
ALGORITHM=cdqn

echo "测试参数:"
echo "  success=$SUCCESS"
echo "  cars_driven=$CARS_DRIVEN"
echo "  waiting_time=$WAITING_TIME"
echo "  max_waiting_time=$MAX_WAITING_TIME"
echo "  timestep=$TIMESTEP"
echo "  algorithm=$ALGORITHM"
echo ""

# 构建命令
CONSTRAINED_FLAG=""
if [ "$ALGORITHM" = "cdqn" ]; then
    CONSTRAINED_FLAG="--constrained"
fi

echo "运行命令:"
echo "python scripts/train_working.py --train $CONSTRAINED_FLAG --m-success $SUCCESS --m-cars-driven $CARS_DRIVEN --m-waiting-time $WAITING_TIME --m-max-waiting-time $MAX_WAITING_TIME --m-timestep $TIMESTEP"
echo ""

# 运行训练
python scripts/train_working.py --train $CONSTRAINED_FLAG \
  --m-success $SUCCESS \
  --m-cars-driven $CARS_DRIVEN \
  --m-waiting-time $WAITING_TIME \
  --m-max-waiting-time $MAX_WAITING_TIME \
  --m-timestep $TIMESTEP

echo ""
echo "训练完成！"