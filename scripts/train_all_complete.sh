#!/bin/bash

# 批量训练脚本（完整解决版本）

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "当前工作目录: $(pwd)"

# 定义8组不同的奖励函数参数
reward_params_list=(
  "0.0 0.0 1.0 0.0 0.0"
  "0.0 0.0 1.0 1.5 0.0"
  "1.0 0.0 1.0 0.0 0.0"
  "1.0 0.0 1.0 1.5 0.0"
  "1.5 0.0 1.0 0.0 0.0"
  "1.5 0.0 1.0 1.5 0.0"
  "2.0 0.0 1.0 0.0 0.0"
  "2.0 0.0 1.0 1.5 0.0"
)

# 遍历算法类型（CDQN 先，DQN 后）
algorithms=("true" "false")  # 先 true (CDQN)，后 false (DQN)
for constrained in "${algorithms[@]}"; do
  # 遍历所有参数组合
  param_idx=1
  for params in "${reward_params_list[@]}"; do
    # 解析参数
    read -r success cars_driven waiting_time max_waiting_time timestep <<< "$params"
    
    # 构建命令
    constrained_flag=""
    algorithm_name="DQN"
    if [ "$constrained" = "true" ]; then
      constrained_flag="--constrained"
      algorithm_name="CDQN"
    fi
    
    echo ""
    echo "===== 开始运行 $param_idx/8 参数组，算法: $algorithm_name ====="
    echo "奖励函数参数: success=$success, cars_driven=$cars_driven, waiting_time=$waiting_time, max_waiting_time=$max_waiting_time, timestep=$timestep"
    
    # 运行训练（使用完整解决版本）
    python scripts/train_complete.py --train $constrained_flag \
      --m-success "$success" \
      --m-cars-driven "$cars_driven" \
      --m-waiting-time "$waiting_time" \
      --m-max-waiting-time "$max_waiting_time" \
      --m-timestep "$timestep"
    
    param_idx=$((param_idx + 1))
  done
done

echo ""
echo "===== 所有训练完成 ====="