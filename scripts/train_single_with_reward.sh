#!/bin/bash

# 单个参数组合训练，带奖励曲线
# 用法: bash train_single_with_reward.sh <success> <cars_driven> <waiting_time> <max_waiting_time> <timestep> <algorithm>

if [ $# -ne 6 ]; then
    echo "用法: $0 <success> <cars_driven> <waiting_time> <max_waiting_time> <timestep> <algorithm>"
    echo "示例: $0 1.0 0.0 1.0 1.5 0.0 cdqn"
    exit 1
fi

success=$1
cars_driven=$2
waiting_time=$3
max_waiting_time=$4
timestep=$5
algorithm=$6

# 构建实验名称
exp_name="agent_s${success}c${cars_driven}w${waiting_time}mw${max_waiting_time}t${timestep}_${algorithm}"

# 构建命令
constrained_flag=""
if [ "$algorithm" = "cdqn" ]; then
    constrained_flag="--constrained"
fi

echo ""
echo "===== 开始训练单个参数组合 ====="
echo "奖励函数参数: success=$success, cars_driven=$cars_driven, waiting_time=$waiting_time, max_waiting_time=$max_waiting_time, timestep=$timestep"
echo "算法: $algorithm"
echo "实验名称: $exp_name"
echo ""

# 运行带奖励跟踪的训练
cd scripts && python simple_reward_tracker.py \
  --exp-name "$exp_name" \
  $constrained_flag \
  --m-success "$success" \
  --m-cars-driven "$cars_driven" \
  --m-waiting-time "$waiting_time" \
  --m-max-waiting-time "$max_waiting_time" \
  --m-timestep "$timestep" \
  --timesteps 3000

echo ""
echo "===== 训练完成 ====="
echo "奖励曲线保存在 experiments/results/final_reward_${exp_name}.png"
echo "模型保存在 models/${exp_name}_final.pt"