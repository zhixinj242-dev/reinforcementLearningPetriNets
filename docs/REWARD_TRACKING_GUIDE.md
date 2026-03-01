# 奖励曲线生成指南

## 概述

本指南介绍如何为每个参数组合生成奖励曲线，以便直观地比较不同参数组合的训练效果。

## 新增脚本

### 1. `simple_reward_tracker.py`
**功能**: 带奖励跟踪的训练脚本
- 记录每个episode的奖励变化
- 每500步保存一次奖励曲线
- 训练结束保存最终奖励曲线

**用法**:
```bash
python scripts/simple_reward_tracker.py \
  --exp-name "agent_s1.0c0.0w1.0mw1.5t0.0_cdqn" \
  --constrained \
  --m-success 1.0 \
  --m-cars-driven 0.0 \
  --m-waiting-time 1.0 \
  --m-max-waiting-time 1.5 \
  --m-timestep 0.0 \
  --timesteps 3000
```

### 2. `train_all_with_rewards.sh`
**功能**: 批量训练所有参数组合，并生成奖励曲线
- 训练16个参数组合（8个CDQN + 8个DQN）
- 每个组合生成独立的奖励曲线

**用法**:
```bash
bash scripts/train_all_with_rewards.sh
```

### 3. `train_single_with_reward.sh`
**功能**: 单个参数组合训练，带奖励曲线
- 训练指定的参数组合
- 生成对应的奖励曲线

**用法**:
```bash
bash scripts/train_single_with_reward.sh 1.0 0.0 1.0 1.5 0.0 cdqn
```

### 4. `compare_all_rewards.py`
**功能**: 奖励曲线对比脚本
- 生成所有参数组合的奖励对比图
- 生成统计报告

**用法**:
```bash
python scripts/compare_all_rewards.py
```

## 使用流程

### 完整实验流程

#### 1. 批量训练所有参数组合
```bash
bash scripts/train_all_with_rewards.sh
```

#### 2. 生成奖励对比图
```bash
python scripts/compare_all_rewards.py
```

#### 3. 查看结果
```bash
# 查看奖励对比图
ls experiments/results/reward_comparison.png

# 查看统计报告
cat experiments/results/reward_report.md

# 查看单个奖励曲线
ls experiments/results/final_reward_*.png
```

### 单个参数组合实验

#### 1. 训练单个参数组合
```bash
bash scripts/train_single_with_reward.sh 1.0 0.0 1.0 1.5 0.0 cdqn
```

#### 2. 查看奖励曲线
```bash
ls experiments/results/final_reward_agent_s1.0c0.0w1.0mw1.5t0.0_cdqn.png
```

## 输出文件

### 奖励曲线文件
- `experiments/results/final_reward_{exp_name}.png` - 最终奖励曲线
- `experiments/results/reward_{exp_name}_ep{timestep}.png` - 中间奖励曲线

### 对比文件
- `experiments/results/reward_comparison.png` - 所有参数组合对比图
- `experiments/results/reward_report.md` - 统计报告

### 模型文件
- `models/{exp_name}_final.pt` - 最终训练模型

## 奖励曲线解读

### 奖励曲线特征
- **上升趋势**: 表示学习效果良好
- **波动幅度**: 表示训练稳定性
- **收敛速度**: 表示学习效率
- **最终值**: 表示最终性能

### 参数对比
- **成功权重**: 对任务完成的影响
- **等待时间权重**: 对交通效率的影响
- **算法差异**: CDQN vs DQN 的表现

## 故障排除

### 常见问题

#### 1. 奖励曲线文件未生成
**原因**: 可能是训练过程中出现错误
**解决**: 检查训练日志，确保训练正常完成

#### 2. 奖励曲线显示异常
**原因**: 可能是奖励函数设置问题
**解决**: 检查奖励函数参数是否正确

#### 3. 对比图生成失败
**原因**: 可能是没有找到奖励曲线文件
**解决**: 确保先运行训练脚本生成奖励曲线

### 调试技巧

#### 1. 检查训练进度
```bash
# 查看训练日志
tail -f experiments/logs/*.log

# 查看中间奖励曲线
ls experiments/results/reward_*.png
```

#### 2. 单独测试参数组合
```bash
# 测试单个参数组合
bash scripts/train_single_with_reward.sh 1.0 0.0 1.0 1.5 0.0 cdqn
```

#### 3. 验证奖励函数
```bash
# 检查奖励函数是否正确
python -c "import rewards; print(rewards.constrained_reward)"
```

## 最佳实践

### 1. 参数选择建议
- **成功权重**: 1.0-2.0 范围内效果较好
- **等待时间权重**: 1.0-1.5 范围内平衡性能
- **最大等待时间权重**: 0.0-1.5 范围内控制违规

### 2. 训练监控
- 定期检查中间奖励曲线
- 关注收敛趋势
- 及时调整参数

### 3. 结果分析
- 比较不同参数组合的收敛速度
- 分析最终性能差异
- 选择最佳参数组合

## 扩展功能

### 自定义奖励曲线
可以修改 `simple_reward_tracker.py` 中的 `SimpleRewardTracker` 类来自定义奖励曲线样式和保存频率。

### 添加更多指标
可以扩展奖励跟踪器，记录更多训练指标，如违规次数、平均等待时间等。

### 实时监控
可以添加实时监控功能，在训练过程中实时显示奖励曲线。