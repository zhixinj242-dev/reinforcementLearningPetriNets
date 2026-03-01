# 重构后的项目运行流程

## 环境准备

### 1. 设置 Python 路径
```bash
# Windows (PowerShell)
$env:PYTHONPATH = "$env:PYTHONPATH;$($PSScriptRoot)\src"

# Windows (CMD)
set PYTHONPATH=%CD%\src;%PYTHONPATH%

# Linux/Mac
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

## 训练流程

### 单个训练
```bash
# 方法1：使用 Makefile（推荐）
make train

# 方法2：直接运行
cd scripts && python train.py --train --constrained --m-success 1.0 --m-cars-driven 0.0 --m-waiting-time 1.0 --m-max-waiting-time 0.0 --m-timestep 0.0
```

### 批量训练
```bash
# 方法1：使用 Makefile（推荐）
make train-all

# 方法2：直接运行
cd scripts && bash train_all.sh
```

## 评估流程

### 评估最优模型
```bash
# 方法1：使用 Makefile（推荐）
make eval-all

# 方法2：直接运行
cd scripts && bash eval_best_all.sh
```

### 单个模型评估
```bash
# 评估 CDQN 模型
cd scripts && python add_to_comparison.py --method CDQN --path models/agent_s1.5c0.0w1.0mw1.5t0.0_cdqn.pt

# 评估 DQN 模型
cd scripts && python add_to_comparison.py --method DQN --path models/agent_s1.5c0.0w1.0mw1.5t0.0_dqn.pt
```

## 仿真测试流程

### 批量仿真
```bash
# 方法1：使用 Makefile（推荐）
make simulate

# 方法2：直接运行
cd scripts && python simulation.py
```

### 基准测试
```bash
# 方法1：使用 Makefile（推荐）
make test

# 方法2：直接运行
cd scripts && python baseline.py
```

## 可视化流程

### 启动 TensorBoard
```bash
# 方法1：使用 Makefile（推荐）
make tensorboard

# 方法2：直接运行
tensorboard --logdir=experiments/logs
```

## 项目管理

### 检查项目状态
```bash
make status
```

### 清理临时文件
```bash
make clean
```

### 生成项目结构报告
```bash
make report
```

## 目录结构说明

```
reinforcementLearningPetriNets/
├── src/                    # 源代码
│   ├── agents/            # CDQN, DQN实现
│   ├── environment/        # Petri网环境
│   ├── utils/             # 工具模块
│   └── rewards/           # 奖励函数
├── experiments/            # 实验相关
│   ├── results/           # CSV结果文件
│   ├── logs/              # 详细日志
│   └── checkpoints/        # 检查点文件
├── models/                 # 训练模型
│   ├── cdqn/             # CDQN模型
│   ├── dqn/              # DQN模型
│   └── gail/             # GAIL模型
├── scripts/                # 所有运行脚本
├── data/                  # 数据文件
└── docs/                  # 文档
```

## 常用命令组合

### 完整实验流程
```bash
# 1. 环境准备
make setup
make install

# 2. 批量训练
make train-all

# 3. 评估最优模型
make eval-all

# 4. 仿真测试
make simulate

# 5. 查看结果
ls experiments/results/
```

### 快速测试流程
```bash
# 1. 单个训练
make train

# 2. 基准测试
make test

# 3. 查看日志
ls experiments/logs/
```

## 注意事项

1. **路径设置**：运行任何脚本前都需要设置 PYTHONPATH
2. **目录权限**：确保 scripts/ 和 experiments/ 有写入权限
3. **依赖安装**：确保所有 Python 包都已安装
4. **GPU 支持**：如果有 GPU，训练会自动使用
5. **日志查看**：详细日志在 experiments/logs/ 目录
6. **结果查看**：最终结果在 experiments/results/ 目录

## 故障排除

### 导入错误
```bash
# 确保路径设置正确
echo $PYTHONPATH  # Linux/Mac
echo %PYTHONPATH%  # Windows
```

### 权限错误
```bash
# 检查目录权限
ls -la experiments/
ls -la models/
```

### 依赖缺失
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```