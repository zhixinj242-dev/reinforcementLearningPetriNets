#!/bin/bash

echo "开始重构项目目录结构..."

# 1. 创建新的目录结构
echo "创建新的目录结构..."
mkdir -p src/{agents,environment,utils,rewards}
mkdir -p experiments/{configs,results,logs,checkpoints}
mkdir -p models/{cdqn,dqn,gail}
mkdir -p scripts
mkdir -p docs
mkdir -p data

# 2. 移动源代码
echo "移动源代码..."
if [ -d "agents" ]; then
    mv agents/* src/agents/ 2>/dev/null || true
    rmdir agents 2>/dev/null || true
fi

if [ -d "environment" ]; then
    mv environment/* src/environment/ 2>/dev/null || true
    rmdir environment 2>/dev/null || true
fi

if [ -d "utils" ]; then
    # 保留 utils 目录结构
    if [ -d "utils/petri_net" ]; then
        mv utils/petri_net src/utils/ 2>/dev/null || true
    fi
    mv utils/log_manager.py src/utils/ 2>/dev/null || true
    mv utils/result_comparison.py src/utils/ 2>/dev/null || true
    # 移动其他utils文件
    find utils -maxdepth 1 -type f -name "*.py" -exec mv {} src/utils/ \; 2>/dev/null || true
    rmdir utils 2>/dev/null || true
fi

if [ -d "rewards" ]; then
    mv rewards/* src/rewards/ 2>/dev/null || true
    rmdir rewards 2>/dev/null || true
fi

# 3. 整理模型文件
echo "整理模型文件..."
if [ -d "lido-run-events" ]; then
    # 移动CDQN模型
    find lido-run-events -name "*_cdqn_*.pt" -exec mv {} models/cdqn/ \; 2>/dev/null || true
    find lido-run-events -name "*_cdqn_best.pt" -exec mv {} models/cdqn/ \; 2>/dev/null || true
    
    # 移动DQN模型
    find lido-run-events -name "*_dqn_*.pt" -exec mv {} models/dqn/ \; 2>/dev/null || true
    find lido-run-events -name "*_dqn_best.pt" -exec mv {} models/dqn/ \; 2>/dev/null || true
    
    # 移动GAIL模型
    find lido-run-events -name "*gail*.pt" -exec mv {} models/gail/ \; 2>/dev/null || true
    
    # 移动checkpoint文件
    find lido-run-events -name "*/checkpoints" -type d -exec cp -r {} experiments/checkpoints/ \; 2>/dev/null || true
    
    echo "模型文件整理完成"
fi

# 4. 整理日志文件
echo "整理日志文件..."
if [ -d "detailed_logs" ]; then
    mv detailed_logs/* experiments/logs/ 2>/dev/null || true
    rmdir detailed_logs 2>/dev/null || true
fi

if [ -d "debug_logs" ]; then
    mv debug_logs/* experiments/logs/ 2>/dev/null || true
    rmdir debug_logs 2>/dev/null || true
fi

# 5. 整理结果文件
echo "整理结果文件..."
# 移动CSV结果文件
find . -maxdepth 1 -name "*.csv" -exec mv {} experiments/results/ \; 2>/dev/null || true

# 移动其他结果文件
find . -maxdepth 1 -name "*.png" -exec mv {} experiments/results/ \; 2>/dev/null || true
find . -maxdepth 1 -name "*.mp4" -exec mv {} experiments/results/ \; 2>/dev/null || true

# 6. 移动脚本文件
echo "移动脚本文件..."
# 移动shell脚本
find . -maxdepth 1 -name "*.sh" -exec mv {} scripts/ \; 2>/dev/null || true

# 移动Python脚本
find . -maxdepth 1 -name "add_to_comparison.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "baseline.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "evaluation.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "simulation.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "train.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "visual.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "verify_*.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "monitor_*.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "plot_*.py" -exec mv {} scripts/ \; 2>/dev/null || true
find . -maxdepth 1 -name "read_*.py" -exec mv {} scripts/ \; 2>/dev/null || true

# 7. 移动文档文件
echo "移动文档文件..."
find . -maxdepth 1 -name "*.md" -exec mv {} docs/ \; 2>/dev/null || true

# 8. 移动数据文件
echo "移动数据文件..."
if [ -d "data" ]; then
    # data目录已存在，保持原样
    echo "data目录已存在，保持原样"
else
    echo "创建data目录"
    mkdir data
fi

# 9. 移动plotting目录
if [ -d "plotting" ]; then
    mv plotting/* scripts/ 2>/dev/null || true
    rmdir plotting 2>/dev/null || true
fi

# 10. 移动slurm目录
if [ -d "slurm" ]; then
    mv slurm/* scripts/ 2>/dev/null || true
    rmdir slurm 2>/dev/null || true
fi

# 11. 清理空目录和特殊目录
echo "清理特殊目录..."
if [ -d ".ipynb_checkpoints" ]; then
    rm -rf .ipynb_checkpoints
    echo "删除 .ipynb_checkpoints 目录"
fi

# 12. 创建新的.gitignore
echo "创建新的.gitignore..."
cat > .gitignore << 'EOF'
# 运行产物
experiments/
models/
*.pt
*.csv
*.log
*.mp4
*.png

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db
EOF

# 13. 创建项目结构说明
echo "创建项目结构说明..."
cat > docs/PROJECT_STRUCTURE.md << 'EOF'
# 项目目录结构说明

## 目录结构

```
reinforcementLearningPetriNets/
├── src/                    # 源代码
│   ├── agents/            # 智能体实现
│   ├── environment/        # 环境实现
│   ├── utils/             # 工具模块
│   └── rewards/           # 奖励函数
├── experiments/            # 实验相关
│   ├── configs/           # 实验配置
│   ├── results/           # 实验结果
│   ├── logs/              # 实验日志
│   └── checkpoints/        # 检查点
├── models/                 # 训练模型
│   ├── cdqn/             # CDQN模型
│   ├── dqn/              # DQN模型
│   └── gail/             # GAIL模型
├── scripts/                # 运行脚本
├── data/                  # 数据文件
└── docs/                  # 文档
```

## 使用说明

### 训练
```bash
cd scripts
python train.py --train --constrained --m-success 1.0 --m-cars-driven 0.0 --m-waiting-time 1.0 --m-max-waiting-time 0.0 --m-timestep 0.0
```

### 评估
```bash
cd scripts
python evaluation.py --best-from-dir ../experiments/checkpoints
```

### 仿真
```bash
cd scripts
python simulation.py
```

### 结果查看
```bash
# 查看实验结果
cat experiments/results/*.csv

# 查看实验日志
ls experiments/logs/
```
EOF

echo "重构完成！"
echo "新的目录结构："
tree -L 2

# 14. 提示用户下一步操作
echo ""
echo "=== 重构完成后的操作建议 ==="
echo "1. 更新脚本中的路径引用"
echo "2. 测试重构后的功能是否正常"
echo "3. 提交重构到版本控制"
echo ""
echo "主要需要更新的路径："
echo "- scripts/train.py 中的日志路径: experiments/logs/"
echo "- scripts/evaluation.py 中的模型路径: models/"
echo "- scripts/simulation.py 中的结果路径: experiments/results/"