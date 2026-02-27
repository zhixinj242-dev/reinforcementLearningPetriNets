#!/bin/bash

# 1. 确保有 _best.pt 模型
echo "正在从训练记录中提取最优模型..."
python evaluation.py --best-from-dir lido-run-events

# 2. 运行仿真验证
echo "============================================================"
echo "开始验证 CDQN 违规情况 (Expect: 0 violations)"
echo "============================================================"

# 运行 simulation.py，并只关注违规相关的输出
# 我们只跑几轮 (iterations=5) 快速验证
# 注意：你需要确保 simulation.py 里的 params 列表里有你实际训练过的参数组合
# 这里我们假设你训练了第一组参数 (0.0, 0.0, 1.0, 0.0, 0.0)

python -c "
import simulation
# 临时修改 iterations 为 5，只跑一小会儿
simulation.main((0.0, 0.0, 1.0, 0.0, 0.0), model_type='cdqn', iterations=5)
" 

echo "============================================================"
echo "验证结束。如果上方显示的 'avg_c_broken' 为 0.00% 或 '违规=0'，则修复成功！"
