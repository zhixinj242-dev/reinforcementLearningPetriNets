#!/bin/bash
# 批量运行 Baseline 测试，参数与 train_all.sh 保持一致

# 8组参数组合 (s, c, w, mw, t)
params=(
    "0.0 0.0 1.0 0.0 0.0"
    "0.0 0.0 1.0 1.5 0.0"
    "1.0 0.0 1.0 0.0 0.0"
    "1.0 0.0 1.0 1.5 0.0"
    "1.5 0.0 1.0 0.0 0.0"
    "1.5 0.0 1.0 1.5 0.0"
    "2.0 0.0 1.0 0.0 0.0"
    "2.0 0.0 1.0 1.5 0.0"
)

count=0
total=${#params[@]}

echo "=========================================="
echo "  开始 Baseline 批量测试 (共 $total 组)"
echo "=========================================="

for param in "${params[@]}"
do
  ((count++))
  # 解析参数
  read -r ms mc mw mmw mt <<< "$param"
  
  echo ""
  echo "[进度 $count/$total] 正在测试 Baseline (s=$ms, w=$mw, mw=$mmw)..."
  
  # 运行 baseline.py，结果会自动追加到 method_comparison.csv
  # 注意：baseline.py 已经修改为会自动根据参数生成 B1_s... 的名称
  python baseline.py \
      --reward-function dynamic_reward \
      --m-success $ms --m-cars-driven $mc --m-waiting-time $mw \
      --m-max-waiting-time $mmw --m-timestep $mt

done

echo ""
echo "=========================================="
echo "  所有 Baseline 测试已完成！"
echo "  结果已保存至 method_comparison.csv"
echo "=========================================="
