#!/usr/bin/env python3
"""
更新重构后脚本中的路径引用
"""

import os
import re

def update_file_paths(file_path, old_path, new_path):
    """更新文件中的路径引用"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换路径
    updated_content = content.replace(old_path, new_path)
    
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"已更新: {file_path}")
        return True
    else:
        print(f"无需更新: {file_path}")
        return False

def main():
    print("开始更新脚本中的路径引用...")
    
    # 1. 更新 train.py 中的日志路径
    train_py = "src/train.py" if os.path.exists("src/train.py") else "scripts/train.py"
    update_file_paths(
        train_py, 
        'log_dir="detailed_logs"', 
        'log_dir="experiments/logs"'
    )
    
    # 2. 更新 evaluation.py 中的模型路径
    eval_py = "src/evaluation.py" if os.path.exists("src/evaluation.py") else "scripts/evaluation.py"
    update_file_paths(
        eval_py,
        '"lido-run-events"',
        '"models"'
    )
    
    # 3. 更新 simulation.py 中的结果路径
    sim_py = "src/simulation.py" if os.path.exists("src/simulation.py") else "scripts/simulation.py"
    update_file_paths(
        sim_py,
        '"simulation-results.csv"',
        '"experiments/results/simulation-results.csv"'
    )
    
    # 4. 更新 add_to_comparison.py 中的路径
    add_comp_py = "src/add_to_comparison.py" if os.path.exists("src/add_to_comparison.py") else "scripts/add_to_comparison.py"
    update_file_paths(
        add_comp_py,
        '"lido-run-events/',
        '"models/'
    )
    
    # 5. 更新 verify_zero_violation.sh 中的路径
    verify_sh = "src/verify_zero_violation.sh" if os.path.exists("src/verify_zero_violation.sh") else "scripts/verify_zero_violation.sh"
    update_file_paths(
        verify_sh,
        'lido-run-events',
        'models'
    )
    
    print("路径更新完成！")

if __name__ == "__main__":
    main()