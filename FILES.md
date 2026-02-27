# 文件多余/冗余分析

## 一、运行产物（建议加入 .gitignore，不必删文件）

| 文件 | 说明 |
|------|------|
| `method_comparison.csv` | baseline 写入，每次跑 baseline 会覆盖 |
| `simulation-results.csv` | simulation.py 写入 |
| `comparison_all.csv` | merge 合并后输出 |
| `baseline_comparison.csv` | **多余**：旧版 baseline 输出，现已用 method_comparison.csv |
| `baseline_results.txt` | **多余**：若为终端重定向的旧结果，属运行产物 |
| `simulation_results.csv` | **多余**：与 simulation-results.csv 命名相似，代码里只用 simulation-results.csv，疑为旧文件 |
| `B1.mp4`, `B2.mp4`, `baseline_v1.mp4` | compare_visual.sh 生成的视频，运行产物 |
| `traffic-scenario.png` | 某脚本生成的图，运行产物 |
| `lido-run-events/` | 训练 checkpoint 与 *_best.pt 目录，运行产物 |

---

## 二、可删除或合并的脚本/文件

| 文件 | 结论 |
|------|------|
| `monitor_training.sh` | 读 `train_logs/*.log`，与 `monitor_progress.py`（读 lido-run-events 与 simulation-results.csv）不是同一套；若项目不用 train_logs，可删 |
| `monitor_training.ps1` | Windows 版，功能同 monitor_training.sh；若不用 log 监控可删 |
| `plotting/generate_plot_from_data.py` | 读的是旧格式 `run-s{}c{}w{}mw{}t{}.csv` 和路径 `~/Downloads/runs-v1`，与当前 simulation-results / comparison_all 不一致，**多余**除非你改写成读当前 CSV |
| `plotting/generate_plot.ipynb` | 若只做一次性画图可保留；若与 generate_plot_from_data.py 重复可二选一 |

---

## 三、建议保留

| 文件 | 说明 |
|------|------|
| `get_transitions.py` | 查 Petri 网变迁名与动作索引，写 baseline 序列时有用 |
| `add_to_comparison.py` | 单模型（GAIL、单个 CDQN/DQN）追加到 method_comparison，与 simulation 批量 + merge 互补 |
| `compare_visual.sh` | 生成 B1/B2/GAIL 视频，一键对比 |
| `eval_best_all.sh` | 16 个实验选最优 *_best.pt |
| `grid_search.sh` | 网格搜索训练 |
| `monitor_progress.py` | 监控训练 checkpoint 与 simulation 进度 |

---

## 四、建议操作汇总

1. **加入 .gitignore**（若尚未包含）：  
   `method_comparison.csv`, `simulation-results.csv`, `comparison_all.csv`, `baseline_comparison.csv`, `baseline_results.txt`, `simulation_results.csv`, `*.mp4`, `traffic-scenario.png`, `lido-run-events/`

2. **可删除的多余文件**（按需）：  
   - `baseline_comparison.csv`  
   - `baseline_results.txt`  
   - `simulation_results.csv`（仅当确认与 simulation-results.csv 重复且不需旧数据时）  
   - `monitor_training.sh` / `monitor_training.ps1`（若不用 log 监控）  
   - `plotting/generate_plot_from_data.py`（若不再用旧 CSV 格式画图）

3. **simulation.py 注释**：第 4 行写的是「simulation_results.csv」，实际写入的是 `simulation-results.csv`，建议把注释改成 simulation-results.csv 避免误解。
