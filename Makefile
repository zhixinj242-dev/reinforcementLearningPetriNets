# 【文件角色】：自动化任务菜单。
# 适配重构后的目录结构：src/, scripts/, experiments/, models/, data/, docs/

.PHONY: all clean train test evaluate simulate tensorboard setup

# 默认动作：训练 + 测试
all: setup train test

# 设置环境变量
setup:
	@echo "********** 设置 Python 路径 **********"
	export PYTHONPATH=$$(pwd)/src:$$PYTHONPATH || set PYTHONPATH=%CD%\src;%PYTHONPATH%
	@echo "Python 路径已设置"

# 清理产生的临时文件
clean:
	@echo "********** 开始清理临时文件 **********"
	rm -f *-output.txt
	rm -f *.png
	rm -f *.pt
	rm -f *.csv
	rm -f *.log
	rm -rf experiments/logs/*
	rm -rf models/*/*
	@echo "清理完成"

# 一键启动训练
train:
	@echo "********** 启动 AI 训练流程 **********"
	cd scripts && python train.py

# 一键启动评估（基准测试）
test:
	@echo "********** 启动基准对照测试 **********"
	cd scripts && python baseline.py

# 一键启动模型评估
evaluate:
	@echo "********** 启动模型评估 **********"
	cd scripts && python evaluation.py --best-from-dir ../experiments/checkpoints

# 一键启动仿真测试
simulate:
	@echo "********** 启动仿真测试 **********"
	cd scripts && python simulation.py

# 批量训练所有参数组合
train-all:
	@echo "********** 批量训练所有参数组合 **********"
	cd scripts && bash train_all.sh

# 批量评估所有最优模型
eval-all:
	@echo "********** 批量评估所有最优模型 **********"
	cd scripts && bash eval_best_all.sh

# 一键启动 TensorBoard 可视化看板
tensorboard:
	@echo "******* 启动 TensorBoard 看板 (请在浏览器打开提示的 URL) *********"
	tensorboard --logdir=experiments/logs

# 安装依赖
install:
	@echo "********** 安装项目依赖 **********"
	pip install -r requirements.txt

# 生成项目结构报告
report:
	@echo "********** 生成项目结构报告 **********"
	tree /F > docs/project-structure.txt
	@echo "项目结构报告已生成: docs/project-structure.txt"

# 检查项目状态
status:
	@echo "********** 检查项目状态 **********"
	@echo "=== 目录结构 ==="
	tree -L 2
	@echo "=== Git 状态 ==="
	git status --short
	@echo "=== Python 路径 ==="
	python -c "import sys; print('Python路径:'); [print(f'  {p}') for p in sys.path[:3]]"

# 运行测试
run-tests:
	@echo "********** 运行项目测试 **********"
	cd src && python -m pytest ../tests/ -v

# 帮助信息
help:
	@echo "********** 可用命令 **********"
	@echo "setup      - 设置 Python 路径"
	@echo "train      - 启动训练"
	@echo "test       - 启动基准测试"
	@echo "evaluate   - 启动模型评估"
	@echo "simulate    - 启动仿真测试"
	@echo "train-all  - 批量训练所有参数组合"
	@echo "eval-all   - 批量评估所有最优模型"
	@echo "tensorboard - 启动 TensorBoard"
	@echo "install    - 安装依赖"
	@echo "clean      - 清理临时文件"
	@echo "report     - 生成项目结构报告"
	@echo "status     - 检查项目状态"
	@echo "run-tests  - 运行测试"
	@echo "help       - 显示帮助信息"