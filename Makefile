# 【文件角色】：自动化任务菜单。
# 它定义了一系列简短的命令，让你不用每次都敲一长串 Python 命令。

.PHONY: all clean

# 默认动作：训练 + 测试
all: train test

# 清理产生的临时文件
clean:
	@echo "********** 开始清理临时文件 **********"
	rm -f *-output.txt
	rm -f *.png
	@echo "清理完成"


# 一键启动训练
train:
	@echo "********** 启动 AI 训练流程 **********"
	python train.py -t

# 一键启动评估（基准测试）
test:
	@echo "********** 启动基准对照测试 **********"
	python baseline.py

# 一键启动 TensorBoard 可视化看板
tensorboard:
	@echo "******* 启动 TensorBoard 看板 (请在浏览器打开提示的 URL) *********"
	tensorboard --logdir=runs
