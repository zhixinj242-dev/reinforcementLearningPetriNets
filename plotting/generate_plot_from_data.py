"""
【文件角色】：根据仿真/训练产生的 CSV 数据生成图表。
可修改 base_path 与 parameters，用于绘制不同超参数组合的对比曲线。
"""
import pandas as pd
from matplotlib import pyplot as plt

# 数据根目录与要绘制的参数组
base_path = "~/Downloads/runs-v1"
parameters = [(1.5, 1.5, 1.5, 0.0, 1.0)]


def main():
    for param in parameters:
        data_path = "{}/run-s{}c{}w{}mw{}t{}.csv".format(base_path, param[0], param[1], param[2], param[3], param[4])
        # 在此处加载 data_path 并绘图，例如：df = pd.read_csv(data_path); plt.plot(...)
        # print(data_path)  # 调试时可取消注释



if __name__ == "__main__":
    main()
