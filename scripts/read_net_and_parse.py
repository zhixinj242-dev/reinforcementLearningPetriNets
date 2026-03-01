"""
【脚本具体作用】：
1. 验证解析器（Parser）：测试代码能不能正确读懂 data/traffic-scenario.PNPRO 等文件，确保“翻译”过程没问题。
2. Petri 网逻辑检查：让你直观确认网的结构，包括：
   - 库所（Place）：检查灯态定义是否完整。
   - 变迁（Transition）：检查切换动作是否被正确识别。
   - 连线（Arc）：检查点与点之间的逻辑连接是否符合交通规则。
3. 快速预览：如果你修改了 data/ 里的 Petri 网图纸（比如加了一个新方向的灯），无需运行沉重的训练程序，
   直接跑这个小脚本，就能通过生成的图片立刻看到代码识别出的“点、线、面”是否正确。
"""
import argparse

from utils.petri_net import PNProParser, PnmlParser


import snakes.plugins
snakes.plugins.load('gv', 'snakes.nets', 'nets')
from nets import *


def main():
    """
    【函数功能】：Petri网读取与绘图。
    它负责把图纸文件（.pnpro或.pnml）读入程序，并顺便画一张PNG图片，让你直观看到网的结构。
    """
    args = parse_args()
    
    # 1. 根据文件后缀名选择合适的解析器
    if args.file.lower().endswith(".pnpro"):
        parser = PNProParser(args.file)
    else:
        parser = PnmlParser(args.file)

    # 2. 执行转换：把文件描述变成可运行的Net对象
    net = parser.convert_to_snakes()
    
    # 3. 绘图：把Petri网结构导出一张图片（需要系统安装了Graphviz）
    net.draw("{}.png".format(net.name))



def parse_args():
    """
    【函数功能】：命令行参数定义。
    让你可以通过 `python read_net_and_parse.py 文件名` 来指定要读取的文件。
    """
    parser = argparse.ArgumentParser("Petri net reader")
    parser.add_argument("file", help="Path to the file, which should be imported.") # 【参数】待解析的文件路径
    return parser.parse_args()


if __name__ == "__main__":
    main()
