from .parser_pnpro import PNProParser
from .parser_pnml import PnmlParser
from enum import Enum

"""
【文件角色】：`utils.petri_net` 包的入口和调度员。
它负责把不同的解析器（PNPRO, PNML）统一封装，给外界提供一个简单的接口。
"""

class Parser(Enum):
    """支持的 Petri 网文件格式枚举"""
    PNPRO = 0
    PNML = 1


def get_petri_net(path: str, type: Parser = Parser.PNPRO):
    """
    【函数功能】：Petri网加载器。
    根据文件类型（PNPRO或PNML），选择对应的解析器，将文件内容转换为可操作的Petri网对象。
    """
    if type == Parser.PNML:
        return PnmlParser(path).convert_to_snakes()
    elif type == Parser.PNPRO:
        return PNProParser(path).convert_to_snakes()   
    else:
        raise Exception("Unknown parser type")
