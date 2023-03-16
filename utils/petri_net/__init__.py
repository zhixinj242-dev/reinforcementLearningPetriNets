from .parser_pnpro import PNProParser
from .parser_pnml import PnmlParser
from enum import Enum

class Parser(Enum):
    PNPRO = 0
    PNML = 1


def get_petri_net(path: str, type: Parser = Parser.PNPRO):
    if type == Parser.PNML:
        return PnmlParser(path).convert_to_snakes()
    elif type == Parser.PNPRO:
        return PNProParser(path).convert_to_snakes()   
    else:
        raise Exception("Unknown parser type")