from .entities import PetriPlace, PetriTransition, PetriArc
from bs4 import BeautifulSoup
from abc import abstractmethod

import snakes.plugins
snakes.plugins.load('gv', 'snakes.nets', 'nets')
from nets import *

class PetriFileReader(object):
    """
    【文件角色】：Petri网文件读取器的抽象基类。
    它定义了所有Petri网解析器都必须遵循的通用流程。
    """
    file: str
    bs: BeautifulSoup = None # XML解析引擎
    
    name: str = None
    places: [PetriPlace] = []
    transitions: [PetriTransition] = []
    arcs: [PetriArc] = []

    def __init__(self, file: str):
        self.file = file

    def convert_to_snakes(self):
        """
        【核心函数】：转换器。
        把静态的文件数据转化成可执行的 Petri 网对象（snakes 库格式）。
        """
        try:
            f = open(self.file, "r")
            self.bs = BeautifulSoup(f, "xml") # 第一步：用BeautifulSoup读取XML格式的文件

            self._get_petri_net_entities()    # 第二步：把文件里的点、线、面都读出来
            return self._generate_snake_petri_net() # 第三步：把这些零件组装成“引擎”
        except IOError:
            # 文件不存在或无法读取时退出
            raise IOError("Petri net file not found or unreadable: {}".format(self.file))

    def _generate_snake_petri_net(self) -> PetriNet:
      """
      【函数功能】：Petri网组装师。
      它会根据读出来的零件清单，创建一个真正的、具备逻辑运算能力的PetriNet对象。
      """
      net = PetriNet(self.name)
      
      # 1. 添加库所（Place）：信号灯的状态节点
      for place in self.places:
          marking = []
          if place.marking is not None:
              marking = range(int(place.marking)) # 放置初始Token
          net.add_place(Place(name=place.id, tokens=marking))

      # 2. 添加变迁（Transition）：信号灯的切换动作
      for transition in self.transitions:
          net.add_transition(Transition(name=transition.id))

      # 3. 添加弧（Arc）：连接状态和动作的线
      for arc in self.arcs:
          if arc.kind.lower() == "input":
              # 输入弧：动作消耗Token（箭头指向动作）
              net.add_input(place=arc.tail, trans=arc.head, label=Variable('x'))
          elif arc.kind.lower() == "output":
              # 输出弧：动作产生Token（箭头指向状态）
              net.add_output(place=arc.head, trans=arc.tail, label=Variable('x'))

      return net

    def _get_petri_net_entities(self):
        """实体解析流程控制"""
        self._parse_name()
        self._parse_places(),
        self._parse_transitions()
        self._parse_arcs()

    @abstractmethod
    def _parse_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _parse_places(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_transitions(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_arcs(self):
        raise NotImplementedError
