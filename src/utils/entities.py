"""
【文件角色】：定义 Petri 网中最基础的“物理实体”。
包含：库所（Place）、变迁（Transition）、弧（Arc）。
这些类是解析器从 XML 文件中提取数据后的临时存放容器。
"""

class PetriPlace(object):
    """
    【类功能】：Petri网中的“库所”（Place）数据结构。
    【角色】：代表信号灯的某种状态（如：南北红灯、东西绿灯）。
    【属性】：
    - id/name: 库所的唯一标识和名称。
    - marking: 初始时库所中Token的数量（黑点）。
    - x, y: 库所在图纸上的坐标。
    """
    marking: int
    id: str
    name: str
    x: float
    y: float

    def __init__(self, id: str, name: str, marking: int, x: float, y: float):
        self.marking = marking
        self.id = id
        self.name = name
        self.x = x
        self.y = y

    def __str__(self):
        return "PetriPlace(name={}, marking={}, x={}, y={})".format(self.name, self.marking, self.x, self.y)

    def __repr__(self):
        return self.__str__()


class PetriTransition(object):
    """
    【类功能】：Petri网中的“变迁”（Transition）数据结构。
    【角色】：代表信号灯的切换动作（如：从红灯切换到绿灯）。
    【属性】：
    - id/name: 变迁的唯一标识和名称。
    - nservers_x, p_type, x, y: 变迁的其他属性和坐标。
    """
    name: str
    id: str
    nservers_x: str
    p_type: str
    x: float
    y: float

    def __init__(self, id: str, name: str, nservers_x: str, p_type: str, x: float, y: float):
        self.nservers_x = nservers_x
        self.name = name
        self.id = id
        self.p_type = p_type
        self.x = x
        self.y = y

    def __str__(self):
        return "PetriTransition(name={}, nservers-x={}, p_type={} x={}, y={})".format(self.name, self.nservers_x,
                                                                                      self.p_type, self.x, self.y)

    def __repr__(self):
        return self.__str__()


class PetriArc(object):
    """
    【类功能】：Petri网中的“弧”（Arc）数据结构。
    【角色】：连接库所和变迁，定义Token的流向和规则。
    【属性】：
    - head: 弧的终点（变迁或库所的ID）。
    - kind: 弧的类型（"input"表示输入弧，"output"表示输出弧）。
    - tail: 弧的起点（库所或变迁的ID）。
    """
    head: str
    kind: str
    tail: str

    def __init__(self, head: str, kind: str, tail: str):
        self.head = head
        self.kind = kind
        self.tail = tail

    def __str__(self):
        return "PetriArc(head={}, kind={}, tail={})".format(self.head, self.kind, self.tail)

    def __repr__(self):
        return self.__str__()
