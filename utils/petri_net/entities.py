

class PetriPlace(object):
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