from .parser_base import PetriFileReader
from .entities import PetriPlace, PetriTransition, PetriArc


class PnmlParser(PetriFileReader):
    """
    【类功能】：PNML文件解析器。
    专门用于读取和解析标准PNML（Petri Net Markup Language）格式的Petri网文件。
    它继承自 `PetriFileReader`，实现了具体的PNML文件解析逻辑。
    """
    def _parse_name(self) -> str:
        """从PNML文件中解析Petri网的名称"""
        project_space = self.bs.find("name")
        project_space = project_space.findChild("text")
        self.name = project_space.text

    def _parse_places(self):
        """解析所有的库所（Place）及其初始Token和坐标"""
        places_xml = self.bs.findAll("place")
        self.places = []
        for place in places_xml:
            name_tag = place.findChild("name")
            marking_tag = place.findChild("initialMarking")
            name = name_tag.findChild("text").text
            mark = None
            if marking_tag is not None:
                mark = int(place.findChild("initialMarking").findChild("text").text)
            offset = place.findChild("name").findChild("graphics").findChild("offset")
            self.places.append(PetriPlace(place.attrs["id"], name, mark, int(offset.attrs["x"]), int(offset.attrs["y"])))

    def _parse_transitions(self):
        """解析所有的变迁（Transition）及其坐标"""
        self.transitions = []
        transitions_xml = self.bs.findAll("transition")
        for trans in transitions_xml:
            name = trans.findChild("name").findChild("text").text
            offset = trans.findChild("name").findChild("graphics").findChild("offset")
            self.transitions.append(PetriTransition(trans.attrs["id"], name, "", "", int(offset.attrs["x"]), int(offset.attrs["y"])))

    def _parse_arcs(self):
        """解析所有的弧（Arc），并判断它是输入弧还是输出弧"""
        self.arcs = []
        arcs_xml = self.bs.findAll("arc")
        for arc in arcs_xml:
            source = arc.attrs["source"]
            target = arc.attrs["target"]
            type = "input"
            # 如果起点是库所，那么对变迁来说它就是输入弧
            if source in [place.id for place in self.places]:
                type = "output"
            self.arcs.append(PetriArc(source, type, target))
