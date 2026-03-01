from .parser_base import PetriFileReader
from .entities import PetriPlace, PetriTransition, PetriArc

import random
import string

class PNProParser(PetriFileReader):
    """
    【类功能】：PNPRO文件解析器。
    专门用于读取和解析GreatSPN编辑器生成的 `.PNPRO` 格式的Petri网文件。
    它是本项目中 `data/traffic-scenario.PNPRO` 文件的专用解析器。
    """
    def _parse_name(self) -> str:
        """从PNPRO文件中解析项目名称"""
        project_space = self.bs.find("project")
        if "name" in project_space.attrs.keys():
            self.name = project_space.attrs.get("name")
        else:
            self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    def _parse_places(self):
        """从PNPRO特定的XML结构中解析库所信息"""
        places_xml = self.bs.findAll("place")
        self.places = [PetriPlace(
                    id=p.attrs.get("name"),
                    name=p.attrs.get("name"),
                    marking=p.attrs.get("marking"),
                    x=p.attrs.get("x"),
                    y=p.attrs.get("y")) for p in places_xml]

    def _parse_transitions(self):
        """从PNPRO特定的XML结构中解析变迁信息"""
        transitions_xml = self.bs.findAll("transition")
        self.transitions =  [PetriTransition(
                    id=t.attrs.get("name"),
                    name=t.attrs.get("name"),
                    nservers_x=t.attrs.get("nservers-x"),
                    p_type=t.attrs.get("type"),
                    x=t.attrs.get("x"),
                    y=t.attrs.get("y")) for t in transitions_xml]

    def _parse_arcs(self):
        """从PNPRO特定的XML结构中解析弧信息"""
        arcs_xml = self.bs.findAll("arc")
        self.arcs = [PetriArc(
            head=a.attrs.get("head"),
            kind=a.attrs.get("kind"),
            tail=a.attrs.get("tail")
        ) for a in arcs_xml]
