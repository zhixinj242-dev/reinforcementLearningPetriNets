from .parser_base import PetriFileReader
from .entities import PetriPlace, PetriTransition, PetriArc

import random
import string

class PNProParser(PetriFileReader):

  def _parse_name(self) -> str:
    project_space = self.bs.find("project")
    if "name" in project_space.attrs.keys():
        self.name = project_space.attrs.get("name")
    else:
        self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

  def _parse_places(self):
    places_xml = self.bs.findAll("place")
    self.places = [PetriPlace(
                id=p.attrs.get("name"),
                name=p.attrs.get("name"),
                marking=p.attrs.get("marking"),
                x=p.attrs.get("x"),
                y=p.attrs.get("y")) for p in places_xml]

  def _parse_transitions(self):
    transitions_xml = self.bs.findAll("transition")
    self.transitions =  [PetriTransition(
                id=t.attrs.get("name"),
                name=t.attrs.get("name"),
                nservers_x=t.attrs.get("nservers-x"),
                p_type=t.attrs.get("type"),
                x=t.attrs.get("x"),
                y=t.attrs.get("y")) for t in transitions_xml]

  def _parse_arcs(self):
    arcs_xml = self.bs.findAll("arc")
    self.arcs = [PetriArc(
        head=a.attrs.get("head"),
        kind=a.attrs.get("kind"),
        tail=a.attrs.get("tail")
    ) for a in arcs_xml]