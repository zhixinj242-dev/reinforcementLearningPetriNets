from .parser_base import PetriFileReader
from .entities import PetriPlace, PetriTransition, PetriArc


class PnmlParser(PetriFileReader):

    def _parse_name(self) -> str:
        project_space = self.bs.find("name")
        project_space = project_space.findChild("text")
        self.name = project_space.text

    def _parse_places(self):
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
        self.transitions = []
        transitions_xml = self.bs.findAll("transition")
        for trans in transitions_xml:
            name = trans.findChild("name").findChild("text").text
            offset = trans.findChild("name").findChild("graphics").findChild("offset")
            self.transitions.append(PetriTransition(trans.attrs["id"], name, "", "", int(offset.attrs["x"]), int(offset.attrs["y"])))

    def _parse_arcs(self):
        self.arcs = []
        arcs_xml = self.bs.findAll("arc")
        for arc in arcs_xml:
            source = arc.attrs["source"]
            target = arc.attrs["target"]
            type = "input"
            if source in [place.id for place in self.places]:
                type = "output"
            self.arcs.append(PetriArc(source, type, target))