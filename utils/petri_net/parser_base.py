from .entities import PetriPlace, PetriTransition, PetriArc
from bs4 import BeautifulSoup
from abc import abstractmethod

import snakes.plugins
snakes.plugins.load('gv', 'snakes.nets', 'nets')
from nets import *

class PetriFileReader(object):
    file: str
    bs: BeautifulSoup = None
    
    name: str = None
    places: [PetriPlace] = []
    transitions: [PetriTransition] = []
    arcs: [PetriArc] = []

    def __init__(self, file: str):
        self.file = file

    def convert_to_snakes(self):
        print("Start parsing petri net from file: {}".format(self.file))

        try:
            f = open(self.file, "r")
            self.bs = BeautifulSoup(f, "xml")

            self._get_petri_net_entities()
            return self._generate_snake_petri_net()
        except IOError:
            print("Error when parsing file")
            exit(1)

    def _generate_snake_petri_net(self) -> PetriNet:
      net = PetriNet(self.name)
      for place in self.places:
          marking = []
          if place.marking is not None:
              marking = range(int(place.marking))
          net.add_place(Place(name=place.id, tokens=marking))

      for transition in self.transitions:
          net.add_transition(Transition(name=transition.id))

      for arc in self.arcs:
          if arc.kind.lower() == "input":
              net.add_input(place=arc.tail, trans=arc.head, label=Variable('x'))
          elif arc.kind.lower() == "output":
              net.add_output(place=arc.head, trans=arc.tail, label=Variable('x'))

      return net

    def _get_petri_net_entities(self):
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