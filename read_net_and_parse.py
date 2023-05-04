import argparse

from utils.petri_net import PNProParser, PnmlParser


import snakes.plugins
snakes.plugins.load('gv', 'snakes.nets', 'nets')
from nets import *


def main():
    args = parse_args()
    
    if args.file.lower().endswith(".pnpro"):
        parser = PNProParser(args.file)
    else:
        parser = PnmlParser(args.file)

    net = parser.convert_to_snakes()
    net.draw("{}.png".format(net.name))



def parse_args():
    parser = argparse.ArgumentParser("Petri net reader")
    parser.add_argument("file", help="Path to the file, which should be imported.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
