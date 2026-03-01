import pandas as pd
from matplotlib import pyplot as plt

base_path = "~/Downloads/runs-v1"

parameters = [(1.5, 1.5, 1.5, 0.0, 1.0)]


def main():
    for param in parameters:
        data_path = "{}/run-s{}c{}w{}mw{}t{}.csv".format(base_path, param[0], param[1], param[2], param[3], param[4])
        print(data_path)



if __name__ == "__main__":
    main()
