from mip import Model, xsum, CONTINUOUS, MINIMIZE
import sys
import numpy as np
from Graph import Graph

if __name__ == "__main__":
    graph = Graph("../../exampleGraphs/graphs/N40.graph", True)
    print(graph)