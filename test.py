import networkx as nx
from itertools import combinations, groupby
import random
from functools import partial
from collections import defaultdict
import time
import random
import numpy as np
import scipy
from problem import MaxCutNXProblem
from schedulers import LinearScheduler

from annealer import QutipSchrodingerAnnealer

data = []

def main():
    num_qubits = 12
    anneal_time = 5
    GraphProblem = MaxCutNXProblem()
    quboproblem = GraphProblem.generate_qubo()
    graph_features = GraphProblem.graph_features()
    Annealer = QutipSchrodingerAnnealer(LinearScheduler(5,1000), quboproblem, num_qubits)
    results = Annealer.anneal()  # At the moment is calling E_OPS and then can't get statevector to sample, probably not what I want
    optimal = Annealer.optimal_result()
    datum = graph_features
    datum['final'] = results.expect[0][-1]
    datum['optimal'] = optimal

    print(f"{optimal=}, got: {results.expect[0][-1]}")




if __name__ == "__main__":
    main()