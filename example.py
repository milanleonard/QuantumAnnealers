import networkx as nx
from itertools import combinations, groupby
import random
from functools import partial
from collections import defaultdict
import time
import random
import numpy as np
import scipy
from problems import MaxCutNXProblem
from schedulers import LinearScheduler
import os
import glob

from annealers import QutipStateVectorAnnealer

data = []

def main():
    num_qubits = 8
    anneal_time = 5
    GraphProblem = MaxCutNXProblem(num_qubits, p=0.5, seed=random.randint(0,2**32))
    quboproblem = GraphProblem.generate_qubo()
    graph_features = GraphProblem.graph_features()
    Annealer = QutipStateVectorAnnealer(LinearScheduler(total_time=anneal_time,num_timesteps=1000), quboproblem, num_qubits)
    results = Annealer.anneal()  # At the moment is calling E_OPS and then can't get statevector to sample, probably not what I want
    optimal = Annealer.optimal_result()
    datum = graph_features
    datum['final'] = results.expect[0][-1]
    datum['optimal'] = optimal

    print(f"{optimal=}, got: {results.expect[0][-1]}")


def cleanup():
    files = glob.glob('./*qobj*')
    for f in files:
        os.remove(f)




if __name__ == "__main__":
    main()
    cleanup()