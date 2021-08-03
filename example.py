import random
import random
import numpy as np
from scipy.linalg.decomp import eig
from problems import MaxCutNXProblem
from schedulers import LinearScheduler
import os
import glob
import matplotlib.pyplot as plt

from annealers import QutipStateVectorAnnealer

data = []


def perform_anneal(num_qubits : int, anneal_time : float, quboproblem : dict) -> QutipStateVectorAnnealer:
    annealer = QutipStateVectorAnnealer(LinearScheduler(total_time=anneal_time,num_timesteps=1000), quboproblem, num_qubits)
    annealer.anneal()
    return annealer

def main():
    num_qubits = 4
    anneal_time = 5
    GraphProblem = MaxCutNXProblem(num_qubits, p=0.5, seed=random.randint(0,2**32))
    quboproblem = GraphProblem.generate_qubo()
    graph_features = GraphProblem.graph_features()
    annealer = perform_anneal(num_qubits, anneal_time, quboproblem)
    test_times = np.linspace(0,anneal_time,10)
    eigspecvals = np.array([annealer.eigenspec_time_t(test_time)[0] for test_time in test_times])
    plt.plot(test_times,eigspecvals, 'r.')
    plt.show()



def cleanup():
    files = glob.glob('./*qobj*')
    for f in files:
        os.remove(f)




if __name__ == "__main__":
    annealer = main()
    cleanup()