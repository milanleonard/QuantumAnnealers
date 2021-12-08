from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
from problems import MaxCutNXProblem
sampler = EmbeddingComposite(DWaveSampler())

NUM_QUBITS = 10
PROBLEMS = [MaxCutNXProblem(NUM_QUBITS, j, (j*i*10000)) for j in np.arange(0.1,0.9,0.1) for i in range(1,15)]
PROBLEM_QUBOS = [problem.generate_qubo() for problem in PROBLEMS]

dwaveresults = []
for problem in PROBLEM_QUBOS:
    computation = sampler.sample_ising([], problem, num_reads=50)
    df = computation.to_pandas_dataframe()
    dwaveresults.append(np.average(df['energy'], weights = df['num_occurrences']))
    print(dwaveresults)

np.save('work_dwaveresults.npy', np.array(dwaveresults))
