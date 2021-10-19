#%%
from problems import MaxCutNXProblem
from schedulers import LinearScheduler
from annealers import QutipDensityMatrixAnnealer, QutipStateVectorAnnealer
import numpy as np  
import multiprocessing
from functools import partial
from dwave.system import DWaveSampler, EmbeddingComposite
import pickle

# %%
def perform_dm_anneal(num_qubits : int, anneal_time : float, quboproblem : dict):
    annealer = QutipDensityMatrixAnnealer(LinearScheduler(total_time=anneal_time,num_timesteps=1000), quboproblem, num_qubits)
    annealer.anneal()
    ratio = annealer.get_expectation_val() / annealer.get_optimal()
    return ratio if 0 < ratio < 1 else 0

def perform_sv_anneal(num_qubits : int, anneal_time : float, quboproblem : dict) -> QutipStateVectorAnnealer:
    annealer = QutipStateVectorAnnealer(LinearScheduler(total_time=anneal_time,num_timesteps=1000), quboproblem, num_qubits)
    annealer.anneal()
    ratio = np.real(annealer.expect[-1]) / annealer.get_optimal()
    return (ratio, annealer.get_optimal()) if 0 < ratio < 1 else (0, annealer.get_optimal())

# %%
NUM_QUBITS = 10
PROBLEMS = [MaxCutNXProblem(NUM_QUBITS, j, (j*i*10000)) for j in np.arange(0.1,0.9,0.1) for i in range(1,15)]
# %%
def do_total_computation(problem : MaxCutNXProblem):
    qubo = problem.generate_qubo()
    output = problem.graph_features()
    ratiosv, optimal = perform_sv_anneal(NUM_QUBITS, 3, qubo)
    ratiodm = perform_dm_anneal(NUM_QUBITS, 3, qubo)
    output['algo_ideal'] = ratiosv
    output['algo_diamond'] = ratiodm
    output['optimal'] = optimal
    return output

with multiprocessing.Pool(16) as pool:
    outputs = pool.map(do_total_computation, PROBLEMS)
# outputs = []
# for problem in PROBLEMS:
#     outputs.append(do_total_computation(problem))

arr = np.load('./data/dwaveresults.npy')
for idx, output in enumerate(outputs):
    output['algo_dwave'] = arr[idx] / output['optimal']

# %%
with open('all_data.pkl','wb') as f:
    pickle.dump(outputs, f)