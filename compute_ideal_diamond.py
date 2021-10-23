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

PROBLEMS = [MaxCutNXProblem(num_qubits, j, (j*i*10000)) for j in np.arange(0.1,0.95,0.05) for i in range(1,5) for num_qubits in range(5,11)]
# %%
def do_total_computation(problem : MaxCutNXProblem):
    print("j")
    qubo = problem.generate_qubo()
    output = problem.graph_features()
    ratiosv, optimal = perform_sv_anneal(problem.num_qubits, 3, qubo)
    ratiodm = perform_dm_anneal(problem.num_qubits, 3, qubo)
    output['algo_ideal'] = ratiosv
    output['algo_diamond'] = ratiodm
    output['optimal'] = optimal
    return output

with multiprocessing.Pool(31) as pool:
    outputs = pool.map(do_total_computation, PROBLEMS)

with open('./data/costofdephasing.pkl','wb') as f:
    pickle.dump(outputs, f)