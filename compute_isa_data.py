#%%
import problems
from schedulers import LinearScheduler
from annealers import QutipDensityMatrixAnnealer
import numpy as np  
import multiprocessing
from functools import partial

# %%
def perform_dm_anneal(num_qubits : int, anneal_time : float, quboproblem : dict):
    print("j")
    annealer = QutipDensityMatrixAnnealer(LinearScheduler(total_time=anneal_time,num_timesteps=1000), quboproblem, num_qubits)
    annealer.anneal()
    ratio = annealer.get_expectation_val() / annealer.get_optimal()
    return ratio if 0 < ratio < 1 else 0
    #return anneal_vals.expect[0][-1]
# %%
NUM_QUBITS = 3
PROBLEMS = [problems.MaxCutNXProblem(NUM_QUBITS, j, (j*i*10000)) for j in np.arange(0.1,0.9,0.1) for i in range(1,3)]
PROBLEM_QUBOS = [problem.generate_qubo() for problem in PROBLEMS]
# %%
anneal_times = np.linspace(0.2,5,20)
ratios = np.zeros(len(anneal_times))
with multiprocessing.Pool(31) as pool:
    for idx, anneal_time in enumerate(anneal_times):
        this_partial = partial(perform_dm_anneal, NUM_QUBITS, anneal_time)
        rats = pool.map(this_partial, PROBLEM_QUBOS)
        ratios[idx] = np.mean(rats)
# %%
test = np.vstack((anneal_times, ratios))
# %%
with open('./timesdata.npy', 'wb') as f:
    np.save(f, test)
