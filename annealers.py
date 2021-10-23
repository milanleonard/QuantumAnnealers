from abc import ABC, abstractmethod
from schedulers import ClassicalScheduler, QuantumScheduler, QuboClassicalScheduler
import qutip
import numpy as np
import scipy
import numba
from typing import Union, Literal

def _construct_qubo_energy_function(problem: list):
   # @numba.njit
    def qubo_energy_function(bitstring: list[Union[Literal[0],Literal[1]]]):
        val = 0
        for l, r, weight in problem:
            val -= weight * bitstring[l] * bitstring[r]
        return val
    return qubo_energy_function

def _construct_ising_energy_function(problem: list):
  #  @numba.njit
    def ising_energy_function(bitstring: list[Union[Literal[0],Literal[1]]]):
        val = 0
        for l, r, weight in problem:
            val -= weight * (2*bitstring[l]-1) * (2*bitstring[r]-1)
        return val
    return ising_energy_function



class QuantumAnnealer(ABC):
    """ Abstract class for Quantum Annealing algorithms
    
    params: scheduler"""

    def __init__(self, scheduler: QuantumScheduler, qubo_problem: dict, num_qubits: int):
        self.scheduler = scheduler
        self.qubo_problem = qubo_problem
        self.num_qubits = num_qubits

    @abstractmethod
    def anneal(self):
        pass


class QutipAnnealer(QuantumAnnealer):
    def __init__(self, scheduler, qubo_problem, num_qubits):
        """ This class is responsible for taking in a QUBO problem and converting it into
            a target Hamiltonian that can easily be used for qutip, as well as setting up initial states etc.

            Still leaves the actual anneal method as abstract.
        """
        super().__init__(scheduler, qubo_problem, num_qubits)
        self.initial_state = self._Sx_eigenstate()
        self.initial_hamil = self._inital_hamil()
        self.target_hamil = self._qubo_to_target_hamil()

    def _Sx_eigenstate(self):
        one_qubit_ket = 1 / np.sqrt(2)* (qutip.basis(2,0) - qutip.basis(2,1))
        return qutip.tensor([one_qubit_ket]*self.num_qubits)
    
    def _inital_hamil(self):
        hamil = qutip.Qobj(dims=[[2]*self.num_qubits, [2]*self.num_qubits])
        for idx in range(self.num_qubits):
            hamil += self._ops_on_qubits([qutip.sigmax()], [idx])
        return hamil


    def _ops_on_qubits(self, ops, poses):
        """ Apply operation op to index poses and return the Hamiltonian """
        assert len(ops) == len(poses), "Must specify a position for each operator, and vice versa"
        tens_list = [qutip.identity(2)] * self.num_qubits
        for op, pos in zip(ops, poses):
            tens_list[pos-1] = op
        return qutip.tensor(tens_list)

    def _qubo_to_target_hamil(self):
        hamil = qutip.Qobj(dims=[[2]*self.num_qubits, [2]*self.num_qubits])
        for (l, r), value in self.qubo_problem.items():
            if l == r:
                hamil += self._ops_on_qubits([value*qutip.identity(2)],[l])
            else:
                hamil += self._ops_on_qubits([value*qutip.sigmaz(),qutip.sigmaz()], [l,r])
        return hamil

    def eigenspec_time_t(self, t, k=10):
        hamil = self.scheduler.initial_t(t) * self.initial_hamil +  self.scheduler.target_t(t) * self.target_hamil
        eigs, eigvecs = scipy.linalg.eig(hamil)
        return np.real(eigs), eigvecs

    def optimal_result(self):
        return np.real(self.target_hamil.get_data().min())

    @abstractmethod
    def anneal(self):
        pass
    

class QutipStateVectorAnnealer(QutipAnnealer):
    """ Statevector Quantum annealing algorithm using the QUTIP library using state-vector simulation.
    """

    def __init__(self, scheduler: QuantumScheduler, qubo_problem, num_qubits):
        super().__init__(scheduler, qubo_problem, num_qubits)
        self.res = None
        
    def anneal(self):
        args = self.scheduler.args()
        times = self.scheduler.create_times()

        Hlist = [[self.initial_hamil, self.scheduler.initial_hamil_timing_str()], [self.target_hamil, self.scheduler.target_hamil_timing_str()]]

        self.res = qutip.sesolve(Hlist, self.initial_state, times, args=args)
        self.expect = self._expect()    

    def _expect(self):
        return np.array([(state.dag()*self.target_hamil*state).get_data().toarray() for state in self.res.states]).flatten()

    def get_optimal(self):
        return self.optimal_result()

    


class QutipDensityMatrixAnnealer(QutipAnnealer):
    """ Statevector Quantum annealing algorithm using the QUTIP library using density matrix simulation 
    """

    def __init__(self, scheduler: QuantumScheduler, qubo_problem, num_qubits):
        super().__init__(scheduler, qubo_problem, num_qubits)
        self.initial_state = self.initial_state * self.initial_state.dag()
        
    def anneal(self):
        args = self.scheduler.args()
        times = self.scheduler.create_times()

        Hlist = [[self.initial_hamil, self.scheduler.initial_hamil_timing_str()], [self.target_hamil, self.scheduler.target_hamil_timing_str()]]
        dephasers = []
        for i in range(self.num_qubits):
            ident = [qutip.qeye(2)] * self.num_qubits
            ident[i] = 0.2 * qutip.sigmaz()
            dephasers.append(qutip.tensor(ident))

        self.res = qutip.mesolve(Hlist, self.initial_state, times, dephasers, args=args, e_ops = [self.target_hamil])

    def get_optimal(self):
        return self.optimal_result()
    
    def get_expectation_val(self):
        return self.res.expect[0][-1]



class CombinatorialAnnealer(ABC):
    """ Abstract class for Quantum Annealing algorithms
    
    params: scheduler"""

    def __init__(self, scheduler: ClassicalScheduler, num_nodes: int):
        self.scheduler = scheduler
        self.num_nodes = num_nodes

    @abstractmethod
    def anneal(self):
        pass


class MetropolisAnnealer(CombinatorialAnnealer):
    def __init__(self, scheduler: QuboClassicalScheduler, qubo_problem, num_nodes, isIsing=True):
        super().__init__(scheduler=scheduler, num_nodes=num_nodes)
        self.problem = [(l,r, value) for (l,r), value in qubo_problem.items()]
        self.state = np.random.randint(2, size=self.num_nodes, dtype = np.uint8)
        if isIsing:
            self.energy_function = _construct_ising_energy_function(self.problem)
        else:
            self.energy_function = _construct_qubo_energy_function(self.problem)
        self.curr_energy = self.energy_function(self.state)
    
    def anneal(self, verbose=False):
        while self.scheduler.step():
            if verbose:
                print(self.curr_energy)
            flip_bits = self.scheduler.flip_which_bits(self.num_nodes)
            proposal_state = self.state.copy()
            proposal_state[flip_bits] = 1^self.state[flip_bits]
            if ((new_energy := self.energy_function(proposal_state)) < self.curr_energy):
                self.state = proposal_state
                self.curr_energy = new_energy
            else:
                temp = self.scheduler.temp
                energy_diff = self.curr_energy - new_energy
                rand = np.random.uniform(0,1)
                rand_compare = np.exp(energy_diff/temp)
                if rand < rand_compare:
                    self.state = proposal_state
                    self.curr_energy = new_energy
        print(f"final energy of {self.curr_energy}")







        





