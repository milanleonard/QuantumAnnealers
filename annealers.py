from abc import ABC, abstractmethod
import io
from schedulers import Scheduler, LinearScheduler
from qutip import identity, sigmaz, sigmax, Qobj, sesolve, mesolve, basis, tensor
import numpy as np


class QuantumAnnealer(ABC):
    """ Abstract class for Quantum Annealing algorithms
    
    params: sch"""

    def __init__(self, scheduler: Scheduler, qubo_problem: dict, num_qubits: int):
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
        one_qubit_ket = 1 / np.sqrt(2)* (basis(2,0) - basis(2,1))
        return tensor([one_qubit_ket]*self.num_qubits)
    
    def _inital_hamil(self):
        hamil = Qobj(dims=[[2]*self.num_qubits, [2]*self.num_qubits])
        for idx in range(self.num_qubits):
            hamil += self._ops_on_qubits([sigmax()], [idx])
        return hamil


    def _ops_on_qubits(self, ops, poses):
        """ Apply operation op to index poses and return the Hamiltonian """
        assert len(ops) == len(poses), "Must specify a position for each operator, and vice versa"
        tens_list = [identity(2)] * self.num_qubits
        for op, pos in zip(ops, poses):
            tens_list[pos] = op
        return tensor(tens_list)

    def _qubo_to_target_hamil(self):
        hamil = Qobj(dims=[[2]*self.num_qubits, [2]*self.num_qubits])
        for (l, r), value in self.qubo_problem.items():
            if l == r:
                hamil += self._ops_on_qubits([value*identity(2)],[l])
            else:
                hamil += self._ops_on_qubits([value*sigmaz(),sigmaz()], [l,r])
        return hamil

    def optimal_result(self):
        return np.real(self.target_hamil.get_data().min())

    @abstractmethod
    def anneal(self):
        pass
    

class QutipStateVectorAnnealer(QutipAnnealer):
    """ Statevector Quantum annealing algorithm using the QUTIP library using state-vector simulation.
        
    """

    def __init__(self, scheduler: Scheduler, qubo_problem, num_qubits):
        super().__init__(scheduler, qubo_problem, num_qubits)
        
    def anneal(self):
        args = self.scheduler.args()
        times = self.scheduler.create_times()

        Hlist = [[self.initial_hamil, self.scheduler.initial_hamil_timing_str()], [self.target_hamil, self.scheduler.target_hamil_timing_str()]]

        res = sesolve(Hlist, self.initial_state, times, args=args, e_ops = [self.target_hamil])

        return res

class QutipDensityMatrixAnnealer(QutipAnnealer):
    """ Statevector Quantum annealing algorithm using the QUTIP library using density matrix simulation 
    """

    def __init__(self, scheduler: Scheduler, qubo_problem, num_qubits):
        super().__init__(scheduler, qubo_problem, num_qubits)
        self.initial_state = self.initial_state * self.initial_state.dag()
        
    def anneal(self):
        args = self.scheduler.args()
        times = self.scheduler.create_times()

        Hlist = [[self.initial_hamil, self.scheduler.initial_hamil_timing_str()], [self.target_hamil, self.scheduler.target_hamil_timing_str()]]

        res = mesolve(Hlist, self.initial_state, times, args=args, e_ops = [self.target_hamil])

        return res








