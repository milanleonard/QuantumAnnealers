from abc import ABC, abstractmethod
from schedulers import Scheduler, LinearScheduler
import qutip
import numpy as np
import scipy


class QuantumAnnealer(ABC):
    """ Abstract class for Quantum Annealing algorithms
    
    params: scheduler"""

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
            tens_list[pos] = op
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

    def __init__(self, scheduler: Scheduler, qubo_problem, num_qubits):
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

        res = qutip.mesolve(Hlist, self.initial_state, times, args=args, e_ops = [self.target_hamil])

        return res








