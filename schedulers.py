from abc import ABC, abstractmethod
import numpy as np

class Scheduler(ABC):
    """ Scheduler for Quantum Annealing algorithms

    params: total_time, the total time for the quantum annealing scheduler
    """
    def __init__(self, total_time : float, num_timesteps: int):
        self.total_time = total_time
        self.num_timesteps = num_timesteps

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def initial_hamil_timing_str(self) -> str:
        pass

    @abstractmethod
    def target_hamil_timing_str(self) -> str:
        pass

    @abstractmethod
    def args(self):
        pass

    def create_times(self):
        return np.linspace(0, self.total_time, self.num_timesteps) # Currently making this depend on numpy, should be fine

class LinearScheduler(Scheduler):
    """ Linear time scheduling, normalizes to 
    
    """
    def __init__(self, total_time : float, num_timesteps):
        super().__init__(total_time, num_timesteps)

    def __str__():
        return '1/total_time * t + (1-1/total_time) * t'

    def initial_hamil_timing_str(self):
        return 'total_time-t'

    def target_hamil_timing_str(self):
        return 't'

    def args(self):
        return {"total_time":self.total_time}




