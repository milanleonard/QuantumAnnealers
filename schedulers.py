from abc import ABC, abstractmethod
import numpy as np

class QuantumScheduler(ABC):
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

class LinearScheduler(QuantumScheduler):
    """ Linear time scheduling, normalizes to 
    
    """
    def __init__(self, total_time : float, num_timesteps):
        super().__init__(total_time, num_timesteps)

    def initial_t(self, t):
        return 1 - 1 / self.total_time * t

    def target_t(self, t):
        return t / self.total_time 

    def __str__():
        return '1/total_time * t + (1-1/total_time) * t'

    def initial_hamil_timing_str(self):
        return 'total_time-t'

    def target_hamil_timing_str(self):
        return 't'

    def args(self):
        return {"total_time":self.total_time}


class ClassicalScheduler(ABC):
    def __init__(self, total_time):
        self.total_time=total_time

class QuboClassicalScheduler(ClassicalScheduler):
    def __init__(self, total_time, prob_of_n_changes_list: list[float], initial_temp, seed=None, dt = 0.01):
        """
        params:
            prob_of_n_changes_list : list[float], the probability of performing index+1 changes. e.g. [0.7,0.2,0.1]
                there is a 70% chance of flipping (0+1) bits, 20% chance to flip 2, 10% chance to flip 3
        """
        assert dt > 0, "Timesteps must be positive"
        if seed is not None:
            np.random.seed(seed)
        super().__init__(total_time)
        self.prob_changes = prob_of_n_changes_list
        assert np.isclose(sum(prob_of_n_changes_list), 1), "The sum of the changes must add to one" 
        self.time = 0
        self.temp = initial_temp
        self.dt = dt
        self.temp_step = self.temp / (self.total_time/dt)

    def flip_which_bits(self, num_bits) -> list[int]:
        random_value = np.random.uniform(0,1)
        prob_so_far = 0
        for idx, prob in enumerate(self.prob_changes):
            if random_value < prob_so_far:
                return np.random.choice(np.arange(num_bits), size=idx+1)
            prob_so_far += prob

    def step(self):
        self.time += self.dt
        if self.time >= self.total_time:
            return False
        self.temp -= self.temp_step
        return True



