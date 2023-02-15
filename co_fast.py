import numpy as np
from numba import int32, float32
from numba.experimental import jitclass

algorithm_decorators = [('N', int32)]

problem_decorators = [('N', int32)]

@jitclass(algorithm_decorators)
class algorithm(object):

    def __init__(self, N):

        self.N = N


hopfield_decorators = [('N', int32)]

@jitclass(hopfield_decorators)
class hopfield(algorithm):

    def __init__(self, N):

        super().__init__(N)

@jitclass(problem_decorators)
class problem(object):

    def __init__(self, N):

        self.N = N

mdkp_decorators = [('N', int32)]

@jitclass(mdkp_decorators)
class mdkp(problem):

    def __init__(self, N):

        super().__init__(N)