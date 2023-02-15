# C compilable maths
import numpy as np

# Makes lots of code C
from numba import int32, float32
from numba.experimental import jitclass

# Types must be given for C
hopfield_decorators = [('N', int32)]

# Types are given so that class code can be compiled in C
@jitclass(hopfield_decorators)
class hopfield(object):

    def __init__(self, N):

        self.N = N

# Types must be given for C
mdkp_decorators = [('N', int32)]

# Types are given so that class code can be compiled in C
@jitclass(mdkp_decorators)
class mdkp(object):

    def __init__(self, N):

        self.N = N