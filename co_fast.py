'''==============================================
Imports
=============================================='''
# C compilable maths
import numpy as np

# Makes lots of code C
from numba import int32, double, jit
from numba.experimental import jitclass

'''==============================================
hopfield
=============================================='''

# Update for individual units compiled to C
@jit(nopython=True)
def update(V, W, i):

    return np.sign(V @ W[i])

# State energy function compiled to C
@jit(nopython=True)
def E(V, W):

    return -((V.T @ W) @ V) / 2

# Types must be given for C
hopfield_decorators = [('N', int32),
                        ('W', double[:,:])]

# Types are given so that class code can be compiled in C
@jitclass(hopfield_decorators)
class hopfield(object):

    def __init__(self, N, W):

        self.N = N

        self.W = W
    
    def E(self, V):

        return E(V, self.W)
    
'''==============================================
Multiple-Dimensional Knapsack Problem (mdkp)
=============================================='''
# Types must be given for C
mdkp_decorators = [('N', int32)]

# Types are given so that class code can be compiled in C
@jitclass(mdkp_decorators)
class mdkp(object):

    def __init__(self, N):

        self.N = N