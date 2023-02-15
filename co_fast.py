'''==============================================
Imports
=============================================='''
# C compilable maths
import numpy as np

# Makes lots of code C
from numba import int32, float32, double, jit
from numba.experimental import jitclass

'''==============================================
hopfield
=============================================='''

# Types must be given for C
# Reads return_type(given from V_type, W_type, i_type)
#update_signature = int32(int32[:], double[:,:], int32)

# Update for individual units compiled to C
@jit(nopython=True)
def update(V, W, i):

    return np.sign(V[i] @ W[i])

# Types must be given for C
hopfield_decorators = [('N', int32),
                        'W', double[:,:]]

# Types are given so that class code can be compiled in C
@jitclass(hopfield_decorators)
class hopfield(object):

    def __init__(self, N, W):

        self.N = N

    
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