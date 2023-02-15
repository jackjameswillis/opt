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
@jit(nopython=True, cache=True)
def update(V, W, i):

    return np.sign(V @ W[i])

# State energy function compiled to C
@jit(nopython=True, cache=True)
def E(V, W):

    return -((V.T @ W) @ V) / 2

# Types must be given for C
hopfield_decorators = [('N', int32),
                        ('W', double[:,:])]

# Types are given so that class code can be compiled in C
@jitclass(hopfield_decorators, cache=True)
class hopfield(object):

    def __init__(self, N, W):

        self.N = N

        self.W = W
    
    def E(self, V):

        return E(V, self.W)
    
    def update(self, V, i):

        return update(V, self.W, i)
    
    def relax(self, V, T):

        V_t = np.zeros((T, self.N))

        V_t[0] = V

        E_t = np.zeros(T)

        i_t = np.random.randint(0, self.N, T)

        for t in np.arange(1, T):

            V_t[t] = V_t[t-1]

            V_t[t, i_t[t]] = self.update(V_t[t], i_t[t])

            E_t[t] = self.E(V_t[t])

        return (V_t, E_t)
    
    def multiple_relax(self, V_r, T):

        V_r_t = np.zeros((len(V_r), T, self.N))

        E_r_t = np.zeros((len(V_r), T))

        for r in np.arange(len(V_r)):

            V_r_t[r], E_r_t[r] = self.relax(V_r[r], T)
        
        return (V_r_t, E_r_t)
    
'''==============================================
Multiple-Dimensional Knapsack Problem (mdkp)
=============================================='''
# Types must be given for C
mdkp_decorators = [('N', int32)]

# Types are given so that class code can be compiled in C
@jitclass(mdkp_decorators, cache=True)
class mdkp(object):

    def __init__(self, N):

        self.N = N