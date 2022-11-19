import numpy as np

class DeepHopfield:

    def __init__(self, N, H, lr):

        self.N = N

        self.H = H

        self.NH = N + H

        self.W = np.random.choice((0.0001, -0.0001), (self.NH, self.NH))

        self.diag = 1 - np.diag(np.ones(self.NH))

        self.W = (self.W @ self.W.T) * self.diag

    def hebb(self, V):

        V = V[:, np.newaxis]

        dW = (V @ V.T) * self.lr

        self.W = (self.W + dW) * self.diag

    def E(self, V):

        V = V[:, np.newaxis]

        return -((V.T @ self.W) @ V) / 2
    
    def relax(self, V, T, f):

        Vs = np.zeros((T, self.NH))

        Es = np.zeros(T)

        i_t = np.random.randint(0, self.NH, T)

        VE = self.E(V) + f(V[:self.N])

        Vs[0] = V

        Es[0] = VE

        for t in range(T):

            V_ = V.copy()

            V_[i_t[t]] = V_[i_t[t]] * -1

            V_E = self.E(V) + f(V[:self.N])

            if V_E <= VE:

                V = V_

                VE = V_E
            
            Vs[t] = V

            Es[t] = VE
        
        self.hebb(V)
    
    def multiple_relax(self, T, R, f):

        runs = {}

        for r in range(R):

            runs[r] = self.relax(np.random.choice((-1, 1), self.NH), T = T, f = f)
        
        return runs
    
    def multiple_relax_constant(self, constant, T, R, f):

        runs = {}

        for r in range(R):

            runs[r] = self.relax(constant, T, f)
        
        return runs