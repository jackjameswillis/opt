import numpy as np

class TSP:

    def __init__(self, C, symmetric=True):

        self.C = C

        self.symmetric = symmetric
    
        self.idx_length = np.sqrt(C).astype(np.int)

        self.tour_string_length = (C * self.idx_length).astype(np.int)

        roads = np.random.uniform(low=0, high=1, size=(C,C))

        diag = 1 - np.diag(np.ones(C))

        self.roads = roads * roads.T * diag
    
    def tour_length(self, tour_string):

        indexes = np.zeros(self.C)

        for i in range(self.C):

            bits = tour_string[i * self.idx_length: (i + 1) * self.idx_length]

            indexes[i] = bits.dot(2**np.arange(bits.size)[::-1])
        
        order = np.argsort(indexes)

        tour_length = sum([self.roads[order[i-1], order[i]] for i in range(1, self.C)])

        return 1/tour_length

class MDKP:

    def __init__(self, L, D, N, a):

        self.D = D

        self.N = N

        self.a = a

        self.sizes = np.random.randint(0, L, [N] + [self.D])

        self.masses = np.random.randint(1, N, N)

        self.S = [(a * np.sum(self.sizes[:,i])).astype(np.int32) for i in range(D)]
    
    def fill_knapsack(self, V):

        filled = np.zeros(self.D)

        S_a = np.array(self.S)

        chosen_sizes = (V * self.sizes.T)

        chosen_masses = V * self.masses

        mass = 0

        for d in range(self.D):

            filled[d] = np.sum(chosen_sizes[d])

        if not (filled <= S_a).all():

            return 0
        
        return np.sum(chosen_masses)