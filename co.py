import numpy as np

class hillclimber:

  def __init__(self, N):

    self.N = N
  
  def climb(self, V, T, E):

    Vs = np.zeros((T, self.N))

    Es = np.zeros(T)

    VE = E(V)

    Vs[0] = V

    Es[0] = VE

    for t in range(1, T):

      V_ = V.copy()

      i = np.random.randint(0, self.N)

      V_[i] = 1 - V_[i]

      V_E = E(V_)

      if V_E <= VE:

        V = V_

        VE = V_E
      
      Vs[t] = V

      Es[t] = VE

    return (Vs, Es)

class PBIL:

  def __init__(self, N, lr, nlr):

    self.P = np.ones(N) - 0.5

    self.N = N

    self.lr = lr

    self.nlr
  
  def optimize(self, T, population_size, E):

    best_inds = np.zeros((self.T, self.N))

    best_inds_E = np.zeros(self.T)

    for t in range(T):

      population = (np.random.uniform(0, 1, (population_size, self.N)) <= self.P).astype(np.int)

      best_ind = population[0]

      best_ind_E = E(best_ind)

      worst_ind = population[0]

      worst_ind_E = E(worst_ind)

      for ind in population:

        ind_E = E(ind)

        if ind_E <= best_ind_E:

          best_ind = ind

          best_ind_E = ind_E
        
        if ind_E >= worst_ind_E:

          worst_ind = ind

          worst_ind_E = ind_E
        
      P = P * (1 - self.lr) + best_ind * self.lr

      neq = best_ind != worst_ind

      for i in range(self.N):

        if neq[i]:

          P[i] = P[i] * (1 - self.nlr) + best_ind[i] * self.nlr

    return (best_inds, best_inds_E)