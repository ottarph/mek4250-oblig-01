import numpy as np

class ex01_sol:

    def __init__(self, mu_val):
        self.k = 1 / mu_val
        self.mu = mu_val

    def __call__(self, x):
        if self.mu < 0.01:
            return self.__specialized_call__(x)
        else:
            return self.__generic_call__(x)
        
    def __generic_call__(self, x):
        """ `np.expm(x) = np.exp(x) - 1`. """
        values = np.expm1(x[0] * self.k) / np.expm1(self.k)
        return values
    
    def __specialized_call__(self, x_in):
        """ A hacky implementation to avoid numerical issues
        when caused by dividing two extremely large numbers.
        Fairly accurate when k is large. 
        
        Cut off-values `cut_low`and `cut_high` not tweaked much.
        `cut_low` chosen such that `u(eps,0)` should evaluate to `0`.
        `cut_high` chosen such that the difference between
        `np.exp(k*x)` and  `np.expm(k*x)` should be insignificant
        in floating point representation. """
        cut_low = 0.002
        cut_high = 10

        x = x_in[0,:]
        values = np.full(x.shape[-1], np.nan)

        values[self.k*x < cut_low] = 0.0

        inds = self.k*x >= cut_low

        la = np.zeros_like(values[inds])
        lb = np.zeros_like(values[inds])

        inds2 = self.k*x[inds] > cut_high
        inds3 = self.k*x[inds] <= cut_high
        la[inds2] = self.k*x[inds][inds2]
        la[inds3] = np.log(np.expm1(self.k*x[inds][inds3]))

        if self.k > cut_high:
            lb = self.k
        else:
            lb = np.log(np.expm1(self.k))

        values[inds] = np.exp(la - lb)

        return values
