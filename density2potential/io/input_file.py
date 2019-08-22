import numpy as np

"""
Input to the calculation in the form of a parameters class
"""

class parameters(object):
    def __init__(self,*args,**kwargs):

        # Space
        self.Nspace = 301
        self.space = kwargs.pop('space',20)
        self.dx = self.space / (self.Nspace-1)

        # Time
        self.Ntime = 10
        self.time = kwargs.pop('time',0.01)
        self.dt = self.time / (self.Ntime-1)

        # N
        self.num_electrons = kwargs.pop('num_electrons',2)

        # Misc.
        self.stencil = kwargs.pop('stencil',9)

        # Grid
        self.space_grid = np.linspace(-0.5*self.space, 0.5*self.space, self.Nspace)
        self.time_grid = np.linspace(0, self.time, self.Ntime)

        # Ground state external potential (e.g. Gaussian and QHO respectively)
        #self.v_ext = -4.0 * np.exp(-0.2 * self.space_grid**2)
        #self.v_ext = 0.5*(0.25**2)*self.space_grid**2
        self.v_ext = 5e-11*self.space_grid**10 - 1.3e-4*self.space_grid**4

        # Shift the potential such that the eigenvalues are negative
        self.v_ext_shift = 0 #abs(2.0*np.amin(self.v_ext))
        self.v_ext += self.v_ext_shift

        # Time dependent external potential
        self.v_ext_td = np.zeros((self.Ntime,self.Nspace))
        self.v_ext_td[0,:] = self.v_ext
        self.v_pert = 0.1*self.space_grid
        for i in range(1,self.Ntime):
            self.v_ext_td[i,:] = self.v_ext[:] + self.v_pert[:]

        # Is the perturbing potential (v_pert) time-dependent?
        self.td_pert = True

        # Method for time-propagation (KS and exact)
        self.time_step_method = 'expm'
