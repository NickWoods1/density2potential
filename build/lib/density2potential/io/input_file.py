import numpy as np

"""
Input to the calculation in the form of a parameters class
"""

class parameters(object):
    def __init__(self,density_reference,*args,**kwargs):

        # Space
        self.Nspace = 61 #(density_reference,1)
        self.space = kwargs.pop('space',10)
        self.dx = self.space / (self.Nspace-1)

        # Time
        self.Ntime = 1000 #np.ma.size(density_reference,0)
        self.time = kwargs.pop('time',5)
        self.dt = self.time / (self.Ntime-1)

        # N
        self.num_electrons = kwargs.pop('num_electrons',2)

        # Misc.
        self.stencil = kwargs.pop('stencil',3)

        # Grid
        self.space_grid = np.linspace(-0.5*self.space, 0.5*self.space, self.Nspace)
        self.time_grid = np.linspace(0, self.time, self.Ntime)

        # Ground state external potential
        #self.v_ext = np.zeros(self.Nspace)
        #self.v_ext = -1.0 * np.exp(-0.01 * self.space_grid**2)
        self.v_ext = 0.5*(0.25**2)*self.space_grid**2
        #self.v_ext = 0.1*self.space_grid

        # Shift the potential such that the eigenvalues are negative
        self.v_ext_shift = np.amax(self.v_ext) + 10
        self.v_ext -= self.v_ext_shift

        # Time dependent external potential
        self.v_ext_td = np.zeros((self.Ntime,self.Nspace))
        self.v_ext_td[0,:] = self.v_ext
        for i in range(1,self.Ntime):
            self.v_ext_td[i,:] = self.v_ext[:] + 0.1*self.space_grid

        # Method for time-propagation (KS and exact)
        self.time_step_method = 'CN'