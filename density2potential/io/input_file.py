import numpy as np

"""
Input to the calculation in the form of a parameters class
"""

class parameters(object):
    def __init__(self,density_reference,*args,**kwargs):

        # Space
        #self.Nspace = np.ma.size(density_reference,1)
        self.Nspace = 41
        self.space = kwargs.pop('space',10)
        self.dx = self.space / (self.Nspace-1)

        # Time
        self.Ntime = 5000 #np.ma.size(density_reference,0)
        self.time =  kwargs.pop('time',5)
        self.dt = self.time / (self.Ntime-1)

        # N
        self.num_electrons = kwargs.pop('num_electrons',2)

        # Misc.
        self.stencil = kwargs.pop('stencil',5)

        # Grid
        self.space_grid = np.linspace(-0.5*self.space, 0.5*self.space, self.Nspace)
        self.time_grid = np.linspace(0,self.time, self.Ntime)

        # External potential (time indep and time dep)
        self.v_ext = 0.5*(0.25**2)*self.space_grid**2
        self.v_pert = 0.1*self.space_grid**2