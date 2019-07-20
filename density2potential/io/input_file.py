import numpy as np

"""
Input to the calculation in the form of a parameters class
"""

class parameters(object):
    def __init__(self,density_reference,*args,**kwargs):

        self.Nspace = np.ma.size(density_reference,1)
        self.space =  kwargs.pop('space',10)
        self.dx =  self.space / (self.Nspace-1)

        self.Ntime = np.ma.size(density_reference,0)
        self.time =  kwargs.pop('time',10)
        self.dt = self.time / self.Ntime

        self.num_electrons = kwargs.pop('num_electrons',2)

        self.stencil = kwargs.pop('stencil',5)