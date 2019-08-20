import numpy as np

"""
Various Math Utilities
"""


def normalise_function(params, function):
    r"""
    Normalise a function using the continuum L2 norm
    """

    norm = 0
    for i in range(params.Nspace):
        norm += (np.conj(function[i]) * function[i]) * params.dx
    norm = norm ** -0.5
    function *= norm

    return function


def norm(params, function, norm_type):
    r"""
    Computes the norm of an input function

    norm_type : type of norm to be computed, string
    continous L2 norm: 'C2'
    discrete L2 norm: 'D2'
    continous L1 norm: 'C1'
    discrete L1 norm: 'D1'
    Mean absolute error (if diff between two functions is passed): 'MAE'
    """

    if (norm_type == 'C2'):
        norm = (np.sum( np.conj(function[:]) * function[:] )*params.dx)**0.5
    elif (norm_type == 'D2'):
        norm = np.linalg.norm(function)
    elif (norm_type == 'C1'):
        norm = np.sum(abs(function))*params.dx
    elif (norm_type == 'D1'):
        norm = np.linalg.norm(function,1)
    elif (norm_type == 'MAE'):
        norm = np.mean(np.abs(function))
    else:
        raise RuntimeError('Invalid norm type "{0}" used in norm()'.format(norm_type))

    return norm


def discrete_Laplace(params):
    r"""
    Returns a 2D array of the discretised Laplace operator using the domain and stencil given in params
    """

    if params.stencil == 3:
        # 3-point stencil for the Laplace operator
        laplace = -2.0*np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(np.ones(params.Nspace-1),1) + np.diag(np.ones(params.Nspace-1),-1)
        laplace *= 1.0 / params.dx**2
    elif params.stencil == 5:
        # 5-point stencil
        laplace = -30.0 * np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(16.0*np.ones(params.Nspace-1),1) + np.diag(16*np.ones(params.Nspace-1),-1)
        laplace += np.diag(-1.0*np.ones(params.Nspace-2),2) + np.diag(-1.0*np.ones(params.Nspace-2),-2)
        laplace *= 1.0 / (12.0 * params.dx**2)
    elif params.stencil == 7:
        # 7-point stencil
        laplace = -490*np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(270*np.ones(params.Nspace-1),1) + np.diag(270*np.ones(params.Nspace-1),-1)
        laplace += np.diag(-27*np.ones(params.Nspace-2),2) + np.diag(-27*np.ones(params.Nspace-2),-2)
        laplace += np.diag(2*np.ones(params.Nspace-3),3) + np.diag(2*np.ones(params.Nspace-3),-3)
        laplace *= 1.0 / (180.0 * params.dx**2)
    elif params.stencil == 9:
        # 9-point stencil
        laplace = -14350.0 * np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(8064*np.ones(params.Nspace-1),1) + np.diag(8064*np.ones(params.Nspace-1),-1)
        laplace += np.diag(-1008*np.ones(params.Nspace-2),2) + np.diag(-1008*np.ones(params.Nspace-2),-2)
        laplace += np.diag(128*np.ones(params.Nspace-3),3) + np.diag(128*np.ones(params.Nspace-3),-3)
        laplace += np.diag(-9*np.ones(params.Nspace-4),4) + np.diag(-9*np.ones(params.Nspace-4),-4)
        laplace *= 1.0 / (5040.0 * params.dx**2)
    elif params.stencil == 11:
        # 11-point stencil
        laplace = -73766.0 * np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(42000*np.ones(params.Nspace-1),1) + np.diag(42000*np.ones(params.Nspace-1),-1)
        laplace += np.diag(-6000*np.ones(params.Nspace-2),2) + np.diag(-6000*np.ones(params.Nspace-2),-2)
        laplace += np.diag(1000*np.ones(params.Nspace-3),3) + np.diag(1000*np.ones(params.Nspace-3),-3)
        laplace += np.diag(-125*np.ones(params.Nspace-4),4) + np.diag(-125*np.ones(params.Nspace-4),-4)
        laplace += np.diag(8*np.ones(params.Nspace-5),5) + np.diag(8*np.ones(params.Nspace-5),-5)
        laplace *= 1.0 / (25200.0 * params.dx**2)
    elif params.stencil == 13:
        # 13-point stencil
        laplace = -2480478.0 * np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(1425600*np.ones(params.Nspace-1),1) + np.diag(1425600*np.ones(params.Nspace-1),-1)
        laplace += np.diag(-222750*np.ones(params.Nspace-2),2) + np.diag(-222750*np.ones(params.Nspace-2),-2)
        laplace += np.diag(44000*np.ones(params.Nspace-3),3) + np.diag(44000*np.ones(params.Nspace-3),-3)
        laplace += np.diag(-7425*np.ones(params.Nspace-4),4) + np.diag(-7425*np.ones(params.Nspace-4),-4)
        laplace += np.diag(864*np.ones(params.Nspace-5),5) + np.diag(864*np.ones(params.Nspace-5),-5)
        laplace += np.diag(-50*np.ones(params.Nspace-6),6) + np.diag(-50*np.ones(params.Nspace-6),-6)
        laplace *= 1.0 / (831600.0 * params.dx**2)
    else:
        raise RuntimeError('Not a valid stencil')

    return laplace
