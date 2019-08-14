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


def norm(params,function,norm_type):
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
        norm = (np.sum( np.conj(function[:]) * function[:] )*params.dx**2)**0.5
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
    elif params.stencil == 9:
        # 9-point stencil
        laplace = -1430.0 * np.eye(params.Nspace, dtype=np.float)
        laplace += np.diag(8064*np.ones(params.Nspace-1),1) + np.diag(8064*np.ones(params.Nspace-1),-1)
        laplace += np.diag(-1008*np.ones(params.Nspace-2),2) + np.diag(-1008*np.ones(params.Nspace-2),-2)
        laplace += np.diag(128*np.ones(params.Nspace-3),3) + np.diag(128*np.ones(params.Nspace-3),-3)
        laplace += np.diag(-9*np.ones(params.Nspace-4),4) + np.diag(-9*np.ones(params.Nspace-4),-4)
        laplace *= 1.0 / (5040.0 * params.dx**2)
    else:
        raise RuntimeError('Not a valid stencil')

    return laplace


def calculate_density_exact(params, wavefunction):
    r"""
    Calculates the electron density given a wavefunction under the coordinate mapping: (x_1, x_2, ... x_N) ---> x
    used in this package.
    """

    density = np.zeros((params.Nspace))

    if params.num_electrons == 1:

        density[:] = np.sum(abs(wavefunction[:])**2) * params.dx

    elif params.num_electrons == 2:

        for i in range(0,params.Nspace):
            l_bound = int(params.Nspace*i)
            u_bound = int(params.Nspace*(i + 1) - 1)
            density[i] = 2.0 * np.sum(abs(wavefunction[l_bound:u_bound])**2) * params.dx

    return density

