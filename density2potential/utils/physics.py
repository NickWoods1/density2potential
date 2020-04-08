import numpy as np

"""
Various physics utils, e.g. density/energy computation.
"""


def calculate_density_exact(params, wavefunction):
    r"""
    Calculates the electron density given a wavefunction under the coordinate mapping: (x_1, x_2, ... x_N) ---> x
    used in this package.
    """

    # TODO There is a mistake in this definition...
    """ 
    density = np.zeros((params.Nspace)) 
    if params.num_electrons == 1:

        density[:] = abs(wavefunction[:])**2

    elif params.num_electrons == 2:

        for i in range(0,params.Nspace):
            l_bound = int(params.Nspace*i)
            u_bound = int(params.Nspace*(i + 1) - 1)
            density[i] = np.sum(abs(wavefunction[l_bound:u_bound])**2) * params.dx

    density *= 2
    #density *= params.num_electrons*(np.sum(density[:]) * params.dx)**-1.0
    """

    wavefunction = pack_wavefunction(params, wavefunction)
    density = np.zeros(params.Nspace)
    if params.num_electrons == 1:
        density[:] = abs(wavefunction[:]) ** 2
    else:
        density = np.sum(abs(wavefunction[:,:])**2, axis=1)
        density *= 2*params.dx

    return density


def calculate_density_ks(params, wavefunctions_ks):
    r"""
    Calculates the KS particle density given some KS wavefunctions [space, orbital #]
    """

    density = np.zeros(params.Nspace)

    if params.num_electrons == 1:

        density[:] = abs(wavefunctions_ks[:,0])**2

    elif params.num_electrons > 1:

        density = np.sum(abs(wavefunctions_ks[:,:])**2, axis=1)

    density *= params.num_electrons*(np.sum(density[:]) * params.dx)**-1.0

    return density


def pack_wavefunction(params,wavefunction):
    """
    Turns an N**2 1D array wavefunction into a NxN 2D array wavefunction given some mapping NxNxNxN ----> N^2 x N^2
    TODO REMOVE THIS....
    """

    wavefunction_packed = np.zeros((params.Nspace,params.Nspace), dtype=np.complex)
    i, j, k = 0, params.Nspace, 0
    while j<= params.Nspace**2:
        wavefunction_packed[:,k] = wavefunction[i:j]
        k += 1
        i += params.Nspace
        j += params.Nspace

    return wavefunction_packed

