import numpy as np

"""
Various physics utils, e.g. density/energy computation.
"""


def calculate_density_exact(params, wavefunction):
    r"""
    Calculates the electron density given a wavefunction under the coordinate mapping: (x_1, x_2, ... x_N) ---> x
    used in this package.
    """

    density = np.zeros((params.Nspace))

    if params.num_electrons == 1:

        density[:] = abs(wavefunction[:])**2

    elif params.num_electrons == 2:

        for i in range(0,params.Nspace):
            l_bound = int(params.Nspace*i)
            u_bound = int(params.Nspace*(i + 1) - 1)
            density[i] = np.sum(abs(wavefunction[l_bound:u_bound])**2) * params.dx

    density *= params.num_electrons*(np.sum(density[:]) * params.dx)**-1.0

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


def element_charges():

    # dict of element charges

    elements = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5,
                'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
                }

    return elements

