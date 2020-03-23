"""
Linear response properties of time-independent and time-dependent QM
"""

import numpy as np


def two_particle_susceptibility(params, wavefunction, energy):
    r"""
    The linear response function \chi(x,x') = d v_ext / d rho, calculated from wavefunctions of H

    :param params: parameters
    :param wavefunction: all excited state wavefunctions of H -- expanded
    :param energy: eigenenergies
    :return: exact response \chi
    """

    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)

    for n in range(1, 101):

        # Placeholder vectors for the matrix elements of \chi
        f, g = np.zeros(params.Nspace, dtype=complex), np.zeros(params.Nspace, dtype=complex)

        # Construct matrix elements <psi|n(x)|psi>
        for row in range(params.Nspace):

            f[row] = np.dot(wavefunction[0,row,:].conj(), wavefunction[n,row,:])
            g[row] = np.dot(wavefunction[n,row,:].conj(), wavefunction[0,row,:])

        # Construct susceptibility matrix
        delta_E = energy[0] - energy[n]
        susceptibility += (8.0 / delta_E) * np.outer(f, g)
        #print(8.0 / delta_E)


    return susceptibility
