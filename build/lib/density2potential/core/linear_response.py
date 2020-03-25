"""
Linear response properties of time-independent and time-dependent QM
"""

import numpy as np
import matplotlib.pyplot as plt

def ks_susceptibility(params, eigenfunctions, eigenenergies):

    num_occ = params.num_electrons
    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)

    # Sum over occupied/unoccupied pairs
    for i in range(num_occ):
        for j in range(num_occ, params.Nspace):
            # Difference in energies between occ/unocc states
            delta_energy = eigenenergies[i] - eigenenergies[j]

            f = eigenfunctions[:, i].conj() * eigenfunctions[:, j]
            g = eigenfunctions[:, j].conj() * eigenfunctions[:, i]
            susceptibility[:, :] += 2 * np.outer(f, g) / delta_energy

    """
    eigv, eigf = np.linalg.eigh(susceptibility)
    print(eigv)
    for i in range(len(eigv)):
        plt.plot(eigf[:,-i].real)
        plt.show()
    """

    return susceptibility

def two_particle_susceptibility(params, wavefunction, energy):
    r"""
    The linear response function \chi(x,x') = d v_ext / d rho, calculated from wavefunctions of H
    in the ground state

    :param params: parameters
    :param wavefunction: all excited state wavefunctions of H -- expanded in NxN form
    :param energy: eigenenergies
    :return: exact response \chi
    """

    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)
    num_states = params.Nspace
    for n in range(1, num_states):

        # Placeholder vectors for the matrix elements of \chi
        f, g = np.zeros(params.Nspace, dtype=complex), np.zeros(params.Nspace, dtype=complex)

        # Construct matrix elements <psi|n(x)|psi>
        for row in range(params.Nspace):

            f[row] = np.dot(wavefunction[0,row,:].conj(), wavefunction[n,row,:])
            g[row] = np.dot(wavefunction[n,row,:].conj(), wavefunction[0,row,:])

        # Construct susceptibility matrix
        delta_E = energy[0] - energy[n]
        susceptibility += (8.0 / delta_E) * np.outer(f, g)

    return susceptibility
