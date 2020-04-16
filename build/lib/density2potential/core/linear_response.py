"""
Linear response properties of time-independent and time-dependent QM
"""

import numpy as np
import matplotlib.pyplot as plt
import findiff


def ks_susceptibility(params, eigenfunctions, eigenenergies):
    """
    KS linear response function
    """

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

    return susceptibility


def exact_susceptibility(params, wavefunction, energy):
    r"""
    The exact density linear response function \chi(x,x') = d v_ext / d rho, calculated from wavefunctions of H
    in the ground state
    """

    susceptibility = np.zeros((params.Nspace, params.Nspace), dtype=complex)
    num_states = len(energy)
    for n in range(1, num_states):

        # Placeholder vectors for the matrix elements of \chi
        f, g = np.zeros(params.Nspace, dtype=complex), np.zeros(params.Nspace, dtype=complex)

        # Construct matrix elements <psi|n(x)|psi>
        for row in range(params.Nspace):

            f[row] = np.dot(wavefunction[0,row,:].conj(), wavefunction[n,row,:])*params.dx
            g[row] = np.dot(wavefunction[n,row,:].conj(), wavefunction[0,row,:])*params.dx

        # Construct susceptibility matrix
        delta_E = energy[0] - energy[n]
        susceptibility += (8.0 / delta_E) * np.outer(f, g)

    return susceptibility


def xc_kernel(params, susceptibility_exact, susceptibility_ks):
    """
    Compute f_xc = chi^-1 - chi_0^-1 - v_int
    """

    chi_inv = np.linalg.pinv(susceptibility_exact)
    chi0_inv = np.linalg.pinv(susceptibility_ks)

    f_xc = (chi0_inv - chi_inv)*(1/params.dx**2) - params.v_int

    return f_xc


def dyson(params, chi_ks, f_xc):
    """
    Solve Dyson equation to recover chi from f_xc and chi0
    """

    identity = np.eye(params.Nspace)
    f_hxc = params.v_int + f_xc
    A = identity - chi_ks @ (f_hxc*params.dx**2)
    chi_dyson = np.linalg.solve(A, chi_ks)

    return chi_dyson


def sum_rule(params, f_xc, density_gs, vxc_gs, i):
    """ Check various exact conditions, see Ullrich TDDFT book """

    N = params.Nspace

    # One dimensional derivative operator
    d_dx = findiff.FinDiff(0, params.dx, acc=10)

    # Check Zero-Force theorem on ground-state XC potential
    grad_vxc = d_dx(vxc_gs)
    error = np.dot(grad_vxc, density_gs)*params.dx
    print('Error in the ground-state zero-force theorem: {}'.format(error))

    # Check weakest ZFSR: (8.12) in Ullrich TTDFT.
    error_weak = 0
    grad_den = d_dx(density_gs)
    for j in range(N):
        for k in range(N):
            error_weak += f_xc[j,k]*grad_den[j]*density_gs[k]*params.dx**2
    print('Weak ZFSR error is {}'.format(error))

    # Check strong ZFSR
    tmp = np.zeros(N, dtype=complex)
    for k in range(N):
        tmp[k] = np.sum(f_xc[k,:]*grad_den[:])*params.dx
    error_strong = np.sum(abs(tmp - grad_vxc))*params.dx
    print('Strong ZFSR error is {}'.format(error))

    # Plot...
    plt.clf()
    plt.title('Weak ZFSR error: {:0.3e}. Strong ZFSR error: {:1.3e}'.format(error_weak, error_strong))
    plt.plot(tmp.real, label='int grad(n) f_xc')
    plt.plot(grad_vxc.real, label='grad(v_xc)')
    plt.legend()
    plt.savefig(f'ZFSR_{i}.png')
