import numpy as np
import scipy as sp
from density2potential.core.exact_TISE import construct_H_sparse
from density2potential.utils.math import norm, normalise_function
from density2potential.utils.physics import calculate_density_exact
from density2potential.plot.animate import animate_function
import matplotlib.pyplot as plt

"""
Solve the exact time-dependent Schrodinger equation in 1D to generate exact evolved
wavefunction and density
"""

def solve_TDSE(params, wavefunction_initial):
    r"""
    Solves the time-dependent Schrodinger equation given an initial wavefunction and external potential v(x,t)
    """

    # Initilise density
    density = np.zeros((params.Ntime,params.Nspace))
    density[0,:] = calculate_density_exact(params, wavefunction_initial)
    wavefunction = np.copy(wavefunction_initial)

    # Time stepping defined by numpy's expm function
    if params.time_step_method == 'expm':

        # Construct the perturbed hamiltonian
        v_ext = np.copy(params.v_ext)
        params.v_ext = params.v_ext_td[1,:]
        hamiltonian = construct_H_sparse(params, basis_type='position')

        for i in range(1,params.Ntime):

            # If the perturbation to the external potential is dependent on time
            # updated H at each time-step
            if params.td_pert:

                params.v_ext = params.v_ext_td[i,:]
                hamiltonian = construct_H_sparse(params, basis_type='position')

            # Evolve the wavefunction
            wavefunction = expm_step(params, wavefunction, hamiltonian)

            # Compute the density
            density[i,:] = calculate_density_exact(params, wavefunction)

            # Renormalise the wavefunction
            wavefunction *= (np.sum(abs(wavefunction[:])**2)*params.dx**2)**-0.5

            # Calculate + normalise density
            density[i,:] = calculate_density_exact(params, wavefunction)

            print('Time passed: {}'.format(round(params.time_grid[i],3)), end='\r')

        params.v_ext = v_ext

    # Crank-Nicolson time-stepping
    elif params.time_step_method == 'CN':

        # Update external potential and construct corresponding hamiltonian
        params.v_ext = params.v_ext_td[1,:]
        hamiltonian = construct_H_sparse(params, basis_type='position')

        # Construct CN matrix A for CN's Ax=b equation
        CN_matrix = 0.5j*params.dt*hamiltonian
        identity = sp.sparse.csr_matrix(np.diag(np.ones(params.Nspace**2)))
        A = identity + CN_matrix

        for i in range(0,params.Ntime):

            # If the perturbation to the external potential is dependent on time
            # updated H at each time-step
            if params.td_pert:

                params.v_ext = params.v_ext_td[i,:]
                hamiltonian = construct_H_sparse(params, basis_type='position')

                wavefunction = crank_nicolson_step(params, wavefunction, hamiltonian)

            else:

                # Construct b for CN's Ax=b equation
                b = (identity - CN_matrix).dot(wavefunction)

                # Evolve the wavefunction
                wavefunction, status = sp.sparse.linalg.cg(A, b, x0=wavefunction, tol=1e-17, atol=1e-15)

            # Renormalise the wavefunction, potentially dubious
            wavefunction *= (np.sum(abs(wavefunction[:])**2) * params.dx**2)**-0.5

            # Calculate and renormalise density
            density[i,:] = calculate_density_exact(params, wavefunction)

            print('Time passed: {}'.format(round(params.time_grid[i],3)), end='\r')

    else:
        raise RuntimeError('Not a valid time-stepping method: {}'.format(params.time_step_method))

    return density


def expm_step(params, wavefunction, hamiltonian):
    r"""
    Update wavefunction based on numpy's expm function psi(t+dt) = e^iHt psi(t)
    """

    wavefunction = sp.sparse.linalg.expm_multiply(1.0j*params.dt*hamiltonian, wavefunction)

    return wavefunction


def crank_nicolson_step(params, wavefunction, hamiltonian):
    r"""
    Solve the CN matrix problem for the evolved wavefunction: Ax = b
    (I + 0.5idtH)psi(x,t+dt) = (I - 0.5idtH)psi(x,t)
    """

    CN_matrix = 0.5j*params.dt*hamiltonian
    identity = sp.sparse.csr_matrix(np.diag(np.ones(params.Nspace**2)))
    b = (identity - CN_matrix).dot(wavefunction)
    A = identity + CN_matrix

    wavefunction, status = sp.sparse.linalg.cg(A,b,x0=wavefunction)

    return wavefunction
