import numpy as np
import scipy as sp
from density2potential.core.exact_TISE import construct_H_sparse, pack_wavefunction, unpack_wavefunction
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

    # Save the time-dependent external potential
    np.save('TD_external_potential', params.v_ext_td)

    # Initilise density
    density = np.zeros((params.Ntime,params.Nspace))
    density[0,:] = 2.0 * (np.sum(abs(wavefunction_initial[:,:])**2,axis=0)) * params.dx
    wavefunction = np.copy(unpack_wavefunction(params, wavefunction_initial))

    # Time stepping defined by numpy's expm function
    if params.time_step_method == 'expm':

        for i in range(1,params.Ntime):
            # Construct the hamiltonian for this time interval
            params.v_ext = params.v_ext_td[i,:]
            hamiltonian = construct_H_sparse(params)

            # Evolve the wavefunction
            wavefunction = expm_step(params, wavefunction, hamiltonian)
            density[i,:] = np.sum(abs(pack_wavefunction(params,wavefunction)[:,:])**2, axis=0)

            print('Time passed: {}'.format(round(params.time_grid[i],3)))

    # Crank-Nicolson time-stepping
    elif params.time_step_method == 'CN':

        for i in range(0,params.Ntime):
            print(i)

    else:
        raise RuntimeError('Not a valid time-stepping method: {}'.format(params.time_step_method))

    np.save('TD_density', density)

    return density


def expm_step(params, wavefunction, hamiltonian):

    wavefunction = sp.sparse.linalg.expm_multiply(1.0j*params.dt*hamiltonian, wavefunction)

    return wavefunction

