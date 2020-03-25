import numpy as np
import scipy as sp
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
from density2potential.plot.animate import animate_function, animate_two_functions
from density2potential.utils.math import norm, normalise_function, discrete_Laplace
from density2potential.utils.physics import calculate_density_ks
from density2potential.core.linear_response import ks_susceptibility

"""
Functions that contain the core functionality to find a Kohn-Sham potential given a density
"""


def generate_ks_potential(params,density_reference):
    r"""
    Reverse engineers a reference density to product a Kohn-Sham potential

    :param params: input parameters object
    :param density_reference: ndarray, the reference density, axes [time,space]
    :return: density_ks, v_ks, wavefunctions_ks: ndarray, the optimised density, Kohn-Sham potential and wavefunctions
    """

    # Deal with ground state before time-dependence
    # Init variables
    v_ks = np.zeros((params.Ntime,params.Nspace))
    density_ks = np.zeros((params.Ntime,params.Nspace))
    wavefunctions_ks = np.zeros((params.Ntime,params.Nspace,params.num_electrons), dtype=complex)

    # Initial guess for the Kohn-Sham potential
    v_ks[0,:] = params.v_ext
    v_ks[1:,:] = params.v_ext + params.v_pert

    # Compute the ground-state Kohn-Sham potential
    i, error = 0, 1
    while(error > 5e-10):

        # Construct Hamiltonian for a given Kohn-Sham potential
        hamiltonian = construct_H(params,v_ks[0,:])

        # Find eigenvectors and eigenvalues of this Hamiltonian
        eigenenergies_ks, eigenfunctions_ks = sp.linalg.eigh(hamiltonian)
        eigenfunctions_ks = eigenfunctions_ks.real

        # Store the lowest N_electron eigenfunctions as wavefunctions
        wavefunctions_ks[0,:,0:params.num_electrons] = eigenfunctions_ks[:,0:params.num_electrons]

        # Normalise the wavefunctions w.r.t the continous L2 norm
        wavefunctions_ks[0,:,0:params.num_electrons] = normalise_function(params,
                                                                          wavefunctions_ks[0,:,0:params.num_electrons])

        # Construct KS density
        density_ks[0,:] = calculate_density_ks(params, wavefunctions_ks[0,:,:])

        # Error in the KS density away from the reference density
        error = norm(params,density_ks[0,:] - density_reference[0,:],'MAE')

        # Update the KS potential with a steepest descent scheme
        v_ks[0,:] -= 0.01*(density_reference[0,:] - density_ks[0,:]) / density_reference[0,:]
        #v_ks[0,:] -= density_reference[0,:]**0.05 - density_ks[0,:]**0.05

        print('Error = {0} after {1} iterations'.format(error,i), end='\r')

        i += 1

    print('Final error in the ground state KS density is {0} after {1} iterations'.format(error,i))
    print(' ')


    # Compute the ground-state potential using a scipy optimiser
    opt_info = root(groundstate_objective_function, v_ks[0, :], args=(params, wavefunctions_ks[:,:,:], density_reference[0,:],
                                                                      ), method='hybr', tol=1e-16)

    # Output v_ks
    v_ks[0,:] = opt_info.x

    # Compute the corresponding wavefunctions, density, and error
    hamiltonian = construct_H(params, v_ks[0, :])
    eigenenergies_ks, eigenfunctions_ks = sp.linalg.eigh(hamiltonian)
    wavefunctions_ks[0,:,0:params.num_electrons] = normalise_function(params,eigenfunctions_ks[:,0:params.num_electrons])
    density_ks[0, :] = calculate_density_ks(params, wavefunctions_ks[0,:,:])
    error = norm(params, density_ks[0, :] - density_reference[0, :], 'MAE')
    print('Final root finder error = {0} after {1} function evaluations. Status: {2}'.format(error,opt_info.nfev,opt_info.success))
    print(' ')

    # Now optimise the time-dependent KS potential
    for i in range(1,params.Ntime):

        # Find the v_ks that minimises the specified objective function
        opt_info = root(evolution_objective_function,v_ks[i,:],args=(params,wavefunctions_ks[i-1,:,:],density_reference[i,:],
                                                                     'root', 'expm'), method='hybr', tol=1e-16)

        # Final (optimal) Kohn-Sham potential at time step i
        v_ks[i,:] = opt_info.x

        # Compute the evolved wavefunctions given the optimal Kohn-Sham potential
        if params.time_step_method == 'CN':
            wavefunctions_ks[i,:,:] = crank_nicolson_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        elif params.time_step_method == 'expm':
            wavefunctions_ks[i,:,:] = expm_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])

        # Final Kohn-Sham density
        density_ks[i,:] = calculate_density_ks(params, wavefunctions_ks[i,:,:])

        # Final error in the Kohn-Sham density away from the reference density
        error = norm(params, density_ks[i,:] - density_reference[i,:], 'MAE')

        print('Final error in KS density is {0} at time {1} after {2} iterations'.format(error,
                                                                                  round(params.time_grid[i],3),
                                                                                  opt_info.nfev), end='\r')

    print(' ')
    print(' ')

    return density_ks, v_ks, wavefunctions_ks


def groundstate_objective_function(v_ks,params,wavefunctions_ks,density_reference):
    r"""
    Ground state objective function, the root of which is the Kohn-Sham potential that generates a ground
    state reference density
    """

    # Construct Hamiltonian for a given Kohn-Sham potential
    hamiltonian = construct_H(params, v_ks[:])

    # Find eigenvectors and eigenvalues of this Hamiltonian
    eigenenergies_ks, eigenfunctions_ks = sp.linalg.eigh(hamiltonian)

    # Store the lowest N_electron eigenfunctions as wavefunctions
    wavefunctions_ks[0, :, 0:params.num_electrons] = eigenfunctions_ks[:, 0:params.num_electrons]

    # Normalise the wavefunctions w.r.t the continous L2 norm
    wavefunctions_ks[0, :, 0:params.num_electrons] = normalise_function(params,wavefunctions_ks[0, :, 0:params.num_electrons])

    # Construct KS density
    density_ks = np.sum(np.abs(wavefunctions_ks[0, :, :])**2, axis=1, dtype=np.float)

    return density_reference[:] - density_ks[:]


def evolution_objective_function(v_ks,params,wavefunctions_ks,density_reference,objective_type,evolution_type):
    r"""
    The objective function for root finding and optimisation algorithms, defined with some
    method of evolving psi(t) --> psi(t+dt).
    """

    # Evolve the wavefunctions according to the given scheme
    if (evolution_type == 'CN'):
        wavefunctions_ks = crank_nicolson_step(params,v_ks,wavefunctions_ks)
    elif (evolution_type == 'expm'):
        wavefunctions_ks = expm_step(params,v_ks,wavefunctions_ks)
    else:
        raise RuntimeError('Invalid time evolution method specified')

    wavefunctions_ks[:,0:params.num_electrons] = normalise_function(params,wavefunctions_ks[:,0:params.num_electrons])

    # Compute evolved density from the evolved wavefunctions
    density_ks = calculate_density_ks(params, wavefunctions_ks[:,:])

    # Error in the KS density away from the reference density
    error = norm(params,density_ks[:] - density_reference[:],'C2')

    # Return a particular output that defines the objective to be minimised
    if (objective_type == 'root'):
        return density_ks - density_reference
    elif (objective_type == 'opt'):
        return error
    else:
        raise RuntimeError('Not a valid type for the objective function output')


def expm_step(params,v_ks,wavefunctions_ks):
    r"""
    Time step defined by numpy's expm function
    """

    hamiltonian = construct_H(params,v_ks)

    #wavefunctions_ks = sp.sparse.linalg.expm_multiply(1.0j*params.dt*hamiltonian, wavefunctions_ks)
    updated_wvfns = np.zeros((params.Nspace,params.num_electrons), dtype=np.complex)
    for i in range(0,params.num_electrons):
        updated_wvfns[:,i] += sp.sparse.linalg.expm_multiply(1.0j*params.dt*hamiltonian, wavefunctions_ks[:,i])

    return updated_wvfns


def expm_evolve(params,wavefunctions_ks,v_ks,density_ks):
    r"""
    Computes time-dependent wavefunctions and density from a given initial wavefunction and v_ks
    """

    for i in range(1,params.Ntime):

        wavefunctions_ks[i,:,:] = expm_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        density_ks[i,:] = np.sum(abs(wavefunctions_ks[i,:,:])**2, axis=1)

    return density_ks


def crank_nicolson_step(params,v_ks,wavefunctions_ks):
    r"""
    Solves CN system of linear equations for the evolved wavefunctions:
    (I + 0.5i dt H) psi(t+dt) = (I - 0.5i dt H) psi(t)
    """

    # Create hamiltonian
    hamiltonian = construct_H(params,v_ks)

    # Create A and b in CN's Ax = b formula
    b = np.dot( ( np.eye(params.Nspace) - 0.5j * params.dt * hamiltonian), wavefunctions_ks )
    A = np.eye(params.Nspace) + 0.5j * params.dt * hamiltonian

    # Solve Ax = b
    wavefunctions_ks = np.linalg.solve(A,b)

    return wavefunctions_ks


def crank_nicolson_evolve(params,wavefunctions_ks,v_ks,density_ks):
    r"""
    Computes time-dependent wavefunctions and density from a given initial wavefunction and v_ks
    """

    for i in range(1,params.Ntime):

        wavefunctions_ks[i,:,:] = crank_nicolson_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        density_ks[i,:] = calculate_density_ks(params, wavefunctions_ks[i,:,:])
    return density_ks


def construct_H(params,v_ks):
    r"""
    Constructs the discretised Hamiltonian with an N-point stencil and the given KS potential
    """

    # Kinetic energy
    hamiltonian = -0.5*discrete_Laplace(params)

    # Potential energy
    hamiltonian += np.diag(v_ks)

    return hamiltonian




