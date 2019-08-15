import numpy as np
from scipy.optimize import root,minimize
import matplotlib.pyplot as plt
from density2potential.plot.animate import animate_function, animate_two_functions
from density2potential.utils.math import norm, normalise_function, discrete_Laplace, calculate_density_exact
from density2potential.core.exact_TISE import construct_H_dense
from scipy.linalg import expm
import scipy as sp

"""
Functions that contain the core functionality to find a Kohn-Sham potential given a density
"""

def generate_ks_potential(params,density_reference):
    r"""
    Reverse engineers a reference density to product a Kohn-Sham potential

    :param params: input parameters object
    :param density_reference: the reference density, 2D array [time,space]
    :return: The output Kohn-Sham potential, the density it produces, and the corresponding wavefunctions, 2D array [time,space]
    """

    # Ground state
    # Init variables
    v_ks = np.zeros((params.Ntime,params.Nspace))
    density_ks = np.zeros((params.Ntime,params.Nspace))
    wavefunctions_ks = np.zeros((params.Ntime,params.Nspace,params.num_electrons), dtype=complex)

    # Initial guess for the Kohn-Sham potential
    v_ks[0,:] = 0#0.5*(0.25**2)*params.space_grid**2 - params.v_ext_shift#params.v_ext - params.v_ext_shift
    v_ks[1:,:] = params.v_ext + params.v_pert
    #v_ks += np.random.normal(0.0,0.01,params.Nspace)

    # Compute the ground-state Kohn-Sham potential
    i, error = 0, 1
    while(error > 1e-19):

        # Construct Hamiltonian for a given Kohn-Sham potential
        hamiltonian = construct_H(params,v_ks[0,:])

        # Find eigenvectors and eigenvalues of this Hamiltonian
        eigenenergies_ks, eigenfunctions_ks = sp.linalg.eigh(hamiltonian)

        # Store the lowest N_electron eigenfunctions as wavefunctions
        wavefunctions_ks[0,:,0:params.num_electrons] = eigenfunctions_ks[:,0:params.num_electrons]

        # Normalise the wavefunctions w.r.t the continous L2 norm
        wavefunctions_ks[0,:,0] = eigenfunctions_ks[:,0]*(np.sum(eigenfunctions_ks[:,0]**2) * params.dx)**-0.5

        # Construct KS density
        density_ks[0,:] = calculate_density_exact(params, wavefunctions_ks[0,:,0])

        # Error in the KS density away from the reference density
        error = norm(params,density_ks[0,:] - density_reference[0,:],'MAE')

        #if i % 1000 == 0:
        #    plt.plot(0.5*(0.25**2)*params.space_grid**2)
        #    plt.plot(v_ks[0,:])
         #   plt.plot(density_reference[0,:])
        #    plt.plot(density_ks[0,:])
        #    plt.show()

        # Update the KS potential with a steepest descent scheme
        #if error > 1e-10:
        #v_ks[0,:] -= 1*(density_reference[0,:]**0.05 - density_ks[0,:]**0.05)#/ density_reference[0,:]
        #else:
        v_ks[0,:] -= 0.1*(density_reference[0,:] - density_ks[0,:]) / density_reference[0,:]
        error_vks = norm(params, v_ks[0, :] - params.v_ext, 'MAE')

        print('Error = {0} at iteration {1} errro2 {2}'.format(error,i,error_vks), end='\r')

        i += 1

    print('Final error in the ground state KS density is {0} after {1} iterations'.format(error,i))

    """
    # Compute the ground-state potential using a scipy optimiser
    opt_info = root(groundstate_objective_function, v_ks[0, :], args=(params, wavefunctions_ks[:,:,:], density_reference[:,:],
                                                                      ), method='hybr', options={'ftol':1e-10,'disp':True})

    #opt_info = minimize(groundstate_objective_function, v_ks[0, :], args=(params, wavefunctions_ks[:,:,:], density_reference[:,:],
    #                                                                  ), method='SLSQP', options={'disp':True})

    # Output v_ks
    v_ks[0,:] = opt_info.x

    # Compute the corresponding wavefunctions, density, and error
    hamiltonian = construct_H(params, v_ks[0, :])
    eigenenergies_ks, eigenfunctions_ks = np.linalg.eigh(hamiltonian)
    wavefunctions_ks[0, :, 0:params.num_electrons] = eigenfunctions_ks[:, 0:params.num_electrons]
    wavefunctions_ks[0, :, 0:params.num_electrons] = normalise_function(params,wavefunctions_ks[0, :, 0:params.num_electrons])
    density_ks[0, :] = np.sum(np.abs(wavefunctions_ks[0, :, :]) ** 2, axis=1, dtype=np.float)
    error = norm(params, density_ks[0, :] - density_reference[0, :], 'C2')
    print('Final error = {0} after {1} function evaluations. Status: {2}'.format(error,opt_info.success,opt_info.success))
   """

    # Set initial guess for time-dependent v_ks
    v_ks[1:,:] = v_ks[0,:] + params.v_pert

    # Optimise the time-dependent KS potential
    for i in range(1,params.Ntime):

        # Find the v_ks that minimises the specified objective function
        opt_info = root(evolution_objective_function,v_ks[i,:],args=(params,wavefunctions_ks[i-1,:,:],density_reference[i,:],
                                                                     'root', 'expm'), method='hybr',options={'maxiter':10000})
        # Final Kohn-Sham potential
        v_ks[i,:] = opt_info.x

        # Compute the final error away from the reference density for this Kohn-Sham potential
        #wavefunctions_ks[i,:,:] = crank_nicolson_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        wavefunctions_ks[i,:,:] = expm_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        density_ks[i,:] = np.sum(abs(wavefunctions_ks[i,:,:])**2, axis=1)
        error = norm(params, density_ks[i,:] - density_reference[i,:], 'MAE')

        print('Time step {0}'.format(i))
        #print('MAE in Kohn-Sham potential away from exact {}'.format(norm(params,v_ks[i,:] - v_ext - v_pert,'MAE')))
        print('Optimiser status: {0} with final error = {1} after {2} iterations'.format(opt_info.success,error,opt_info.nfev))
        print('Integrated Kohn-Sham potential = {}'.format(norm(params,v_ks[i,:],'C2')))
        print(' ')

    # Plot animated Kohn-Sham potential
    animate_function(params,v_ks,5,'TD_KS_potential','v_ks')

    return density_ks, v_ks, wavefunctions_ks


def groundstate_objective_function(v_ks,params,wavefunctions_ks,density_reference):
    r"""
    Ground state objective function, the root of which is the Kohn-Sham potential that generates a ground
    state reference density
    """

    # Construct Hamiltonian for a given Kohn-Sham potential
    hamiltonian = construct_H(params, v_ks[:])

    # Find eigenvectors and eigenvalues of this Hamiltonian
    eigenenergies_ks, eigenfunctions_ks = np.linalg.eigh(hamiltonian)

    # Store the lowest N_electron eigenfunctions as wavefunctions
    wavefunctions_ks[0, :, 0:params.num_electrons] = eigenfunctions_ks[:, 0:params.num_electrons]

    # Normalise the wavefunctions w.r.t the continous L2 norm
    wavefunctions_ks[0, :, 0:params.num_electrons] = normalise_function(params,wavefunctions_ks[0, :, 0:params.num_electrons])

    # Construct KS density
    density_ks = np.sum(np.abs(wavefunctions_ks[0, :, :])**2, axis=1, dtype=np.float)

    return abs(density_reference[0,:] - density_ks[:]) / density_reference[0,:]


def evolution_objective_function(v_ks,params,wavefunctions_ks,density_reference,objective_type,evolution_type):
    r"""
    The objective function for root finding and optimisation algorithms, defined with some
    method of evolving psi(t) --> psi(t+dt).

    :param v_ks: Kohn-Sham potential for which the evolution of psi is computed, 1D array, [space]
    :param params: input params for calculation
    :param wavefunctions_ks: current Kohn-Sham wavefunctions psi(t). 2D array labelled [space,band] for a given t
    :param density_reference: the reference densiy at the t+dt timestep, 1D array [space]
    :param type: Whether to return a root finding objective function (R^n --> R^n),
                 or an optimisation objective function (R^n --> R). String, choices: "opt" or "root".
    :return: Output of the objective, the 'error', in some sense.
    """

    # Evolve the wavefunctions according to the given scheme
    if (evolution_type == 'CN'):
        wavefunctions_ks = crank_nicolson_step(params,v_ks,wavefunctions_ks)
    elif (evolution_type == 'expm'):
        wavefunctions_ks = expm_step(params,v_ks,wavefunctions_ks)
    else:
        raise RuntimeError('Invalid time evolution method specified')

    # Compute evolved density from the evolved wavefunctions
    density_ks = np.sum(abs(wavefunctions_ks[:,:])**2, axis=1)

    # Error in the KS density away from the reference density
    error = norm(params,density_ks[:] - density_reference[:],'C2')

    v_ks_average = np.sum(v_ks[:])

    # Return a particular output that defines the objective to be minimised
    if (objective_type == 'root'):
        return abs(density_reference - density_ks)
    elif (objective_type == 'opt'):
        return error
    else:
        raise RuntimeError('Not a valid type for the objective function output')


def expm_step(params,v_ks,wavefunctions_ks):
    r"""
    Time step defined by numpy's expm function
    """

    hamiltonian = construct_H(params,v_ks)
    wavefunctions_ks = np.dot(expm(1.0j*params.dt*hamiltonian), wavefunctions_ks)

    return wavefunctions_ks


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

    # Ensure the wavefunctions are normalised (although with unitary time evolution they should be)
    #wavefunctions_ks[:,0:params.num_electrons] = normalise_function(params,wavefunctions_ks[:,0:params.num_electrons])

    return wavefunctions_ks


def crank_nicolson_evolve(params,wavefunctions_ks,v_ks,density_ks):
    r"""
    Computes time-dependent wavefunctions and density from a given initial wavefunction and v_ks
    """

    for i in range(1,params.Ntime):

        wavefunctions_ks[i,:,:] = crank_nicolson_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        density_ks[i,:] = np.sum(abs(wavefunctions_ks[i,:,:])**2, axis=1)

    return density_ks


def construct_H(params,v_ks):
    r"""
    Constructs the discretised Hamiltonian with an N-point stencil and the given KS potential
    """

    # Kinetic energy
    hamiltonian = -0.5 * discrete_Laplace(params)

    # Potential energy
    hamiltonian += np.diag(v_ks)

    return hamiltonian




