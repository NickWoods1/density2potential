import numpy as np
from scipy.optimize import root,minimize
import matplotlib.pyplot as plt
from density2potential.plot.animate import animate_function, animate_two_functions
from scipy.linalg import expm

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


    i, error = 0, 1
    while(error > 1e-14):

        # Construct Hamiltonian for a given Kohn-Sham potential
        hamiltonian = construct_H(params,v_ks[0,:])

        # Find eigenvectors and eigenvalues of this Hamiltonian
        eigenenergies_ks, eigenfunctions_ks = np.linalg.eigh(hamiltonian)

        # Store the lowest N_electron eigenfunctions as wavefunctions
        wavefunctions_ks[0,:,0:params.num_electrons] = eigenfunctions_ks[:,0:params.num_electrons]

        # Normalise the wavefunctions w.r.t the continous L2 norm
        wavefunctions_ks[0,:,0:params.num_electrons] = normalise_function(params,wavefunctions_ks[0,:,0:params.num_electrons])

        # Construct KS density
        density_ks[0,:] = np.sum(np.abs(wavefunctions_ks[0,:,:])**2,axis=1,dtype=np.float)

        # Error in the KS density away from the reference density
        error = norm(params,density_ks[0,:] - density_reference[0,:],'C2')

        # Update the KS potential with a steepest descent scheme
        v_ks[0,:] -= (density_reference[0,:]**0.05 - density_ks[0,:]**0.05)

        print('Error = {0} at iteration {1}'.format(error,i), end='\r')

        i += 1


    #opt_info = root(groundstate_objective_function, v_ks[0, :], args=(params, wavefunctions_ks[:,:,:], density_reference[:,:],
    #                                                                    ), method='hybr', options={'maxiter': 10000})

    #v_ks[0,:] = opt_info.x

    #hamiltonian = construct_H(params, v_ks[0, :])
    #eigenenergies_ks, eigenfunctions_ks = np.linalg.eigh(hamiltonian)
    #wavefunctions_ks[0, :, 0:params.num_electrons] = eigenfunctions_ks[:, 0:params.num_electrons]
    #wavefunctions_ks[0, :, 0:params.num_electrons] = normalise_function(params,wavefunctions_ks[0, :, 0:params.num_electrons])
    #density_ks[0, :] = np.sum(np.abs(wavefunctions_ks[0, :, :]) ** 2, axis=1, dtype=np.float)
    #error = norm(params, density_ks[0, :] - density_reference[0, :], 'C2')

    print('Final error in the ground state KS density is {0} after {1} iterations'.format(error,i))

    # Time-Dependence
    # First generate the perturbing potential from exact system as initial guess
    L = 0.5*params.space
    x = np.linspace(-L,L,params.Nspace)
    v_pert = 0.1*x
    v_ext = -0.4*np.exp(-0.2*x**2)

    # Set initial guess for time-dependent v_ks
    v_ks[1:,:] = v_ks[0,:] + v_pert[:]

    #v_ks = np.load('v_ks_evolved.npy')
    #density_ks_idea = np.load('density_ks_evolved.npy')
    #wvfns = np.load('wavefunctions_ks.npy')
    #wavefunctions_ks[0,:,0] = wvfns[:,0]
    #wavefunctions_ks[0,:,1] = wvfns[:,1]

    #plt.plot(v_ks[0,:])
    #plt.plot(v_ks[1,:])
    #plt.show()

    # Test CN evolution for a given v_ks
    #density_ks[:,:] = expm_evolve(params,wavefunctions_ks,v_ks,density_ks)
    #for i in range(1,params.Ntime):
    #    print('error at iteration {0} is {1}'.format(i,norm(params,density_ks[i,:] - density_ks_idea[i,:],'C2')))
    #animate_function(params,v_ks,20,'vks_idea','vks_idea')
    #animate_function(params,density_ks_idea,10,'density_iDEA','ks')
    #animate_two_functions(params,density_ks,density_ks_idea,30,'compare','My density','iDEA density')

    # Optimise the time-dependent KS potential
    for i in range(1,500):

        # Find the v_ks that minimises the specified objective function
        opt_info = root(evolution_objective_function,v_ks[i,:],args=(params,wavefunctions_ks[i-1,:,:],density_reference[i,:],
                                                                     'root', 'expm'), method='hybr',options={'maxiter':10000})
        # Final Kohn-Sham potential
        v_ks[i,:] = opt_info.x

        # Compute the final error away from the reference density for this Kohn-Sham potential
        #wavefunctions_ks[i,:,:] = crank_nicolson_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        wavefunctions_ks[i,:,:] = expm_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        density_ks[i,:] = np.sum(abs(wavefunctions_ks[i,:,:])**2, axis=1)
        error = norm(params, density_ks[i,:] - density_reference[i,:], 'C1')

       # plt.plot(v_ks[i,:])
       # plt.show()

        print('Time step {0}'.format(i))
        print('Optimiser status = {0} with final error {1} after {2} iterations'.format(opt_info.success,error,opt_info.nfev))
        print('Integrated Kohn-Sham potential = {}'.format(norm(params,v_ks[i,:],'C2')))
        #print('error {}'.format(error))
        print(' ')


    animate_function(params,v_ks,40,'v_ks','v_ks')

    return density_ks, v_ks, wavefunctions_ks


def groundstate_objective_function(v_ks,params,wavefunctions_ks,density_reference):

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

    return abs(density_reference[0,:]**0.05 - density_ks[:]**0.05)


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
        raise Exception('Invalid time evolution method specified')

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
        raise Exception('Not a valid type for the objective function output')


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
    Solves CN system of linear equations for the evolved wavefunctions

    (I + 0.5i dt H) psi(t+dt) = (I - 0.5i dt H) psi(t)

    :param params:
    :param v_ks:
    :param wavefunctions_ks:
    :return:
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
    Computes a time-dependent wavefunctions and density from a given initial wavefunction and v_ks
    """

    for i in range(1,params.Ntime):

        wavefunctions_ks[i,:,:] = crank_nicolson_step(params,v_ks[i,:],wavefunctions_ks[i-1,:,:])
        density_ks[i,:] = np.sum(abs(wavefunctions_ks[i,:,:])**2, axis=1)

    return density_ks



def construct_H(params,v_ks):
    r"""
    Constructs the discretised Hamiltonian with an N-point stencil and the given KS potential

    :param params: input parameters object
    :param v_ks: Kohn-Sham potential at a given timestep, 1D array [space]
    :return: Hamiltonian, 2D array [space,space]
    """

    # 3-point stencil for the Laplace operator
    hamiltonian = -2.0 * np.eye(params.Nspace, dtype=np.float)
    hamiltonian += np.diag(np.ones(params.Nspace-1),1) + np.diag(np.ones(params.Nspace-1),-1)
    hamiltonian *= -0.5 / params.dx**2

    # Add KS potential
    hamiltonian += np.diag(v_ks)

    return hamiltonian


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
    """

    if (norm_type == 'C2'):
        norm = (np.sum( np.conj(function[:]) * function[:] )*params.dx)**0.5
    elif (norm_type == 'D2'):
        norm = np.linalg.norm(function)
    elif (norm_type == 'C1'):
        norm = np.sum(abs(function))*params.dx
    elif (norm_type == 'D1'):
        norm = np.linalg.norm(function,1)
    else:
        raise Exception('Invalid norm type "{0}" used in norm()'.format(norm_type))

    return norm

