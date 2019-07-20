import numpy as np
import matplotlib.pyplot as plt

"""
Functions that contain the core functionality to find a Kohn-Sham potential given a density
"""

def generate_ks_potential(params,density_reference):
    r"""
    Reverse engineers a reference density to product a Kohn-Sham potential

    :param params: input parameters object
    :param density_reference: the reference density, 2D array [time,space]
    :return: The output Kohn-Sham potential, the density it produces, and the corresponding wavefunctions.
    """

    # Ground state
    # Init variables
    v_ks = np.zeros((params.Ntime,params.Nspace))
    density_ks = np.zeros((params.Ntime,params.Nspace))
    wavefunctions_ks = np.zeros((params.Ntime,params.Nspace,params.num_electrons), dtype=complex)

    i, error = 0, 1
    while(error > 1e-15):

        # Construct Hamiltonian for a given Kohn-Sham potential
        hamiltonian = construct_H(params,v_ks[0,:])

        # Find eigenvectors and eigenvalues of this Hamiltonian
        eigenenergies_ks, eigenfunctions_ks = np.linalg.eigh(hamiltonian)

        # Store the lowest N_electron eigenfunctions as wavefunctions
        wavefunctions_ks[0,:,0:params.num_electrons] = eigenfunctions_ks[:,0:params.num_electrons]

        # Normalise the wavefunctions w.r.t the continous L2 norm
        wavefunctions_ks[0,:,0:params.num_electrons] = normalise_function(params,wavefunctions_ks[0,:,0:params.num_electrons])

        # Construct KS density
        density_ks[0,:] = np.sum(wavefunctions_ks[0,:,:]**2,axis=1,dtype=np.float)

        # Error in the KS density away from the reference density
        error = norm(params,density_ks[0,:] - density_reference[0,:],'C2')

        # Update the KS potential with a steepest descent scheme
        v_ks[0,:] -= (density_reference[0,:]**0.05 - density_ks[0,:]**0.05)

        print('Error = {0} at iteration {1}'.format(error,i), end='\r')

        i += 1

    print('Final error in the ground state KS density is {0} after {1} iterations'.format(error,i))

    return density_ks, v_ks, wavefunctions_ks


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

    norm_type : string -- type of norm to be computed
    continous L2 norm C2
    discrete L2 norm D2
    continous L1 norm C1
    discrete L1 norm D1
    """

    if (norm_type == 'C2'):
        norm = (np.sum(function**2)*params.dx)**0.5
    elif (norm_type == 'D2'):
        norm = np.linalg.norm(function)
    elif (norm_type == 'C1'):
        norm = np.sum(abs(function))
    elif (norm_type == 'D1'):
        norm = np.linalg.norm(function,1)

    return norm

