import numpy as np
from density2potential.utils.math import discrete_Laplace, norm, normalise_function
import matplotlib.pyplot as plt
import scipy as sp

"""
Solve the exact time-dependent Schrodinger equation in 1D to generate an exact density
"""

def solve_ground_state(params):
    r"""
    Solves the ground state Schrodinger equation
    """

    """
    ### ONE PARTICLE
    params.num_electrons = 1
    hamiltonian = construct_H(params)
    params.dt = 0.001
    eigenfunction_gs = np.zeros(params.Nspace,dtype=np.complex)
    eigenfunction_gs += np.random.normal(0.0,0.1,params.Nspace)
    eigenfunction_gs += 1.0j*np.random.normal(0.0,0.1,params.Nspace)
    for i in range(0,10000):
        eigenfunction_gs = crank_nicolson_step(params,hamiltonian,eigenfunction_gs)
    

    #eigenenergy_gs, eigenfunction_gs = np.linalg.eigh(hamiltonian)#sp.sparse.linalg.eigsh(hamiltonian,1)
    #print('Ground state energy: {}'.format(eigenenergy_gs[0]))
    wavefunction = normalise_function(params,eigenfunction_gs)#[:,2])
    density = abs(wavefunction[:])**2
    plt.plot(params.v_ext)
    plt.plot(density)
    plt.show()
    params.num_electrons = 2
    """

    # Initilise the many-body wavefunction, and random initial guess.
    wavefunction = np.zeros((params.Nspace**2), dtype=np.complex)
    wavefunction[:] = np.random.normal(0,1.0,params.Nspace**2)
    wavefunction[:] += 1.0j * np.random.normal(0,1.0,params.Nspace**2)

    # Construct full MB hamiltonian, and sparsify
    hamiltonian = construct_H(params)
    hamiltonian = operator_to_sparse_matrix(params,hamiltonian)
    antisymm_subspace_xform = antisymmetric_subspace_matrix(params)
    antisymm_subspace_xform_inverse = sp.sparse.linalg.inv(antisymm_subspace_xform)
    #hamiltonian_antisymm = hamiltonian.dot()


    # Find g.s. eigenvector and eigenvalue using Lanszcos algorithm
    eigenenergy_gs, eigenfunction_gs = sp.sparse.linalg.eigsh(hamiltonian_sparse, 2, which='SM')

    # Normalise g.s. wavefunction
    norm = 0
    for i in range(params.Nspace**2):
        norm += (np.conj(eigenfunction_gs[i,0]) * eigenfunction_gs[i,0]) * params.dx**2
    eigenfunction_gs[i,0] *= norm ** -0.5

    print('Ground state energy: {}'.format(eigenenergy_gs))

    # Revert to tensor form
    wavefunction = pack_wavefunction(params,eigenfunction_gs[:,1])

    # Compute density
    density = np.zeros(params.Nspace)
    for i in range(0,params.Nspace):
        density[i] = 2.0 * (np.sum( abs(wavefunction[:,i])**2 )) * params.dx

    plt.plot(density)
    plt.show()


def construct_H(params):
   r"""
   Constructs the (dense) Hamiltonian in tensor form for an N particle system
   """

    if params.num_electrons == 1:
        # One-particle Hamiltonian
        hamiltonian = np.zeros((params.Nspace,params.Nspace))

        # Kinetic energy
        hamiltonian[:,:] += -0.5*discrete_Laplace(params)

        # External potential
        hamiltonian[:,:] += np.diag(params.v_ext)

    elif params.num_electrons == 2:
        # Two-particle Hamiltonian
        hamiltonian = np.zeros((params.Nspace,params.Nspace,params.Nspace,params.Nspace))

        # External potential
        for i in range(0,params.Nspace):
            hamiltonian[:,:,i,i] += np.diag(params.v_ext)
            hamiltonian[i,i,:,:] += np.diag(params.v_ext)

        # Kinetic energy
        for i in range(0,params.Nspace):
            hamiltonian[:,:,i,i] += -0.5*discrete_Laplace(params)
            hamiltonian[i,i,:,:] += -0.5*discrete_Laplace(params)

        # Softened Coulomb Potential
        for i in range(0,params.Nspace):
            for j in range(0,params.Nspace):
                hamiltonian[i,i,j,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

    else:
        raise Exception('Cannot construct the Hamiltonian for the given particle number: {}'.format(params.num_electrons))

    return hamiltonian


def operator_to_sparse_matrix(params,hamiltonian):
    r"""
    Takes a linear operator (in tensor form) and converts it to a csr (scipy) sparse matrix
    """

    # Matrix index
    mu, nu = 0, -1
    hamiltonian_matrix = np.zeros((params.Nspace**2,params.Nspace**2), dtype=np.float)

    # Map the tensor to a matrix of appropriate dimension with some mapping NxNxNxN ---> N^2 x N^2
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):
            nu += 1
            mu = 0
            for k in range(0,params.Nspace):
                for l in range(0,params.Nspace):
                    hamiltonian_matrix[mu,nu] = hamiltonian[l,j,k,i]
                    mu += 1

    # Convert this output matrix (which is dense) to a sparse matrix
    hamiltonian_sparse = sp.sparse.csr_matrix(hamiltonian_matrix)

    return hamiltonian_sparse


def antisymmetric_subspace_matrix(params):
    """
    Defines the matrix that reduces the Hamiltonian s.t. it operates on an antisymmetric domain (i.e. its eigenfunctions
    are manifestly antisymm.
    """

    # Compute # of non-zero elements in the subspace transformation and init the arrays that hold these elements
    num_nonzero_elements = int(params.Nspace/2)*(params.Nspace-1)
    col, row, entries = np.zeros(num_nonzero_elements), np.zeros(num_nonzero_elements), np.ones(num_nonzero_elements)

    # Specify the row and column number of the non-zero elements
    m = -params.Nspace
    n = -1
    counter = 0
    for i in range(0,int(params.Nspace/2)):
        n += 1
        m += params.Nspace
        for j in range(0,params.Nspace):
            if j != n:
                col[counter] = j + m
                row[counter] = counter
                counter += 1

    # Output as a sparse matrix
    return sp.sparse.csr_matrix(entries, (row, col))


def apply_H(params,hamiltonian,wavefunction):
    """
    Apply a 'tridiagonal' Hamiltonian in tensor form to a vector (wavefunction)
    """

    wavefunction_new = np.zeros((params.Nspace,params.Nspace), dtype=np.complex)

    # Apply the tensor diagonal
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):
            wavefunction_new[i,j] = hamiltonian[i,i,j,j]*wavefunction[i,j]

    # Apply the tensor offdiagonal
    for i in range(0,params.Nspace-1):
        for j in range(0,params.Nspace):
            wavefunction_new[i,j] += hamiltonian[i,i+1,j,j]*wavefunction[i+1,j] + hamiltonian[i,i+1,j,j]*wavefunction[i,j]
            wavefunction_new[i,j] += hamiltonian[j,j,i,i+1]*wavefunction[j,i+1] + hamiltonian[j,j,i,i+1]*wavefunction[j,i]

    return wavefunction_new


def unpack_wavefunction(params,wavefunction):
    """
    Turns a NxN 2D array wavefunction into a N**2 1D array wavefunction given some mapping NxNxNxN ---> N^2 x N^2
    """

    wavefunction_unpacked = np.zeros((params.Nspace**2), dtype=np.complex)
    i, j, k = 0, params.Nspace, 0
    while j<=params.Nspace**2:
        wavefunction_unpacked[i:j] = wavefunction[k,:]
        k += 1
        i += params.Nspace
        j += params.Nspace

    return wavefunction_unpacked

def pack_wavefunction(params,wavefunction):
    """
    Turns an N**2 1D array wavefunction into a NxN 2D array wavefunction given some mapping NxNxNxN ----> N^2 x N^2
    """

    wavefunction_packed = np.zeros((params.Nspace,params.Nspace), dtype=np.complex)
    i, j, k = 0, params.Nspace, 0
    while j<= params.Nspace**2:
        wavefunction_packed[:,k] = wavefunction[i:j]
        k += 1
        i += params.Nspace
        j += params.Nspace

    return wavefunction_packed


def crank_nicolson_step(params,hamiltonian,wavefunction):
    """
    Perform a CN step -- imaginary or real time
    """

    if params.num_electrons == 2:
        identity_sparse = sp.sparse.csr_matrix(np.eye(params.Nspace**2))
        A = identity_sparse - 0.5*params.dt*hamiltonian
        B = identity_sparse + 0.5*params.dt*hamiltonian

        b = B.dot(wavefunction)

        # Solve Ax = b
        wavefunction_propagated = sp.sparse.linalg.spsolve(A,b)

        norm = 0
        for i in range(0,int(params.Nspace**2 / 2)):
            norm += (np.conj(wavefunction_propagated[i]) * wavefunction_propagated[i]) * params.dx
        norm = norm ** -0.5
        wavefunction_propagated *= norm


        norm = 0
        for i in range(int((params.Nspace**2 / 2) + 1),params.Nspace**2):
            norm += (np.conj(wavefunction_propagated[i]) * wavefunction_propagated[i]) * params.dx
        norm = norm ** -0.5
        wavefunction_propagated *= norm

    if params.num_electrons == 1:

        identity = np.eye(params.Nspace)
        A = identity - 0.5*params.dt*hamiltonian
        B = identity + 0.5*params.dt*hamiltonian
        b = np.dot(B,wavefunction)

        wavefunction_propagated = np.linalg.solve(A,b)

        wavefunction_propagated = normalise_function(params,wavefunction_propagated)


    print(np.dot(wavefunction_propagated,(np.dot(hamiltonian,wavefunction_propagated))))
    print('Change in the wavefunction: {}'.format(np.sum(abs(wavefunction[:] - wavefunction_propagated[:]))))

    # Ensure the wavefunctions are normalised (although with unitary time evolution they should be)
    #wavefunctions_ks[:,0:params.num_electrons] = normalise_function(params,wavefunctions_ks[:,0:params.num_electrons])

    return wavefunction_propagated