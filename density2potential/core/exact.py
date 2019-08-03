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

    # Construct full MB hamiltonian, and sparsify
    hamiltonian = construct_H(params,'antisymmetric')
    hamiltonian = operator_to_sparse_matrix(params,hamiltonian)

    # Find eigenfunctions and eigenvalues
    eigenenergy_gs,eigenfunction_gs = sp.linalg.eigh(hamiltonian)
    # Find g.s. eigenvector and eigenvalue using Lanszcos algorithm
    #eigenenergy_gs, eigenfunction_gs = sp.sparse.linalg.eigsh(hamiltonian, 1, which='SM')

   # Normalise g.s. wavefunction
    eigenfunction_gs[:,0] *= (np.sum(eigenfunction_gs[:,0]**2) * params.dx**2)**-0.5

    # Plot g.s. wvfn
    fig, ax = plt.subplots(nrows=2,ncols=1)
    ax[0].plot(eigenfunction_gs[:,0])

    # Ground state energy
    print('Ground state energy: {}'.format(eigenenergy_gs[0]))

    # Revert to tensor form
    wavefunction = pack_wavefunction(params,eigenfunction_gs[:,0])

    # Compute density
    density = np.zeros(params.Nspace)
    for i in range(0,params.Nspace):
        density[i] = 2.0 * (np.sum( abs(wavefunction[:,i])**2 )) * params.dx


    # Same but symmetric
    hamiltonian = construct_H(params,'symmetric')
    hamiltonian = operator_to_sparse_matrix(params,hamiltonian)

    # Find g.s. eigenvector and eigenvalue using Lanszcos algorithm
    eigenenergy_gs, eigenfunction_gs = sp.sparse.linalg.eigsh(hamiltonian, 2, which='SM')

    # Normalise g.s. wavefunction
    eigenfunction_gs[:,1] *= (np.sum(eigenfunction_gs[:,1]**2) * params.dx**2)**-0.5
    print('Ground state energy: {}'.format(eigenenergy_gs[1]))
    wavefunction = pack_wavefunction(params,eigenfunction_gs[:,1])
    density2 = np.zeros(params.Nspace)
    for i in range(0,params.Nspace):
        density2[i] = 2.0 * (np.sum( abs(wavefunction[:,i])**2 )) * params.dx

    ax[0].plot(eigenfunction_gs[:,1])

    ax[1].plot(density2)
    ax[1].plot(density)
    plt.show()

def construct_H(params,basis_type):
    r"""
    Constructs the (dense) Hamiltonian in a delta function basis (tensor form) for an N particle system.
    Either a symmetric (basis_type = symmetric) or (basis_type =)antisymmetric delta function basis is used.
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
        for j in range(0,params.Nspace):
            for i in range(0,params.Nspace):
                hamiltonian[i,i,j,j] += params.v_ext[i] + params.v_ext[j]

        # Kinetic energy
        for i in range(0,params.Nspace):
            hamiltonian[:,:,i,i] += -0.5*discrete_Laplace(params)
            hamiltonian[i,i,:,:] += -0.5*discrete_Laplace(params)

        # Softened Coulomb Potential
        for i in range(0,params.Nspace):
            for j in range(0,params.Nspace):
                hamiltonian[i,i,j,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

        # Add components corresponding to the antisymmetric parts of the basis
        if basis_type == 'antisymmetric':

            # Kinetic energy
            for i in range(0,params.Nspace):
                # Antisymmetric
                hamiltonian[:,i,i,:] -= -0.5*discrete_Laplace(params)
                hamiltonian[i,:,:,i] -= -0.5*discrete_Laplace(params)

            # External potential
            for j in range(0,params.Nspace):
                for i in range(0,params.Nspace):
                    hamiltonian[i,j,j,i] -= params.v_ext[i] + params.v_ext[j]

            # Softened Coulomb Potential
            for i in range(0,params.Nspace):
                for j in range(0,params.Nspace):
                    hamiltonian[i,j,j,i] -= 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

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

    return hamiltonian_matrix


def antisymmetric_subspace_matrix(params):
    """
    Defines the matrix that reduces the Hamiltonian s.t. it operates on an antisymmetric domain (i.e. its eigenfunctions
    are manifestly antisymm.
    """

    # Compute # of non-zero elements in the subspace transformation and init the arrays that hold these elements
    num_nonzero_elements = int(params.Nspace/2)*(params.Nspace-1)
    col, row, entries = np.zeros(num_nonzero_elements, dtype=np.int), \
                        np.zeros(num_nonzero_elements, dtype=np.int), \
                        np.ones(num_nonzero_elements, dtype=np.int)


    col = np.zeros(int(params.Nspace/2)*(params.Nspace-1), dtype=np.int)
    row = np.zeros(int(params.Nspace/2)*(params.Nspace-1), dtype=np.int)
    entries = np.ones(int(params.Nspace/2)*(params.Nspace-1), dtype=np.int)



    print(col)

    # Specify the row and column number of the non-zero elements
    #m = -params.Nspace
    #n = -1
    #counter = 0
    #for i in range(0,int(params.Nspace/2)):
    #    n += 1
    #    m += params.Nspace
    #    for j in range(0,params.Nspace):
    #        if j != n:
    #            col[counter] = j + m
    #            row[counter] = counter
    #            counter += 1

    # Output as a sparse matrix
    return sp.sparse.csr_matrix(entries, (row, col))


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

