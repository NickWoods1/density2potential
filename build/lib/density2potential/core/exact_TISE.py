import numpy as np
from density2potential.utils.math import discrete_Laplace, norm, normalise_function
from density2potential.plot.animate import animate_function
import matplotlib.pyplot as plt
import scipy as sp

"""
Solve the exact time-independent Schrodinger equation in 1D to generate exact ground state
density, wavefunction, and energy
"""

def solve_TISE(params):
    r"""
    Solves the time-independent Schrodinger equation
    """

    antisymm_expansion, antisymm_reduction = expansion_and_reduction_matrix(params)


    # Generate the initial guess wavefunction for the iterative diagonaliser
    # Slater determinant of single-particle wavefunctions
    if params.num_electrons > 1:
        wavefunction = initial_guess_wavefunction(params)

    wavefunction = antisymm_reduction.dot(wavefunction)


    # Construct the sparse many-body Hamiltonian in an antisymmetric basis
    hamiltonian = construct_H_sparse(params)

    H_reduced = hamiltonian.dot(antisymm_expansion)
    H_reduced = antisymm_reduction.dot(H_reduced)

    # Find g.s. eigenvector and eigenvalue using Lanszcos algorithm
    eigenenergy_gs, eigenfunction_gs = sp.sparse.linalg.eigsh(H_reduced, 1, which='SM', v0=wavefunction)

    # Normalise g.s. eigenfunction
    #eigenfunction_gs[:,0] *= (np.sum(eigenfunction_gs[:,0]**2) * params.dx**2)**-0.5

    # Ground state energy (n.b. undo the potential and regularisation shift)
    print('Ground state energy: {0}'.format(eigenenergy_gs[0] + 2.0*params.v_ext_shift))

    # Revert to tensor form
    new = antisymm_expansion.dot(eigenfunction_gs[:,0])
    wavefunction = pack_wavefunction(params,new).real

    # Compute density
    density = 2.0 * (np.sum(abs(wavefunction[:, :])**2,axis=0)) * params.dx

    plt.plot(density)
    plt.show()



    # Generate the initial guess wavefunction for the iterative diagonaliser
    # Slater determinant of single-particle wavefunctions
    if params.num_electrons > 1:
        wavefunction = initial_guess_wavefunction(params)

    # Construct the sparse many-body Hamiltonian in an antisymmetric basis
    hamiltonian = construct_H_sparse(params)

    # Add to diagonal to regularise
    hamiltonian += sp.sparse.csr_matrix(np.diag(100*np.ones(params.Nspace**2)))

    # Find g.s. eigenvector and eigenvalue using Lanszcos algorithm
    eigenenergy_gs, eigenfunction_gs = sp.sparse.linalg.eigsh(hamiltonian, 1, which='SM', v0=wavefunction)

    # Normalise g.s. eigenfunction
    eigenfunction_gs[:,0] *= (np.sum(eigenfunction_gs[:,0]**2) * params.dx**2)**-0.5

    # Ground state energy (n.b. undo the potential and regularisation shift)
    print('Ground state energy: {0}'.format(eigenenergy_gs[0] + 2.0*params.v_ext_shift - 100))

    # Revert to tensor form
    wavefunction = pack_wavefunction(params,eigenfunction_gs[:, 0]).real

    # Compute density
    density = 2.0 * (np.sum(abs(wavefunction[:, :])**2,axis=0)) * params.dx

    plt.plot(density)
    plt.show()

    np.save('groundstate_density', density)
    np.save('groundstate_wavefunction', wavefunction)

    return wavefunction, density, eigenenergy_gs[0]


def initial_guess_wavefunction(params):
    r"""
    Generate the initial guess for the many-body iterative diagonaliser: Slater determinant of single-particle states.
    """

    # Solve single particle equation
    params.num_electrons = 1

    # Construct single-particle H
    hamiltonian = construct_H_dense(params, basis_type='position')

    # Find two lowest lying eigenfunctions and normalise (L2)
    eigenenergy, eigenfunction = sp.sparse.linalg.eigsh(hamiltonian,2)

    # Init many-body wavefunction and construct as a Slater determinant of single-particle eigenfunctions
    wavefunction = pack_wavefunction(params,np.zeros(params.Nspace**2))
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):
            wavefunction[i,j] += (1 / np.sqrt(2)) * (np.conj(eigenfunction[i,0])*eigenfunction[j,1] - np.conj(eigenfunction[i,1])*eigenfunction[j,0])

    wavefunction = unpack_wavefunction(params,wavefunction)
    params.num_electrons = 2

    return wavefunction.real


def construct_H_sparse(params):
    r"""
    Constructs the Hamiltonian directly in sparse form using an antisymmetric delta function (position) basis
    """

    potential = np.zeros((params.Nspace,params.Nspace))
    laplace = discrete_Laplace(params)
    hamiltonian = np.zeros((params.Nspace**2,params.Nspace**2))

    # Construct local potential matrix
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):

            # Add Coulomb
            potential[i,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

            # Add external
            potential[i,j] += params.v_ext[i] + params.v_ext[j]

    # Construct Hamiltonian as KE + V
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):

            # Add potential term (V)
            hamiltonian[i * params.Nspace + j, j * params.Nspace + i] -= potential[i,j]
            hamiltonian[i * params.Nspace + j, i * params.Nspace + j] += potential[i,j]

            for k in range(0,params.Nspace):

                # Add (non-local) kinetic term (KE)
                hamiltonian[i*params.Nspace + j,i*params.Nspace+k] += -0.5*laplace[j,k]
                hamiltonian[j*params.Nspace + i,k*params.Nspace+i] += -0.5*laplace[j,k]
                hamiltonian[i*params.Nspace + j,k*params.Nspace+i] -= -0.5*laplace[j,k]
                hamiltonian[j*params.Nspace + i,i*params.Nspace+k] -= -0.5*laplace[j,k]

    hamiltonian *= 0.5

    return sp.sparse.csr_matrix(hamiltonian)


def construct_H_dense(params,basis_type):
    r"""
    Constructs the (dense) Hamiltonian in a delta function basis (tensor form) for an N particle system.
    Position space basis (basis_type = position) or (basis_type =)position_antisymmetric -- delta function.
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

        if basis_type == 'position':

            # Kinetic energy
            for i in range(0,params.Nspace):
                hamiltonian[i,:,i,:] += -0.5*discrete_Laplace(params)
                hamiltonian[:,i,:,i] += -0.5*discrete_Laplace(params)

            for j in range(0,params.Nspace):
                for i in range(0,params.Nspace):

                    # External Potential
                    hamiltonian[i,j,i,j] += params.v_ext[i] + params.v_ext[j]

                    # Softened Coulomb potential
                    hamiltonian[i,j,i,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

        # Add components corresponding to the antisymmetric parts of the basis
        elif basis_type == 'position_antisymmetric':

            laplace = discrete_Laplace(params)

            # Kinetic energy
            for i in range(0,params.Nspace):
                hamiltonian[i,:,i,:] += -0.5*laplace[:,:]
                hamiltonian[:,i,:,i] += -0.5*laplace[:,:]
                hamiltonian[i,:,:,i] -= -0.5*laplace[:,:]
                hamiltonian[:,i,i,:] -= -0.5*laplace[:,:]

            for j in range(0,params.Nspace):
                for i in range(0,params.Nspace):
                    # External potential
                    hamiltonian[i,j,j,i] -= params.v_ext[i] + params.v_ext[j]
                    hamiltonian[i,j,i,j] += params.v_ext[i] + params.v_ext[j]

                    # Softened Coulomb potential
                    hamiltonian[i,j,j,i] -= 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)
                    hamiltonian[i,j,i,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

            hamiltonian *= 0.5

    else:
        raise RuntimeError('Cannot construct the Hamiltonian for the given particle number: {}'.format(params.num_electrons))

    return hamiltonian


def dense_to_sparse(params,hamiltonian):
    r"""
    Takes a hamiltonian (in dense tensor form) and converts it to a csr (scipy) sparse matrix
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
                    hamiltonian_matrix[nu,mu] = hamiltonian[i,j,k,l]
                    mu += 1

    # Convert this output matrix (which is dense) to a sparse matrix
    hamiltonian_sparse = sp.sparse.csr_matrix(hamiltonian_matrix)

    return hamiltonian_sparse


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


def expansion_and_reduction_matrix(params):
    """
    Defines the matrix A that when applied to a general wavefunction (phi) gives its antisymmetric
    compliment (psi): psi = A phi. I.e. given an antisymmetric wavefunction psi, A^-1 psi = phi
    = the distinct components of psi.
    """

    # Number of distinct elements of the antisymmetric wavefunction
    num_distinct_elements = int( params.Nspace*(params.Nspace - 1) / 2 )


    # Number of matrix (A) elements required to make the transformation psi_full(x,t) = A*psi_reduced(x,t)
    col, row, entries = np.zeros(2*num_distinct_elements + 1, dtype=np.int), \
                        np.zeros(2*num_distinct_elements + 1, dtype=np.int), \
                        np.ones(2*num_distinct_elements + 1, dtype=np.int)

    # Construct an array (counter) of the relevant (i,j) pairs in order to directly populate sparse matrix
    counter = np.zeros((num_distinct_elements,2))
    k = 0
    for i in range(0,params.Nspace):
        j = 0
        while j < i:
            counter[k,:] = [i,j]
            j += 1
            k += 1

    # Occupy col, row, and entries of the sparse matrix
    k = 0
    for i in range(0,num_distinct_elements):

        # Extracts the relevant element into (i,j)
        col[k] = i
        row[k] = params.Nspace*counter[i,0] + counter[i,1]
        entries[k] = 1

        # Extracts negative of the relevant element into (j,i)
        col[k+1] = i
        row[k+1] = params.Nspace*counter[i,1] + counter[i,0]
        entries[k+1] = -1

        k += 2

    # Final row of zeros to deal with zeroing of the diagonal
    col[2*num_distinct_elements] = 0
    row[2*num_distinct_elements] = params.Nspace**2 - 1
    entries[2*num_distinct_elements] = 0

    # The sparse matrix that when applied to a wavefunction phi returns its antisymmetric expansion
    antisymm_expansion = sp.sparse.csr_matrix((entries, (row, col)))
    # The inverse of the above operation
    antisymm_reduction = 0.5*antisymm_expansion.getH()

    # Output as a sparse matrix
    return antisymm_expansion, antisymm_reduction