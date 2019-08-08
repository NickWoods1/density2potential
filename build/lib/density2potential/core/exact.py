import numpy as np
from density2potential.utils.math import discrete_Laplace, norm, normalise_function
from density2potential.plot.animate import animate_function
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize

"""
Solve the exact time-dependent Schrodinger equation in 1D to generate an exact density
"""

def solve_ground_state(params):
    r"""
    Solves the ground state Schrodinger equation
    """

    # Generate the initial guess wavefunction for the iterative diagonaliser
    # Slater determinant of single-particle wavefunctions
    if params.num_electrons > 1:
        wavefunction = initial_guess_wavefunction(params)

    # Construct full MB hamiltonian, and sparsify
    hamiltonian = construct_H(params,'antisymmetric')
    hamiltonian = operator_to_sparse_matrix(params,hamiltonian)

    # Add one to diagonal to regularise
    #hamiltonian += np.diag(np.ones(params.Nspace**2))

    # Find g.s. eigenvector and eigenvalue using Lanszcos algorithm
    #eigenenergy_gs, eigenfunction_gs = sp.sparse.linalg.eigsh(hamiltonian, 2, which='SM')#, v0=wavefunction)
    eigenenergy_gs, eigenfunction_gs = sp.linalg.eigh(hamiltonian)

    # Normalise g.s. eigenfunction
    eigenfunction_gs[:,0] *= (np.sum(eigenfunction_gs[:,0]**2) * params.dx**2)**-0.5

    # Ground state energy (n.b. undo the potential and regularisation shift)
    print('Ground state energy: {0}'.format(eigenenergy_gs[0] + 2.0*params.v_ext_shift - 1))

    print(eigenenergy_gs)

    # Revert to tensor form
    wavefunction = pack_wavefunction(params,eigenfunction_gs[:,0]).real

    # Compute density
    density = 2.0 * (np.sum(abs(wavefunction[:,:])**2,axis=0)) * params.dx

    plt.plot(params.v_ext)
    plt.plot(density)
    plt.show()

    np.save('groundstate_density', density)
    np.save('groundstate_wavefunction',wavefunction)

    wavefunction = unpack_wavefunction(params,wavefunction)

    return wavefunction, density, eigenenergy_gs[0]


def initial_guess_wavefunction(params):
    r"""
    Generate the initial guess for the many-body iterative diagonaliser: Slater determinant of single-particle states.
    """

    # Solve single particle equation
    params.num_electrons = 1

    # Construct single-particle H
    hamiltonian = construct_H(params,'general')

    # Find two lowest lying eigenfunctions and normalise (L2)
    eigenenergy, eigenfunction = sp.sparse.linalg.eigsh(hamiltonian,2)
    eigenfunction[:,0:1] = normalise_function(params,eigenfunction[:,0:1])

    # Init many-body wavefunction and construct as a Slater determinant of single-particle eigenfunctions
    wavefunction = pack_wavefunction(params,np.zeros(params.Nspace**2))
    for i in range(0,params.Nspace):
        for j in range(0,params.Nspace):
            wavefunction[i,j] += (1 / np.sqrt(0.5)) * (np.conj(eigenfunction[i,0])*eigenfunction[j,1] - np.conj(eigenfunction[i,1])*eigenfunction[j,0])

    wavefunction = unpack_wavefunction(params,wavefunction)
    params.num_electrons = 2

    return wavefunction.real


def solve_time_dependence(params, wavefunction, density_gs):
    r"""
    Solve the TDSE for a given initial wavefunction, and external potential v(x,t)
    """

    np.save('TD_external_potential', params.v_ext_td)

    density = np.zeros((params.Ntime,params.Nspace))
    density[0,:] = density_gs

    if params.time_step_method == 'expm':

        params.v_ext = params.v_ext_td[1,:]
        hamiltonian = construct_H(params,'antisymmetric')
        hamiltonian = operator_to_sparse_matrix(params, hamiltonian)
        for i in range(1,params.Ntime):
            wavefunction = expm_step(params,wavefunction, hamiltonian)
            density[i,:] = np.sum(abs(pack_wavefunction(params,wavefunction)[:,:])**2, axis=0)
            print('Time passed: {}'.format(round(params.time_grid[i],3)))

    elif params.time_step_method == 'CN':

        for i in range(0,params.Ntime):
            print(i)

    else:
        raise RuntimeError('Not a valid time-stepping method: {}'.format(params.time_step_method))

    np.save('timedependent_density', density)

    return density


def expm_step(params, wavefunction, hamiltonian):

    #hamiltonian = construct_H(params,'antisymmetric')
    #hamiltonian = operator_to_sparse_matrix(params,hamiltonian)
    wavefunction = sp.sparse.linalg.expm_multiply(1.0j*params.dt*hamiltonian, wavefunction)

    return wavefunction


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

        if basis_type == 'general':

            # Kinetic energy
            for i in range(0,params.Nspace):
                hamiltonian[i,:,i,:] += -0.5*discrete_Laplace(params)
                hamiltonian[:,i,:,i] += -0.5*discrete_Laplace(params)

            # External potential
            for j in range(0,params.Nspace):
                for i in range(0,params.Nspace):
                    hamiltonian[i,j,i,j] += params.v_ext[i] + params.v_ext[j]

            # Softened Coulomb Potential
            for i in range(0,params.Nspace):
                for j in range(0,params.Nspace):
                    hamiltonian[i,j,i,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

        # Add components corresponding to the antisymmetric parts of the basis
        elif basis_type == 'antisymmetric':

            laplace = discrete_Laplace(params)

            # Kinetic energy
            for i in range(0,params.Nspace):
                hamiltonian[i,:,i,:] += -0.5*laplace[:,:]
                hamiltonian[:,i,:,i] += -0.5*laplace[:,:]
                hamiltonian[i,:,:,i] -= -0.5*laplace[:,:]
                hamiltonian[:,i,i,:] -= -0.5*laplace[:,:]

            # External potential
            for j in range(0,params.Nspace):
                for i in range(0,params.Nspace):
                    hamiltonian[i,j,j,i] -= params.v_ext[i] + params.v_ext[j]
                    hamiltonian[i,j,i,j] += params.v_ext[i] + params.v_ext[j]

            # Softened Coulomb Potential
            for i in range(0,params.Nspace):
                for j in range(0,params.Nspace):
                        hamiltonian[i,j,j,i] -= 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)
                        hamiltonian[i,j,i,j] += 1 / (abs(params.space_grid[i] - params.space_grid[j]) + 1)

            hamiltonian *= 0.5

    else:
        raise RuntimeError('Cannot construct the Hamiltonian for the given particle number: {}'.format(params.num_electrons))

    return hamiltonian


def construct_H_sparse(params):
    r"""
    Constructs the Hamiltonian directly in sparse form
    """

    # Compute # of non-zero elements in the subspace transformation and init the arrays that hold these elements
    col, row, entries = np.zeros(num_nonzero_elements, dtype=np.int), \
                        np.zeros(num_nonzero_elements, dtype=np.int), \
                        np.ones(num_nonzero_elements, dtype=np.int)


    col = np.zeros(int(params.Nspace/2)*(params.Nspace-1), dtype=np.int)
    row = np.zeros(int(params.Nspace/2)*(params.Nspace-1), dtype=np.int)
    entries = np.ones(int(params.Nspace/2)*(params.Nspace-1), dtype=np.int)


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
                    hamiltonian_matrix[nu,mu] = hamiltonian[i,j,k,l]
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

