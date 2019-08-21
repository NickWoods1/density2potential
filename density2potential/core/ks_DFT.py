import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from density2potential.utils.physics import element_charges, calculate_density_ks
from density2potential.utils.math import discrete_Laplace, normalise_function

"""
Computes the self-consistent Kohn-Sham orbitals, density, and energy given a density functional and external
potential 
"""


def minimise_energy(params):

    # Array that will store SCF iterative densities and residuals
    history_of_densities_in = np.zeros((params.history_length, params.Nspace_dft))
    history_of_densities_out = np.zeros((params.history_length, params.Nspace_dft))
    history_of_residuals = np.zeros((params.history_length, params.Nspace_dft))
    density_differences = np.zeros((params.history_length, params.Nspace_dft))
    residual_differences = np.zeros((params.history_length, params.Nspace_dft))

    # Calculate number of particles
    num_particles = 0
    elements = element_charges(params)
    for i in range(0,len(params.species)):
        num_particles += elements[params.species[i]]
    params.num_electrons = num_particles

    # Generate initial guess density (sum weighted Gaussians)
    density_in = initial_guess_density(params)

    # SCF loop
    i, error = 0, 1
    while error > 1e-10:

        # Iteration number modulus history length
        i_mod = i % params.history_length
        i_mod_prev = (i-1) % params.history_length

        # Construct Hamiltonian
        hamiltonian = construct_ks_hamiltonian(params, density_in)

        # Solve H psi = E psi
        eigenvalues, eigenvectors = sp.linalg.eigh(hamiltonian)

        # Extract lowest lying num_particles eigenfunctions and normalise
        wavefunctions_ks = eigenvectors[:,0:num_particles]
        wavefunctions_ks[:,0:num_particles] = normalise_function(params, wavefunctions_ks[:,0:num_particles])

        # Calculate the output density
        density_out = calculate_density_ks(params, wavefunctions_ks)

        # Calculate total energy
        energy = calculate_total_energy(params, eigenvalues[0:num_particles], density_out)

        # L1 error between input and output densities
        error = np.sum(abs(density_in - density_out)*params.dx_dft)
        print('SCF error = {0} at iteration {1} with energy {2}'.format(error, i, energy))

        # Store densities/residuals within the iterative history data
        history_of_densities_in[i_mod,:] = density_in
        history_of_densities_out[i_mod,:] = density_out
        history_of_residuals[i_mod,:] = density_out - density_in

        if i == 0:

            # Damped linear step for the first iteration
            density_in = density_in - params.step_length * (density_in - density_out)

        elif i > 0:

            # Store more iterative history data...
            density_differences[i_mod_prev] = history_of_densities_in[i_mod] - history_of_densities_in[i_mod_prev]
            residual_differences[i_mod_prev] = history_of_residuals[i_mod] - history_of_residuals[i_mod_prev]

            # Perform Pulay step using the iterative history data
            density_in = pulay_mixing_kresse(params, density_differences, residual_differences,
                                             history_of_residuals[i_mod], history_of_densities_in[i_mod], i)

        i += 1

    plt.plot(density_out)
    plt.show()


def pulay_mixing_kresse(params, density_differences, residual_differences, current_residual, current_density_in, i):
    r"""
    As shown in Kresse (1998)
    """

    # Allocates arrays appropriately before and after max history size is reached
    if i >= params.history_length:
        history_size = params.history_length
    elif i < params.history_length:
        history_size = i

    # The Pulay residual dot product matrix
    pulay_matrix = np.zeros((history_size,history_size))

    # The RHS vector for the Pulay linear system Ax=b
    b = np.zeros(history_size)
    for j in range(0,history_size):
        b[j] = -np.dot(residual_differences[j,:], current_residual[:])

    # Construct Pulay matrix
    k = 0
    for j in range(0,history_size):
        for k in range(0,history_size):

                # Pulay matrix is matrix of dot products of residuals
                pulay_matrix[j,k] = np.dot(residual_differences[j],residual_differences[k])
                pulay_matrix[k,j] = pulay_matrix[j,k]

                k += 1

    # Solve for the (Pulay) optimal coefficients
    pulay_coefficients = np.linalg.solve(pulay_matrix, b)

    # Final Pulay update: n_new = n_opt + \alpha R_opt
    density_in = current_density_in + params.step_length*current_residual
    for j in range(0,history_size):
        density_in[:] += pulay_coefficients[j]*(density_differences[j,:] + params.step_length*residual_differences[j,:])

    # Pulay predicing negative densities?!
    density_in = abs(density_in)

    return density_in


def initial_guess_density(params):
    r"""
    Generate an initial guess for the density: Gaussians centered on atoms scaled by charge
    """

    # Dict of elements + charges
    elements = element_charges(params)
    density = np.zeros(params.Nspace_dft)

    i = 0
    while i < len(params.species):

        charge = elements[params.species[i]]
        density += charge*np.exp(-(params.grid_dft - params.position[i])**2)
        i += 1

    density *= len(params.species)*(np.sum(density)*params.dx_dft)**-1

    return density


def construct_v_ext(params):
    r"""
    Constructs the external potential generated by atoms
    """

    # Dict of elements + charges
    elements = element_charges(params)

    i = 0
    v_ext = np.zeros(params.Nspace_dft)
    while i < len(params.species):

        charge = elements[params.species[i]]
        v_ext += coulomb(params.Nspace_dft, params.grid_dft, charge, params.position[i], params.soft)

        i += 1

    np.save('v_ext.npy',v_ext)

    return v_ext


def coulomb(Nspace_dft, grid_dft, charge, position, soft):
    r"""
    Creates a softened Coulomb potential at a given atomic position scaled with atomic
    charge (+ve int)
    """

    potential = np.zeros(Nspace_dft)

    for i in range(0,Nspace_dft):
        potential[i] = -charge / (abs(grid_dft[i] - position) + soft)

    return potential


def construct_ks_hamiltonian(params, density):
    r"""
    Constructs the KS Hamiltonian given a density, particle number, external potential, etc.
    """

    laplace = discrete_Laplace(params)
    hamiltonian = np.zeros((params.Nspace_dft,params.Nspace_dft))

    # Add Kinetic energy
    hamiltonian += -0.5*laplace

    # Add external potential
    v_ext = construct_v_ext(params)
    hamiltonian += np.diag(v_ext)

    # Add Hartree potential
    v_h = np.zeros(params.Nspace_dft)
    for i in range(0,params.Nspace_dft):
        for j in range(0,params.Nspace_dft):
            v_h[i] += density[j] / (abs(params.grid_dft[i] - params.grid_dft[j]) + params.soft)

    v_h *= params.dx_dft
    hamiltonian += np.diag(v_h)

    # Add XC potential (Entwistle 2017)
    #v_xc = (-1.24 + 2.1*density - 1.7*density**2)*density**0.61
    #hamiltonian += np.diag(v_xc)

    return hamiltonian


def calculate_total_energy(params, eigenenergies, density):
    r"""
    Calculates the total energy given a set of occupied eigenenergies
    and corresponding density
    """

    # Hartree energy
    # Add Hartree potential
    v_h = np.zeros(params.Nspace_dft)
    for i in range(0,params.Nspace_dft):
        for j in range(0,params.Nspace_dft):
            v_h[i] += density[j] / (abs(params.grid_dft[i] - params.grid_dft[j]) + params.soft)
    v_h *= params.dx_dft
    E_h = 0.5*np.sum(density * v_h)*params.dx_dft

    total_energy = np.sum(eigenenergies) - E_h

    return total_energy

