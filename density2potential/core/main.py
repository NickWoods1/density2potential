import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from density2potential.io.input_file import parameters
from density2potential.plot.animate import animate_function, animate_two_functions
from density2potential.plot.matrix_plot import plot_surface_3d, plot_heat_map, plot_eigenvector_decomp, plot_build_eigenvector_decomp
from density2potential.core.ks_potential import generate_ks_potential, generate_gsks_potential, hartree_potential, xc_potential
from density2potential.core.exact_TISE import solve_TISE
from density2potential.core.exact_TDSE import solve_TDSE
from density2potential.utils.math import discrete_Laplace

#/TODO DELETE AFTER TESTING DONE
from density2potential.core.linear_response import exact_susceptibility, ks_susceptibility, \
    xc_kernel, dyson, sum_rule
from density2potential.core.exact_TISE import expansion_and_reduction_matrix, construct_H_sparse,  \
    initial_guess_wavefunction, calculate_density_exact, pack_wavefunction
from density2potential.core.ks_potential import construct_H
from density2potential.utils.physics import calculate_density_ks
from density2potential.utils.math import normalise_function
import scipy as sp
from scipy.optimize import root, minimize


"""
Entry point for the requested action
"""


def main():

    __version__ = 0.1

    # Parser class for density2potential
    parser = argparse.ArgumentParser(
        prog='density2potential',
        description='Find the Kohn-Sham effective potential that '
                    ' generates a given density..',
        epilog='written by Nick Woods')

    # Specify arguments that the package can take
    parser.add_argument('--version', action='version', version='This is version {0} of density2potential.'.format(__version__))
    parser.add_argument('task', help='what do you want density2potential to do: find-vks, exact')

    args = parser.parse_args()

    # Code header
    print('    ██████╗ ██████╗ ██████╗')
    print('    ██╔══██╗╚════██╗██╔══██╗')
    print('    ██║  ██║ █████╔╝██████╔╝')
    print('    ██║  ██║██╔═══╝ ██╔═══╝')
    print('    ██████╔╝███████╗██║')
    print('    ╚═════╝ ╚══════╝╚═╝')
    print('    ------------------------')
    print('       density2potential')
    print('    ------------------------')
    print('           Written by')
    print('           Nick Woods')
    print(' ')

    # Find the Kohn-Sham potential that generates a given reference density
    if args.task == 'find-vks':

        # Read in the parameters used to generate reference density
        params_save = open('params.obj', 'rb')
        params = pickle.load(params_save)

        # Read in reference density
        density_reference = np.load('td_density.npy')

        # Generate v_ks
        density_ks, v_ks, wavefunctions_ks = generate_ks_potential(params,density_reference)

        # Animate the time-dependent Kohn-Sham potential
        print('Animating output...')
        animate_function(params, v_ks, 10, 'td_ks_potential','KS potential')
        print(' ')
        print(' ')

        # Save and Graph output
        plt.clf()
        plt.plot(density_ks[0,:], label='Ground state KS density')
        plt.plot(v_ks[0,:], label='Ground state KS potential')
        plt.legend()
        plt.savefig('groundstate_den_and_vks.pdf')

        np.save('td_ks_potential', v_ks)
        np.save('td_ks_density', density_ks)
        np.save('td_ks_wavefunctions', wavefunctions_ks)

        print('Finished successfully')

    # Solve exact QM for time-dependent wavefunctions, energies, densities, etc.
    elif args.task == 'exact':

        # Create parameters object
        params = parameters()

        # Save the parameters object for the run
        params_save = open('params.obj', 'wb')
        pickle.dump(params, params_save)

        # Save the external potential for the run
        np.save('td_external_potential', params.v_ext_td)

        # Solve the TISE for the ground-state wavefunction, density, and energy.
        print('Solving the TISE...')
        wavefunction, density, energy = solve_TISE(params)
        print(' ')

        gradv = np.gradient(params.v_ext, params.dx)
        x = np.sum(density*gradv)*params.dx
        print(x)
        plt.plot(density)
        plt.show()

        # Plot ground state quantities
        plt.plot(density, label='Ground state density')
        plt.plot(params.v_ext - np.amin(params.v_ext), label='External potential')
        plt.title('Ground state energy = {} a.u.'.format(energy))
        plt.legend()
        plt.savefig('groundstate_den_and_vext.pdf')

        # Save wavefunction and density
        np.save('groundstate_density', density)
        np.save('groundstate_wavefunction', wavefunction)

        # Solve the TDSE for the evolved wavefunction and density, starting from initial wavefunction
        print('Solving the TDSE...')
        density = solve_TDSE(params, wavefunction)
        print('Time passed: {}'.format(round(params.time,3)))
        print(' ')

        # Save time-dependent density
        np.save('td_density', density)

        # Animate the time-dependent density
        print('Animating output...')
        animate_function(params, density, 10, 'td_density','density')
        print(' ')
        print(' ')

        print('Finished successfully')

    # Solve exact TDSE, then reverse-engineer to get v_ks
    elif args.task == 'exact-then-vks':

        # Create parameters object
        params = parameters()

        # Solve the TISE for the ground-state wavefunction, density, and energy.
        print('Solving the TISE...')
        wavefunction, density, energy = solve_TISE(params)
        print(' ')

        # Solve the TDSE for the evolved wavefunction and density, starting from initial wavefunction
        print('Solving the TDSE...')
        density = solve_TDSE(params, wavefunction)
        print('Time passed: {}'.format(round(params.time,3)))
        print(' ')

        # Animate the time-dependent density
        print('Animating output...')
        animate_function(params, density, 10, 'td_density','density')

        # Set the reference density as the density computed from the exact calculation
        density_reference = density

        # Generate v_ks
        density_ks, v_ks, wavefunctions_ks = generate_ks_potential(params,density_reference)

        # Save relevant objects
        np.save('td_ks_potential', v_ks)
        np.save('td_ks_density', density_ks)
        np.save('td_ks_wavefunctions', wavefunctions_ks)

    elif args.task == 'LR':

        # EXPERIMENTAL JUNK
        # \TODO TIDY THIS UP

        params = parameters()

        ###### GET ALL EIGENSTATES OF EXACT 2 PARTICLE H ############
        antisymm_expansion, antisymm_reduction = expansion_and_reduction_matrix(params)
        hamiltonian = construct_H_sparse(params, basis_type='position')
        hamiltonian = hamiltonian.dot(antisymm_expansion)
        hamiltonian = antisymm_reduction.dot(hamiltonian)
        eigenenergy_gs, eigenfunction_gs = sp.linalg.eigh(sp.sparse.csr_matrix.todense(hamiltonian))
        wavefunction = np.zeros((len(eigenenergy_gs), params.Nspace**2), dtype=complex)
        for i in range(len(eigenenergy_gs)):
            wavefunction[i,:] = antisymm_expansion.dot(eigenfunction_gs[:,i])
            wavefunction[i,:] *= (np.sum(wavefunction[i,:]**2) * params.dx**2)**-0.5
        wvfn_packed = np.zeros((len(eigenenergy_gs), params.Nspace, params.Nspace), dtype=complex)
        for i in range(len(eigenenergy_gs)):
            wvfn_packed[i,:,:] = pack_wavefunction(params, wavefunction[i,:])
        #############################################################

        #### Reverse engineered KS system #######
        density_reference = calculate_density_exact(params, wavefunction[0,:])
        density_ks, v_ks, wavefunctions_ks, eigenenergies_ks = generate_gsks_potential(params,density_reference)
        #########################################

        ######## SAVE QUANTITIES #########
        plt.plot(density_reference, label='Ground state density')
        plt.plot(params.v_ext - np.amin(params.v_ext), label='External potential')
        plt.legend()
        plt.savefig('groundstate_den_and_vext.pdf')
        #################################

        ######## RESPONSE ###########
        exact_response = exact_susceptibility(params, wvfn_packed, eigenenergy_gs)
        ks_response = ks_susceptibility(params, wavefunctions_ks, eigenenergies_ks)

        """
        identity = np.eye(params.Nspace)
        alpha = 0.0000001
        exact_response += alpha * identity
        ks_response += alpha * identity
        """
        f_xc = xc_kernel(params, exact_response, ks_response)
        exact_response_dyson = dyson(params, ks_response, f_xc)

        error = np.sum(abs(exact_response_dyson - exact_response)) / params.Nspace**2
        print(f'Error in ||chi - chi_dyson|| is {error}')

        v_xc = xc_potential(params, density_ks, v_ks)

        #sum_rule_tests(params, f_xc, ks_response, exact_response, density_ks, v_xc)
        sum_rule(params, f_xc, density_ks, v_xc)


        plt.close('all')
        fig = plt.figure()
        a = fig.add_subplot(2,2,1)
        b = fig.add_subplot(2,2,2)
        c = fig.add_subplot(2,2,3)
        d = fig.add_subplot(2,2,4)

        a.plot(np.diag(f_xc.real))
        a.set_title("Diagonal x=x'")
        a.set_xlabel("x")
        a.set_ylabel("fxc")

        b.plot(f_xc[:,55].real)
        b.set_title("|x-x'| along x=0")

        c.plot(f_xc[:,15].real)
        c.set_title("|x-x'| along x=1.5")

        d.plot(f_xc[:,0].real)
        d.set_title("|x-x'| along x=5")

        fig.tight_layout()
        plt.savefig("f_xc_axes.pdf")

        plot_surface_3d(params, f_xc.real)
        plot_heat_map(params, f_xc.real, figtitle='xc kernel, f_xc', figname='xc_kernel.pdf')
        plot_eigenvector_decomp(params, f_xc.real, figtitle='xc kernel', dirname='exact_xc_kernel')
        plot_build_eigenvector_decomp(params, f_xc.real, figtitle='xc kernel', dirname='exact_xc_kernel_build')

    elif args.task == '3dshow':

        params = parameters()

        den = np.load('density.npy')
        plt.plot(den)
        plt.show()
        # Read in the parameters used to generate reference density
        f_xc = np.load('f_xc.npy')
        plot_heat_map(params, f_xc)
        #plot_surface_3d(params, f_xc)
        plot_eigenvector_decomp(params,f_xc.real)
        #plot_build_eigenvector_decomp(params, f_xc.real, dirname='test')

    elif args.task == 'exact_perturbation':

        params = parameters()

        ###### GET ALL EIGENSTATES OF EXACT 2 PARTICLE H ############
        antisymm_expansion, antisymm_reduction = expansion_and_reduction_matrix(params)
        hamiltonian = construct_H_sparse(params, basis_type='position')
        hamiltonian = hamiltonian.dot(antisymm_expansion)
        hamiltonian = antisymm_reduction.dot(hamiltonian)
        eigenenergy_gs, eigenfunction_gs = sp.linalg.eigh(sp.sparse.csr_matrix.todense(hamiltonian))
        wavefunction = np.zeros((len(eigenenergy_gs), params.Nspace**2), dtype=complex)
        for i in range(len(eigenenergy_gs)):
            wavefunction[i,:] = antisymm_expansion.dot(eigenfunction_gs[:,i])
            wavefunction[i,:] *= (np.sum(wavefunction[i,:]**2) * params.dx**2)**-0.5
        wvfn_packed = np.zeros((len(eigenenergy_gs), params.Nspace, params.Nspace), dtype=complex)
        for i in range(len(eigenenergy_gs)):
            wvfn_packed[i,:,:] = pack_wavefunction(params, wavefunction[i,:])
        #############################################################

        #### Reverse engineered KS system #######
        density_reference = calculate_density_exact(params, wavefunction[0,:])
        n0, vks0, wavefunctions_ks, eigenenergies_ks = generate_gsks_potential(params,density_reference)
        vxc0 = xc_potential(params, n0, vks0)
        #########################################

        ######## RESPONSE ###########
        exact_response = exact_susceptibility(params, wvfn_packed, eigenenergy_gs)
        ks_response = ks_susceptibility(params, wavefunctions_ks, eigenenergies_ks)
        f_xc = xc_kernel(params, exact_response, ks_response)
        ##############################

        params.v_ext += 0.001*(np.sin(params.space_grid))**2

        ######## Perturbed system ##############
        ###### GET ALL EIGENSTATES OF EXACT 2 PARTICLE H ############
        antisymm_expansion, antisymm_reduction = expansion_and_reduction_matrix(params)
        hamiltonian = construct_H_sparse(params, basis_type='position')
        hamiltonian = hamiltonian.dot(antisymm_expansion)
        hamiltonian = antisymm_reduction.dot(hamiltonian)
        eigenenergy_gs, eigenfunction_gs = sp.linalg.eigh(sp.sparse.csr_matrix.todense(hamiltonian))
        wavefunction = np.zeros((len(eigenenergy_gs), params.Nspace**2), dtype=complex)
        for i in range(len(eigenenergy_gs)):
            wavefunction[i,:] = antisymm_expansion.dot(eigenfunction_gs[:,i])
            wavefunction[i,:] *= (np.sum(wavefunction[i,:]**2) * params.dx**2)**-0.5
        wvfn_packed = np.zeros((len(eigenenergy_gs), params.Nspace, params.Nspace), dtype=complex)
        for i in range(len(eigenenergy_gs)):
            wvfn_packed[i,:,:] = pack_wavefunction(params, wavefunction[i,:])
        #############################################################

        #### Reverse engineered KS system #######
        density_reference = calculate_density_exact(params, wavefunction[0,:])
        n, vks, wavefunctions_ks, eigenenergies_ks = generate_gsks_potential(params,density_reference)
        vxc = xc_potential(params, n, vks)
        #########################################

        n1 = n - n0
        vxc1_from_fxc = ((f_xc @ n1)*params.dx).real
        vxc1 = vxc - vxc0

        def objective_function(x, v1, v2):
            v2 += x
            return np.sum(abs(v2 - v1))
        opt_info = root(objective_function, 0, args=(np.copy(vxc1), np.copy(vxc1_from_fxc),
                                                                      ), method='hybr', tol=1e-16)
        vxc1_from_fxc += opt_info.x # np.max(vxc1) - np.max(vxc1_from_fxc)


        error = np.sum(abs(vxc1_from_fxc - (vxc-vxc0)))
        print('Error between exact vxc1 and vxc1 from fxc: {}'.format(error*params.dx))

        plt.plot(vxc1)
        plt.plot(vxc1_from_fxc)
        plt.show()

        grad_vxc0 = np.gradient(vxc0)
        grad_vxc = np.gradient(vxc)
        print('Perturbed system zero force agreement: {}'.format(np.dot(grad_vxc,n)*params.dx))
        print('Unperturbed system zero force agreement: {}'.format(np.dot(grad_vxc0,n0)*params.dx))

        grad_vxc0 = np.gradient(vxc0, params.dx)
        grad_vxc1 = np.gradient(vxc1, params.dx)
        a = n0*grad_vxc0 + n0*grad_vxc1 + n1*grad_vxc0 + n1*grad_vxc1
        error = np.sum(a)*params.dx
        print('Perturbed zero force explicit: {}'.format(error))

        grad_vxc0 = np.gradient(vxc0, params.dx)
        grad_vxc1 = np.gradient(vxc1_from_fxc, params.dx)
        a = n0*grad_vxc0 + n0*grad_vxc1 + n1*grad_vxc0 + n1*grad_vxc1
        error = np.sum(a)*params.dx
        print('Perturbed zero force explicit from fxc: {}'.format(error))

