import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from density2potential.io.input_file import parameters
from density2potential.plot.animate import animate_function, animate_two_functions
from density2potential.core.ks_potential import generate_ks_potential
from density2potential.core.exact_TISE import solve_TISE
from density2potential.core.exact_TDSE import solve_TDSE

"""
Main hook for the requested action
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

        # Read in reference density
        density_reference = np.load('td_density.npy')

        # Create parameters object
        params = parameters()

        # Generate v_ks
        density_ks, v_ks, wavefunctions_ks = generate_ks_potential(params,density_reference)

        # Save and Graph output

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
        #animate_two_functions(params,density,density_idea,10,'exact_den','Exact TD Density me','density-idea')
        print(' ')
        print(' ')

        print('Finished successfully')

    # Plot files generated prior
    elif args.task == 'plot':

        params = parameters()

        den1 = np.load('TD_density.npy')
        den2 = np.load('TD_densityCN.npy')

        animate_two_functions(params, den1, den2, 5, 'compare_two_densities', 'expm', 'CN')

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



        # Set the reference density as the density computed from the exact calculation
        density_reference = density

        # Generate v_ks
        density_ks, v_ks, wavefunctions_ks = generate_ks_potential(params,density_reference)
