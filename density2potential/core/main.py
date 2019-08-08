import argparse
import numpy as np
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

    # Execute what the user has specified

    # Find the Kohn-Sham potential that generates a given reference density
    if args.task == 'find-vks':

        # Read in reference density
        density_reference = np.load('density_reference.npy')

        #density_reference = np.swapaxes(density_reference,0,1)

        # Create parameters object
        params = parameters(density_reference)

        # Generate v_ks
        density_ks, v_ks, wavefunctions_ks = generate_ks_potential(params,density_reference)

        # Save and Graph output

    # Solve exact QM for time-dependent wavefunctions, energies, densities, etc.
    elif args.task == 'exact':

        density_reference = np.load('den_idea.npy')

        # Create parameters object
        params = parameters(density_reference)

        # Solve the TISE for the ground-state wavefunction, density, and energy.
        wavefunction, density, energy = solve_TISE(params)

        # Solve the TDSE for the evolved wavefunction and density
        density = solve_TDSE(params, wavefunction)

        animate_function(params,density,10,'TD_den','density')
        #animate_two_functions(params,density,density_idea,10,'exact_den','Exact TD Density me','density-idea')