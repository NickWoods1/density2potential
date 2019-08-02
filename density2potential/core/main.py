import argparse
import numpy as np
from density2potential.io.input_file import parameters
from density2potential.core.ks_potential import generate_ks_potential
from density2potential.core.exact import solve_ground_state

"""
Main structure: given a density in the form of a 2D array, find the
corresponding Kohn-Sham potential that generates this density.
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
    parser.add_argument('task', help='what do you want density2potential to do: get_vks')

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

        # Create parameters object
        params = parameters(density_reference)

        # Solve the TISE for the ground-state wavefunction, density, and energy.
        solve_ground_state(params)

        # Solve the TDSE for the evolved wavefunction and density