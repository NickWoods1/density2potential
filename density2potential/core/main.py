import argparse
import numpy as np
from density2potential.io.input_file import parameters
from density2potential.core.ks_potential import generate_ks_potential

"""
Main structure: given a density in the form of a 2D array, find the
corresponding Kohn-Sham potential that generates this density.
"""

def main():

    __version__ = 0.1

    # Parser class for runscf
    parser = argparse.ArgumentParser(
        prog='density2potential',
        description='Given a time-dependent density in the form of a 2D array,'
                    ' find the Kohn-Sham potential that generates it.',
        epilog='written by Nick Woods')

    # Specify arguments that the package can take
    parser.add_argument('--version', action='version', version='This is version {0} of density2potential.'.format(__version__))
    parser.add_argument('task', help='what do you want density2potential to do: get_vks')

    args = parser.parse_args()

    # Execute what the user has specified
    if args.task == 'get_vks':

        # Read in reference density
        density_reference = np.load('density_reference.npy')

        # Create parameters object
        params = parameters(density_reference)

        # Generate v_ks
        density_ks, v_ks, wavefunctions_ks = generate_ks_potential(params,density_reference)

        # Graph output