import numpy as np

"""
Computes the self-consistent Kohn-Sham orbitals, density, and energy given a density functional and external
potential 
"""

def minimise_energy():

    # Initial guess rho
    # Construct H[rho]
    # Diag for lowest N+something states H psi = E psi
    # Is there a big gap?
    # get output density
    # density mixing, add history, pass history
    # Converged? yes/no


def density_mixing(params, density_history):

    # den_history[den, den_in, den_out]
    # Solve Pulay scheme
    # Generate new density


def ks_construct_H(params):

    # -0.5delsq phi
    # v_ext phi
    # v_xc[den] phi
    # v_hartree[den]

def construct_v_ext(params):

    # Dict of elements.
    # H, He, ...
    # Place Z/r at position
    # Supercell with zero b.c.s


def generate_supercell():   



def initial_guess_density():

    # for elements in dict
    # construct_v_ext(elements)
    # solve -0.5delsq + vext matrix test cell
    # get den(element)

    # Or add scaled Gaussian on each element Ze^(-x^2).

    # superimpose den(element)

    # Output density_initial[space]


