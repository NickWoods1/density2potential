from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

"""
Various plotting utilities
"""

def plot_surface_3d(params, matrix):
    """
    Interactive 3D surface plot of a matrix.
    """

    plt.close('all')

    # Set x, y (real space grid)
    x = np.outer(params.space_grid, np.ones(params.Nspace))
    y = x.copy().T

    # Create fig w/ 2 subplots for real and imag
    fig = plt.figure(figsize=(10,3))
    plot1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot2 = fig.add_subplot(1, 2, 2, projection='3d')

    plot1.plot_surface(x, y, matrix.real, cmap='YlGnBu')#, edgecolor='none')
    plot2.plot_surface(x, y, matrix.imag, cmap='viridis', edgecolor='none')

    plot1.set_title('Real')
    plot1.set_xlabel('x (a.u)')
    plot1.set_ylabel('y (a.u)')

    plot2.set_title('Imaginary')
    plot2.set_xlabel('x (a.u)')
    plot2.set_ylabel('y (a.u)')

    plt.show()


def plot_heat_map(params, matrix, figtitle='f(x,y)', figname='default.pdf'):
    """
    Heat map of the real and imag parts of a matrix
    """

    plt.close('all')

    # Create fig w/ 1x2 grid of subplots
    fig = plt.figure(figsize=(10,5))
    plot1 = fig.add_subplot(1, 2, 1)
    plot2 = fig.add_subplot(1, 2, 2)

    # Each plot is heatmap w/ colorbar on each
    heatmap1 = plot1.imshow(matrix.real, cmap='Spectral', origin='lower', interpolation='gaussian')
    heatmap2 = plot2.imshow(matrix.imag, cmap='jet', origin='lower', interpolation='gaussian')
    fig.colorbar(heatmap1, ax=plot1, fraction=0.046, pad=0.04)
    fig.colorbar(heatmap2, ax=plot2, fraction=0.046, pad=0.04)

    # Set labels
    fig.suptitle(figtitle)

    plot1.set_title("Real")
    plot1.set_xlabel("x (a.u.)")
    plot1.set_ylabel("x' (a.u.)")

    plot2.set_title("Imaginary")
    plot2.set_xlabel("x (a.u.)")
    plot2.set_ylabel("x' (a.u.)")

    # Plot...
    fig.tight_layout()
    plt.show()
    plt.savefig(figname)


def plot_eigenvector_decomp(params, matrix, figtitle='f(x,y)', dirname='default'):
    """
    Plot each component of a Hermition matrix (input) from its eigendecomposition:
    M(x,x') = \sum_i \lambda_i \psi_i(x)\psi_i(x')
    """

    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

    # Get eigenvectors
    eigv, eigf = np.linalg.eigh(matrix)

    # Sort eigenvalues and eigenvectors by absolute value
    args = np.argsort(abs(eigv))
    eigv_tmp, eigf_tmp = np.copy(eigv), np.copy(eigf)
    for i in range(len(eigv)):
        eigf_tmp[:,i] = eigf[:, args[i]]
        eigv_tmp[i] = eigv[args[i]]
    eigf, eigv = eigf_tmp, eigv_tmp

    # Plot
    for p in range(len(eigv)):

        # Create fig object w/ subplots
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        # Labels and titles
        fig.suptitle(figtitle)
        ax1.set_title(f'Eigenvector {p}')
        ax2.set_title('Eigenvalue {}'.format(round(eigv[p], 4)))
        ax3.set_xlabel('Grid Index')
        ax1.set_ylabel('Eigenvector')
        ax4.set_xlabel('x (index)')
        ax4.set_ylabel("x' (index)")

        # The plots...
        ax1.plot(eigf[:, p].real)
        ax3.plot(eigf[:, p].imag)
        im2 = ax2.imshow(np.outer(eigf[:, p], eigf[:, p].conj()).real, cmap='bwr', origin='lower', interpolation='gaussian')
        im4 = ax4.imshow(np.outer(eigf[:, p], eigf[:, p].conj()).imag, cmap='bwr', origin='lower', interpolation='gaussian')

        # Save
        plt.colorbar(im2, ax=ax2)
        plt.colorbar(im4, ax=ax4)

        plt.savefig('{0}/{1}.png'.format(dirname,p))
        plt.close('all')


def plot_build_eigenvector_decomp(params, matrix, figtitle='f(x,y)', dirname='default'):
    """
    Build a Hermition matrix M from its eigendecomposition:
    M(x,x') = \sum_i \lambda_i \psi_i(x)\psi_i(x')
    where plot p is the matrix constructed from p lowest lying eigenvectors
    """

    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

    # Get eigenvectors
    eigv, eigf = np.linalg.eigh(matrix)

    # Sort eigenvalues and eigenvectors by absolute value
    args = np.argsort(abs(eigv))
    eigv_tmp, eigf_tmp = np.copy(eigv), np.copy(eigf)
    for i in range(len(eigv)):
        eigf_tmp[:,i] = eigf[:, args[i]]
        eigv_tmp[i] = eigv[args[i]]
    eigf, eigv = eigf_tmp, eigv_tmp

    # Plot
    for p in range(len(eigv)):

        tmp = np.zeros((params.Nspace, params.Nspace), dtype=complex)
        for k in range(p + 1):
            tmp += eigv[k]*np.outer(eigf[:,k], eigf[:,k].conj())

        # Create fig object w/ subplots
        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        # Labels and titles
        fig.suptitle(figtitle)
        ax1.set_title(f'Eigenvector {p}')
        ax2.set_title('Eigenvalue {}'.format(round(eigv[p], 4)))
        ax3.set_xlabel('Grid Index')
        ax1.set_ylabel('Eigenvector')
        ax4.set_xlabel('x (index)')
        ax4.set_ylabel("x' (index)")

        # The plots...
        ax1.plot(eigf[:, p].real)
        ax3.plot(eigf[:, p].imag)
        im2 = ax2.imshow(tmp.real, cmap='Spectral', origin='lower', interpolation='gaussian')
        im4 = ax4.imshow(tmp.imag, origin='lower', interpolation='gaussian')

        # Save
        plt.colorbar(im2, ax=ax2)
        plt.colorbar(im4, ax=ax4)

        plt.savefig('{0}/{1}.png'.format(dirname,p))
        plt.close('all')
