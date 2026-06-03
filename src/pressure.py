"""FFT-based solver for the constant-coefficient pressure Poisson equation.

Solves  ∇²p = rhs  on periodic domains using the discrete eigenvalues of
the 2nd-order central-difference Laplacian.

This is the key property exploited by Roccon (2025): because the
one-fluid Navier-Stokes formulation yields a *constant-coefficient*
Poisson equation for pressure (Eq. 18), FFT-based direct solvers apply
without any iterative outer loop.  The same structure is preserved in 3-D,
making the extension straightforward and directly GPU-portable.
"""
import numpy as np


def _eig_laplacian_1d(N, h):
    """Discrete eigenvalues of the 1-D 2nd-order Laplacian on N points."""
    k = np.fft.fftfreq(N) * N          # integer wavenumbers: 0,1,...,N/2,-N/2+1,...,-1
    return (2 * np.cos(2 * np.pi * k / N) - 2) / h**2


def solve_poisson_2d(rhs, dx, dy):
    """
    Solve  ∇²p = rhs  (2-D, periodic BCs) via FFT.

    Parameters
    ----------
    rhs : (Ny, Nx) array
    dx, dy : grid spacing

    Returns
    -------
    p : (Ny, Nx) array with zero mean
    """
    Ny, Nx = rhs.shape
    rhs_hat = np.fft.fft2(rhs)

    lam_x = _eig_laplacian_1d(Nx, dx)   # shape (Nx,)
    lam_y = _eig_laplacian_1d(Ny, dy)   # shape (Ny,)
    LAM = lam_y[:, None] + lam_x[None, :]  # (Ny, Nx) eigenvalue matrix

    # Avoid division by zero at the DC mode (k=0)
    LAM[0, 0] = 1.0
    p_hat = rhs_hat / LAM
    p_hat[0, 0] = 0.0          # enforce zero-mean pressure

    return np.real(np.fft.ifft2(p_hat))


def solve_poisson_3d(rhs, dx, dy, dz):
    """
    Solve  ∇²p = rhs  (3-D, periodic BCs) via FFT.

    Parameters
    ----------
    rhs : (Nz, Ny, Nx) array

    Returns
    -------
    p : (Nz, Ny, Nx) array with zero mean
    """
    Nz, Ny, Nx = rhs.shape
    rhs_hat = np.fft.fftn(rhs)

    lam_x = _eig_laplacian_1d(Nx, dx)
    lam_y = _eig_laplacian_1d(Ny, dy)
    lam_z = _eig_laplacian_1d(Nz, dz)
    LAM = lam_z[:, None, None] + lam_y[None, :, None] + lam_x[None, None, :]

    LAM[0, 0, 0] = 1.0
    p_hat = rhs_hat / LAM
    p_hat[0, 0, 0] = 0.0

    return np.real(np.fft.ifftn(p_hat))
