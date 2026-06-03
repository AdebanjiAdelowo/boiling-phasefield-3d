"""Second-order central-difference operators on uniform Cartesian grids.

All functions assume **periodic boundary conditions** (via np.roll).
Array layout:
  2-D:  (Ny, Nx)
  3-D:  (Nz, Ny, Nx)

The axis conventions match NumPy's default (last axis = x, second-to-last = y).
"""
import numpy as np


# ── 1-D helpers (operate on the specified axis) ───────────────────────────────

def _d1(f, h, axis):
    """First derivative along *axis* using 2nd-order central differences."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)


def _d2(f, h, axis):
    """Second derivative along *axis* using 2nd-order central differences."""
    return (np.roll(f, -1, axis=axis) - 2 * f + np.roll(f, 1, axis=axis)) / h**2


# ── 2-D operators ─────────────────────────────────────────────────────────────

def grad_x_2d(f, dx): return _d1(f, dx, axis=-1)
def grad_y_2d(f, dy): return _d1(f, dy, axis=-2)


def grad_2d(f, dx, dy):
    """∇f = (∂f/∂x, ∂f/∂y)."""
    return grad_x_2d(f, dx), grad_y_2d(f, dy)


def div_2d(ux, uy, dx, dy):
    """∇·u = ∂ux/∂x + ∂uy/∂y."""
    return grad_x_2d(ux, dx) + grad_y_2d(uy, dy)


def laplacian_2d(f, dx, dy):
    """∇²f = ∂²f/∂x² + ∂²f/∂y²."""
    return _d2(f, dx, axis=-1) + _d2(f, dy, axis=-2)


# ── 3-D operators ─────────────────────────────────────────────────────────────

def grad_x_3d(f, dx): return _d1(f, dx, axis=-1)
def grad_y_3d(f, dy): return _d1(f, dy, axis=-2)
def grad_z_3d(f, dz): return _d1(f, dz, axis=-3)


def grad_3d(f, dx, dy, dz):
    """∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)."""
    return grad_x_3d(f, dx), grad_y_3d(f, dy), grad_z_3d(f, dz)


def div_3d(ux, uy, uz, dx, dy, dz):
    """∇·u in 3-D."""
    return grad_x_3d(ux, dx) + grad_y_3d(uy, dy) + grad_z_3d(uz, dz)


def laplacian_3d(f, dx, dy, dz):
    """∇²f in 3-D."""
    return _d2(f, dx, axis=-1) + _d2(f, dy, axis=-2) + _d2(f, dz, axis=-3)
