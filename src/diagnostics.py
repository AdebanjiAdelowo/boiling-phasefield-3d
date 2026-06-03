"""Post-processing utilities: interface tracking, Nusselt number, bubble radius."""
import numpy as np


def bubble_radius_2d(phi, dx, dy):
    """Estimate bubble radius from the vapour-phase area  A = π R²."""
    area = np.sum(phi) * dx * dy
    return np.sqrt(area / np.pi)


def bubble_radius_3d(phi, dx, dy, dz):
    """Estimate bubble radius from the vapour-phase volume  V = (4/3)π R³."""
    vol = np.sum(phi) * dx * dy * dz
    return (3 * vol / (4 * np.pi)) ** (1 / 3)


def interface_position_1d(phi_1d, x):
    """
    Locate the interface (φ = 0.5 iso-contour) in a 1-D array.
    Returns the position δ via linear interpolation, or None if not found.
    """
    f = phi_1d - 0.5
    idx = np.where(np.diff(np.sign(f)))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    frac = -f[i] / (f[i + 1] - f[i])
    return x[i] + frac * (x[i + 1] - x[i])


def nusselt_number_2d(T, k, dT_ref, L, dy):
    """
    Wall Nusselt number from the bottom-wall temperature gradient.

    Nu = (q_wall · L) / (k · ΔT_ref)
    q_wall ≈ k · (T[1,:] − T[0,:]) / dy  (1st-order one-sided)
    """
    q_wall = k * np.mean((T[1, :] - T[0, :]) / dy)
    return q_wall * L / (k * dT_ref)


def lp_norm(f, p_order, dx, dy):
    """Discrete Lᵖ norm over a 2-D field."""
    return (np.sum(np.abs(f) ** p_order) * dx * dy) ** (1 / p_order)


def kinetic_energy_2d(ux, uy, rho, dx, dy):
    """Total kinetic energy  E = ½ ∫ ρ|u|² dV."""
    return 0.5 * np.sum(rho * (ux**2 + uy**2)) * dx * dy
