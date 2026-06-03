"""Energy equation with latent-heat source term.

Governing equation (Roccon 2025, Eq. 9):

  ∂T/∂t + ∇·(uT) = ∇·(α∇T) + Sₜ

where the thermal diffusivity is a mixture property (Eq. 10):
  α(φ) = αᵥ φ + αₗ(1−φ),  αₚ = kₚ / (ρₚ Cₚ,ₚ)

and the latent-heat source (Eq. 11):
  Sₜ = −(h_lv / Cₚ) · (∂φ/∂t + ∇·(uφ))

Implementation note
-------------------
Following the paper, the energy equation is solved only in the
*superheated* phase (the one driving phase change).  The other phase
is held at T_sat.  The caller sets the saturation mask after each step.
"""
import numpy as np
from .operators import (grad_x_2d, grad_y_2d, grad_x_3d, grad_y_3d, grad_z_3d)


def alpha_mix(phi, p):
    """Mixture thermal diffusivity α(φ) = αᵥφ + αₗ(1−φ)."""
    alpha_v = p.k_v / (p.rho_v * p.Cp_v)
    alpha_l = p.k_l / (p.rho_l * p.Cp_l)
    return alpha_v * phi + alpha_l * (1 - phi)


def Cp_mix(phi, p):
    return p.Cp_v * phi + p.Cp_l * (1 - phi)


# ── 2-D energy RHS ────────────────────────────────────────────────────────────

def energy_rhs_2d(T, phi, ux, uy, ac_rhs, p):
    """
    RHS of the 2-D energy equation.

    Parameters
    ----------
    T       : (Ny, Nx)  temperature field
    phi     : (Ny, Nx)  phase-field at time n
    ux, uy  : (Ny, Nx)  velocity
    ac_rhs  : (Ny, Nx)  Allen-Cahn RHS  A = ∂φ/∂t + ∇·(uφ) (net, no source)
    p       : SimParams

    Returns
    -------
    B : (Ny, Nx)  such that  Tⁿ⁺¹ = Tⁿ + Δt · B  (explicit Euler, Eq. 15)
    """
    dx, dy = p.dx, p.dy
    alpha  = alpha_mix(phi, p)
    Cp     = Cp_mix(phi, p)

    # Advection: −u·∇T
    adv = -(ux * grad_x_2d(T, dx) + uy * grad_y_2d(T, dy))

    # Diffusion: ∇·(α∇T)
    dTx = grad_x_2d(T, dx)
    dTy = grad_y_2d(T, dy)
    diff = grad_x_2d(alpha * dTx, dx) + grad_y_2d(alpha * dTy, dy)

    # Latent-heat source: Sₜ = −(h_lv/Cₚ) · (∂φ/∂t + ∇·(uφ))
    # ac_rhs = ∂φ/∂t + ∇·(uφ) already computed by phase_field module
    St = -(p.h_lv / Cp) * ac_rhs

    return adv + diff + St


# ── 3-D energy RHS ────────────────────────────────────────────────────────────

def energy_rhs_3d(T, phi, ux, uy, uz, ac_rhs, p):
    """3-D version of energy_rhs_2d."""
    dx, dy, dz = p.dx, p.dy, p.dz
    alpha = alpha_mix(phi, p)
    Cp    = Cp_mix(phi, p)

    adv = -(ux * grad_x_3d(T, dx)
            + uy * grad_y_3d(T, dy)
            + uz * grad_z_3d(T, dz))

    diff = (grad_x_3d(alpha * grad_x_3d(T, dx), dx)
            + grad_y_3d(alpha * grad_y_3d(T, dy), dy)
            + grad_z_3d(alpha * grad_z_3d(T, dz), dz))

    St = -(p.h_lv / Cp) * ac_rhs

    return adv + diff + St


# ── Vaporisation rate from heat-flux balance ──────────────────────────────────

def mdot_from_heatflux_2d(T, phi, p):
    """
    Compute volumetric vaporisation rate ṁ‴ from the Rankine-Hugoniot
    heat-flux balance at the interface (Roccon 2025, Eq. 12):

      ṁ = (kᵥ ∇Tᵥ − kₗ ∇Tₗ) · n̂ / h_lv

    Smeared to volumetric rate via Eq. 2:  ṁ‴ = ṁ φ(1−φ)/ε

    This simplified version evaluates the heat-flux balance at each
    grid point (not via probe interpolation).  For the full probe
    method see Roccon (2025) Section 2.4 — implementing that is
    Year 1 work for the PhD project.
    """
    dx, dy = p.dx, p.dy

    dTx = grad_x_2d(T, dx)
    dTy = grad_y_2d(T, dy)

    # Interface normal n̂ = ∇φ/|∇φ|
    dphix = grad_x_2d(phi, dx)
    dphiy = grad_y_2d(phi, dy)
    mag   = np.sqrt(dphix**2 + dphiy**2 + 1e-14)
    nx, ny = dphix / mag, dphiy / mag

    grad_T_n = dTx * nx + dTy * ny        # (∇T)·n̂

    mdot_surf = (p.k_v - p.k_l) * grad_T_n / p.h_lv
    mdot_vol  = mdot_surf * phi * (1 - phi) / p.eps
    return mdot_vol
