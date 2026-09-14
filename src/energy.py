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

    ∇Tᵥ and ∇Tₗ are two *distinct one-sided* gradients — the temperature
    gradient evaluated on the vapour side of the interface and on the
    liquid side respectively. They are NOT the same quantity: the whole
    physical content of the Rankine-Hugoniot balance is that they differ
    (that difference, weighted by each phase's conductivity, is what
    drives phase change). A previous version of this function collapsed
    both to a single shared central-difference ∇T and factored out
    (kᵥ − kₗ); that is only valid when ∇Tᵥ = ∇Tₗ, which is exactly the
    condition that never holds at a genuine phase-change interface, and
    it made the function vanish identically whenever kᵥ = kₗ (e.g. the
    Stefan benchmark's own conductivities) regardless of the actual
    temperature field, plus gave the wrong sign for real fluids where
    kᵥ < kₗ (e.g. water: kᵥ≈0.024, kₗ≈0.677 W/m/K).

    This version evaluates ∇Tᵥ and ∇Tₗ separately using phase-masked
    one-sided (forward/backward) differences at each grid point: along
    each axis, the one-sided stencil leaning toward the more-vapour
    neighbour (higher φ) estimates ∇Tᵥ, and the stencil leaning toward
    the more-liquid neighbour estimates ∇Tₗ. This is a local, grid-point
    approximation appropriate for a smeared interface a few cells wide;
    it is exact in the one-sided limit (one phase held at T_sat, i.e.
    zero gradient there) used by the Stefan benchmark, and — unlike the
    previous formula — it does not vanish merely because kᵥ = kₗ, only
    when the two one-sided gradients genuinely coincide (no heat-flux
    discontinuity). The fully general *probe* method of Roccon (2025)
    Section 2.4 (extrapolating along n̂ by ~ε on each side rather than
    using grid-adjacent one-sided differences) remains future work.
    """
    dx, dy = p.dx, p.dy

    # Interface normal n̂ = ∇φ/|∇φ|  (central difference; unaffected by
    # the one-sided-gradient fix below, since it doesn't multiply k)
    dphix = grad_x_2d(phi, dx)
    dphiy = grad_y_2d(phi, dy)
    mag   = np.sqrt(dphix**2 + dphiy**2 + 1e-14)
    nx, ny = dphix / mag, dphiy / mag

    # One-sided (forward / backward) differences of T along each axis
    Txf = (np.roll(T, -1, axis=-1) - T) / dx
    Txb = (T - np.roll(T, 1, axis=-1)) / dx
    Tyf = (np.roll(T, -1, axis=-2) - T) / dy
    Tyb = (T - np.roll(T, 1, axis=-2)) / dy

    # Phase of each neighbour, to decide which one-sided stencil samples
    # the vapour side and which samples the liquid side
    phi_xf = np.roll(phi, -1, axis=-1)
    phi_xb = np.roll(phi, 1, axis=-1)
    phi_yf = np.roll(phi, -1, axis=-2)
    phi_yb = np.roll(phi, 1, axis=-2)

    dTx_v = np.where(phi_xf >= phi_xb, Txf, Txb)   # leans toward vapour neighbour
    dTx_l = np.where(phi_xf >= phi_xb, Txb, Txf)   # leans toward liquid neighbour
    dTy_v = np.where(phi_yf >= phi_yb, Tyf, Tyb)
    dTy_l = np.where(phi_yf >= phi_yb, Tyb, Tyf)

    grad_Tv_n = dTx_v * nx + dTy_v * ny    # (∇Tᵥ)·n̂
    grad_Tl_n = dTx_l * nx + dTy_l * ny    # (∇Tₗ)·n̂

    mdot_surf = (p.k_v * grad_Tv_n - p.k_l * grad_Tl_n) / p.h_lv
    mdot_vol  = mdot_surf * phi * (1 - phi) / p.eps
    return mdot_vol
