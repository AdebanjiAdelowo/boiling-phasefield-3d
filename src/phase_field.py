"""Conservative Allen-Cahn (CAC) phase-field equation with phase-change source.

Governing equation (Roccon 2025, Eq. 1):

  ∂φ/∂t + ∇·(uφ) = ∇·[ γ ( ε∇φ − φ(1−φ) ∇φ/|∇φ| ) ] + ṁ‴/ρᵥ

where:
  φ ∈ [0,1]   phase-field variable  (φ=1 vapour, φ=0 liquid)
  ε           interface width [m]
  γ           Allen-Cahn mobility
  ṁ‴         volumetric vaporisation rate [kg/(m³·s)]
  ρᵥ          vapour density [kg/m³]

The divergence form of the RHS ensures mass conservation of each phase.

Volumetric–to–surface vaporisation rate conversion (Eq. 2):
  ṁ‴ = ṁ |∇φ| ≈ ṁ φ(1−φ)/ε
"""
import numpy as np
from .operators import grad_x_2d, grad_y_2d, grad_x_3d, grad_y_3d, grad_z_3d


# ── Vaporisation source ───────────────────────────────────────────────────────

def mdot_volumetric(phi, mdot_surf, eps):
    """
    Convert surface vaporisation rate ṁ [kg/(m²·s)] to volumetric ṁ‴ [kg/(m³·s)].

    Uses the phase-field approximation |∇φ| ≈ φ(1−φ)/ε  (Eq. 2).
    """
    return mdot_surf * phi * (1 - phi) / eps


# ── 2-D Allen-Cahn RHS ────────────────────────────────────────────────────────

def ac_rhs_2d(phi, ux, uy, mdot_vol, p):
    """
    Right-hand side of the 2-D conservative Allen-Cahn equation.

    Returns  A  such that  φⁿ⁺¹ = φⁿ + Δt · A  (explicit Euler, Eq. 14).

    Parameters
    ----------
    phi      : (Ny, Nx)  phase-field
    ux, uy   : (Ny, Nx)  velocity components
    mdot_vol : (Ny, Nx)  volumetric vaporisation rate ṁ‴
    p        : SimParams
    """
    dx, dy   = p.dx, p.dy
    eps, gam = p.eps, p.gamma

    dphix = grad_x_2d(phi, dx)
    dphiy = grad_y_2d(phi, dy)
    mag   = np.sqrt(dphix**2 + dphiy**2 + 1e-14)   # |∇φ|

    # Total Allen-Cahn flux  J = γ( ε∇φ − φ(1−φ) n̂ )
    coeff = phi * (1 - phi) / mag
    Jx = gam * (eps * dphix - coeff * dphix)
    Jy = gam * (eps * dphiy - coeff * dphiy)

    # ∇·J
    div_J = grad_x_2d(Jx, dx) + grad_y_2d(Jy, dy)

    # Conservative advection  −∇·(uφ)
    adv = -(grad_x_2d(ux * phi, dx) + grad_y_2d(uy * phi, dy))

    # Phase-change source
    source = mdot_vol / p.rho_v

    return adv + div_J + source


# ── 3-D Allen-Cahn RHS ────────────────────────────────────────────────────────

def ac_rhs_3d(phi, ux, uy, uz, mdot_vol, p):
    """
    Right-hand side of the 3-D conservative Allen-Cahn equation.

    Parameters
    ----------
    phi          : (Nz, Ny, Nx)
    ux, uy, uz   : (Nz, Ny, Nx)
    mdot_vol     : (Nz, Ny, Nx)
    p            : SimParams
    """
    dx, dy, dz = p.dx, p.dy, p.dz
    eps, gam   = p.eps, p.gamma

    dphix = grad_x_3d(phi, dx)
    dphiy = grad_y_3d(phi, dy)
    dphiz = grad_z_3d(phi, dz)
    mag   = np.sqrt(dphix**2 + dphiy**2 + dphiz**2 + 1e-14)

    coeff = phi * (1 - phi) / mag
    Jx = gam * (eps * dphix - coeff * dphix)
    Jy = gam * (eps * dphiy - coeff * dphiy)
    Jz = gam * (eps * dphiz - coeff * dphiz)

    div_J = (grad_x_3d(Jx, dx) + grad_y_3d(Jy, dy) + grad_z_3d(Jz, dz))
    adv   = -(grad_x_3d(ux * phi, dx)
              + grad_y_3d(uy * phi, dy)
              + grad_z_3d(uz * phi, dz))

    return adv + div_J + mdot_vol / p.rho_v
