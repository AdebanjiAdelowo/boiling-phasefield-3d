"""Navier-Stokes solver: one-fluid approach with projection-correction method.

Governing equations (Roccon 2025, Eqs. 3-4):

  Mass conservation:
    ∇·(ρu) = ṁ‴ (1 − ρᵥ/ρₗ)

  Momentum:
    ∂(ρu)/∂t + ∇·(ρuu) = −∇p + ∇·[μ(∇u + ∇uᵀ)] + fσ

Density and viscosity are linear functions of φ (Eqs. 5-6):
  ρ(φ) = ρᵥ φ + ρₗ(1−φ)
  μ(φ) = μᵥ φ + μₗ(1−φ)

Surface tension force via CSF model (Eq. 7):
  fσ = 6σ κ φ(1−φ) ∇φ

Projection-correction time stepping (Eqs. 16-20):
  1.  w* = wⁿ + Δt Cⁿ                   (intermediate momentum, explicit Euler)
  2.  ∇²p = [∇·w* − ṁ‴(1−ρᵥ/ρₗ)] / Δt  (constant-coefficient Poisson)
  3.  wⁿ⁺¹ = w* − Δt ∇p                (correction)
  4.  uⁿ⁺¹ = wⁿ⁺¹ / ρⁿ⁺¹               (recover velocity)
"""
import numpy as np
from .operators import (grad_x_2d, grad_y_2d, grad_x_3d, grad_y_3d, grad_z_3d,
                        laplacian_2d, laplacian_3d, div_2d, div_3d)
from .pressure import solve_poisson_2d, solve_poisson_3d


# ── Mixture properties ────────────────────────────────────────────────────────

def rho(phi, p):
    """ρ(φ) = ρᵥ φ + ρₗ(1−φ)  (Eq. 5)."""
    return p.rho_v * phi + p.rho_l * (1 - phi)


def mu(phi, p):
    """μ(φ) = μᵥ φ + μₗ(1−φ)  (Eq. 6)."""
    return p.mu_v * phi + p.mu_l * (1 - phi)


# ── Surface tension (CSF) ─────────────────────────────────────────────────────

def _curvature_2d(phi, dx, dy):
    """κ = ∇·(∇φ/|∇φ|)."""
    dphix = grad_x_2d(phi, dx)
    dphiy = grad_y_2d(phi, dy)
    mag   = np.sqrt(dphix**2 + dphiy**2 + 1e-14)
    return grad_x_2d(dphix / mag, dx) + grad_y_2d(dphiy / mag, dy)


def _curvature_3d(phi, dx, dy, dz):
    dphix = grad_x_3d(phi, dx)
    dphiy = grad_y_3d(phi, dy)
    dphiz = grad_z_3d(phi, dz)
    mag   = np.sqrt(dphix**2 + dphiy**2 + dphiz**2 + 1e-14)
    return (grad_x_3d(dphix / mag, dx)
            + grad_y_3d(dphiy / mag, dy)
            + grad_z_3d(dphiz / mag, dz))


def surface_tension_2d(phi, p):
    """fσ = 6σ κ φ(1−φ) ∇φ  (Eq. 7) — returns (fx, fy)."""
    kappa = _curvature_2d(phi, p.dx, p.dy)
    c = 6 * p.sigma * kappa * phi * (1 - phi)
    return c * grad_x_2d(phi, p.dx), c * grad_y_2d(phi, p.dy)


def surface_tension_3d(phi, p):
    """Returns (fx, fy, fz)."""
    kappa = _curvature_3d(phi, p.dx, p.dy, p.dz)
    c = 6 * p.sigma * kappa * phi * (1 - phi)
    return (c * grad_x_3d(phi, p.dx),
            c * grad_y_3d(phi, p.dy),
            c * grad_z_3d(phi, p.dz))


# ── 2-D Navier-Stokes step ────────────────────────────────────────────────────

def ns_step_2d(phi_new, phi_old, ux, uy, mdot_vol, p):
    """
    One explicit time step of the 2-D Navier-Stokes equations.

    Parameters
    ----------
    phi_new  : (Ny, Nx)  phase-field at time n+1 (already advanced)
    phi_old  : (Ny, Nx)  phase-field at time n
    ux, uy   : (Ny, Nx)  velocity at time n
    mdot_vol : (Ny, Nx)  volumetric vaporisation rate ṁ‴ at time n+1
    p        : SimParams

    Returns
    -------
    ux_new, uy_new : (Ny, Nx)  updated velocity
    pres           : (Ny, Nx)  pressure field
    """
    dx, dy, dt = p.dx, p.dy, p.dt

    rho_n  = rho(phi_old, p)
    mu_n   = mu(phi_old,  p)
    rho_np = rho(phi_new, p)    # ρⁿ⁺¹

    # Momentum variable  w = ρu
    wx = rho_n * ux
    wy = rho_n * uy

    # --- Advection: −∇·(ρuu) = −ρ(u·∇u) ---
    adv_x = -rho_n * (ux * grad_x_2d(ux, dx) + uy * grad_y_2d(ux, dy))
    adv_y = -rho_n * (ux * grad_x_2d(uy, dx) + uy * grad_y_2d(uy, dy))

    # --- Viscous stress: ∇·[μ(∇u + ∇uᵀ)] (simplified diagonal part) ---
    visc_x = (mu_n * laplacian_2d(ux, dx, dy)
              + grad_x_2d(mu_n, dx) * grad_x_2d(ux, dx)
              + grad_y_2d(mu_n, dy) * grad_y_2d(ux, dy))
    visc_y = (mu_n * laplacian_2d(uy, dx, dy)
              + grad_x_2d(mu_n, dx) * grad_x_2d(uy, dx)
              + grad_y_2d(mu_n, dy) * grad_y_2d(uy, dy))

    # --- Surface tension ---
    fsx, fsy = surface_tension_2d(phi_old, p)

    # --- Intermediate momentum (Eq. 16) ---
    Cx = adv_x + visc_x + fsx
    Cy = adv_y + visc_y + fsy

    wx_star = wx + dt * Cx
    wy_star = wy + dt * Cy

    # --- Pressure Poisson (Eq. 18) ---
    mass_src   = mdot_vol * (1.0 - p.rho_v / p.rho_l)
    div_w_star = div_2d(wx_star, wy_star, dx, dy)
    rhs_p      = (div_w_star - mass_src) / dt

    pres = solve_poisson_2d(rhs_p, dx, dy)

    # --- Correction (Eqs. 19-20) ---
    wx_new = wx_star - dt * grad_x_2d(pres, dx)
    wy_new = wy_star - dt * grad_y_2d(pres, dy)

    ux_new = wx_new / rho_np
    uy_new = wy_new / rho_np

    return ux_new, uy_new, pres


# ── 3-D Navier-Stokes step ────────────────────────────────────────────────────

def ns_step_3d(phi_new, phi_old, ux, uy, uz, mdot_vol, p):
    """
    One explicit time step of the 3-D Navier-Stokes equations.

    Parameters
    ----------
    phi_new, phi_old : (Nz, Ny, Nx)
    ux, uy, uz       : (Nz, Ny, Nx)
    mdot_vol         : (Nz, Ny, Nx)

    Returns
    -------
    ux_new, uy_new, uz_new : updated velocity components
    pres                   : pressure field
    """
    dx, dy, dz, dt = p.dx, p.dy, p.dz, p.dt

    rho_n  = rho(phi_old, p)
    mu_n   = mu(phi_old,  p)
    rho_np = rho(phi_new, p)

    wx = rho_n * ux
    wy = rho_n * uy
    wz = rho_n * uz

    # Advection
    adv_x = -rho_n * (ux*grad_x_3d(ux,dx) + uy*grad_y_3d(ux,dy) + uz*grad_z_3d(ux,dz))
    adv_y = -rho_n * (ux*grad_x_3d(uy,dx) + uy*grad_y_3d(uy,dy) + uz*grad_z_3d(uy,dz))
    adv_z = -rho_n * (ux*grad_x_3d(uz,dx) + uy*grad_y_3d(uz,dy) + uz*grad_z_3d(uz,dz))

    # Viscous stress
    visc_x = mu_n * laplacian_3d(ux, dx, dy, dz)
    visc_y = mu_n * laplacian_3d(uy, dx, dy, dz)
    visc_z = mu_n * laplacian_3d(uz, dx, dy, dz)

    # Surface tension
    fsx, fsy, fsz = surface_tension_3d(phi_old, p)

    wx_star = wx + dt * (adv_x + visc_x + fsx)
    wy_star = wy + dt * (adv_y + visc_y + fsy)
    wz_star = wz + dt * (adv_z + visc_z + fsz)

    mass_src   = mdot_vol * (1.0 - p.rho_v / p.rho_l)
    div_w_star = div_3d(wx_star, wy_star, wz_star, dx, dy, dz)
    pres       = solve_poisson_3d((div_w_star - mass_src) / dt, dx, dy, dz)

    ux_new = (wx_star - dt * grad_x_3d(pres, dx)) / rho_np
    uy_new = (wy_star - dt * grad_y_3d(pres, dy)) / rho_np
    uz_new = (wz_star - dt * grad_z_3d(pres, dz)) / rho_np

    return ux_new, uy_new, uz_new, pres
