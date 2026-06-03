"""Main time-integration loop — couples all physics modules.

Time-stepping follows Roccon (2025) Section 2.5:
  1. Advance Allen-Cahn  φ (explicit Euler)
  2. Advance energy      T (explicit Euler)
  3. Advance Navier-Stokes  u via projection-correction
"""
import numpy as np
from .params import SimParams
from .phase_field import ac_rhs_2d, ac_rhs_3d, mdot_volumetric
from .flow import ns_step_2d, ns_step_3d
from .energy import energy_rhs_2d, energy_rhs_3d, mdot_from_heatflux_2d


def run_2d(p: SimParams, phi0, ux0=None, uy0=None, T0=None,
           callback=None, superheated_phase='vapour'):
    """
    2-D boiling simulation.

    Parameters
    ----------
    p                 : SimParams
    phi0              : (Ny, Nx)  initial phase-field
    ux0, uy0          : (Ny, Nx)  initial velocity  [default: zero]
    T0                : (Ny, Nx)  initial temperature  [default: T_sat everywhere]
    callback          : callable(step, t, state) called every p.save_every steps
                        state = dict(phi, ux, uy, T, pres)
    superheated_phase : 'vapour' or 'liquid' — which phase is superheated
                        (controls which phase is held at T_sat)

    Returns
    -------
    dict with final fields {'phi', 'ux', 'uy', 'T', 'pres', 't'}
    """
    phi  = phi0.copy()
    ux   = np.zeros_like(phi) if ux0 is None else ux0.copy()
    uy   = np.zeros_like(phi) if uy0 is None else uy0.copy()
    T    = np.full_like(phi, p.T_sat, dtype=float) if T0 is None else T0.copy()
    pres = np.zeros_like(phi)

    n_steps = int(round(p.t_end / p.dt))
    t = 0.0

    print(f"2-D boiling simulation  |  grid {p.Nx}×{p.Ny}  |  "
          f"dt={p.dt:.2e} s  |  {n_steps} steps  |  mode='{p.mode}'")

    for step in range(n_steps):

        # ── 1. Vaporisation rate ──────────────────────────────────────────────
        if p.mode == 'prescribed':
            mdot_vol = mdot_volumetric(phi, p.mdot_surf, p.eps)
        else:
            mdot_vol = mdot_from_heatflux_2d(T, phi, p)

        # ── 2. Allen-Cahn (Eq. 14) ────────────────────────────────────────────
        A    = ac_rhs_2d(phi, ux, uy, mdot_vol, p)
        phi_new = np.clip(phi + p.dt * A, 0.0, 1.0)

        # ── 3. Energy equation (Eq. 15) ───────────────────────────────────────
        if p.mode == 'heat_flux':
            # ac_rhs for energy source excludes the phase-change source term
            A_no_src = ac_rhs_2d(phi, ux, uy, np.zeros_like(mdot_vol), p)
            B = energy_rhs_2d(T, phi, ux, uy, A_no_src, p)
            T_new = T + p.dt * B
            # Enforce T_sat in the non-superheated phase
            if superheated_phase == 'vapour':
                T_new = np.where(phi_new < 0.5, p.T_sat, T_new)
            else:
                T_new = np.where(phi_new > 0.5, p.T_sat, T_new)
        else:
            T_new = T

        # ── 4. Navier-Stokes (Eqs. 16-20) ────────────────────────────────────
        ux_new, uy_new, pres = ns_step_2d(phi_new, phi, ux, uy, mdot_vol, p)

        # Clamp velocity to prevent blow-up in explicit scheme
        u_max = np.sqrt(np.max(ux_new**2 + uy_new**2) + 1e-30)
        u_lim = p.Lx / p.dt          # ~ 1 grid cell per step
        if u_max > u_lim:
            ux_new *= u_lim / u_max
            uy_new *= u_lim / u_max

        phi, ux, uy, T = phi_new, ux_new, uy_new, T_new
        t += p.dt

        if callback is not None and (step % p.save_every == 0):
            callback(step, t, dict(phi=phi, ux=ux, uy=uy, T=T, pres=pres))

    print(f"Done.  t_final = {t:.6f} s")
    return dict(phi=phi, ux=ux, uy=uy, T=T, pres=pres, t=t)


def run_3d(p: SimParams, phi0, ux0=None, uy0=None, uz0=None, T0=None,
           callback=None, superheated_phase='vapour'):
    """
    3-D boiling simulation.  Mirrors run_2d with an extra z-velocity component.

    Parameters
    ----------
    phi0            : (Nz, Ny, Nx)
    ux0, uy0, uz0   : (Nz, Ny, Nx)  [default: zero]
    T0              : (Nz, Ny, Nx)  [default: T_sat]
    """
    phi  = phi0.copy()
    ux   = np.zeros_like(phi) if ux0 is None else ux0.copy()
    uy   = np.zeros_like(phi) if uy0 is None else uy0.copy()
    uz   = np.zeros_like(phi) if uz0 is None else uz0.copy()
    T    = np.full_like(phi, p.T_sat, dtype=float) if T0 is None else T0.copy()
    pres = np.zeros_like(phi)

    n_steps = int(round(p.t_end / p.dt))
    t = 0.0

    print(f"3-D boiling simulation  |  grid {p.Nx}×{p.Ny}×{p.Nz}  |  "
          f"dt={p.dt:.2e} s  |  {n_steps} steps")

    for step in range(n_steps):

        if p.mode == 'prescribed':
            mdot_vol = mdot_volumetric(phi, p.mdot_surf, p.eps)
        else:
            raise NotImplementedError("heat_flux mode for 3-D: implement probe method")

        A       = ac_rhs_3d(phi, ux, uy, uz, mdot_vol, p)
        phi_new = np.clip(phi + p.dt * A, 0.0, 1.0)

        if p.mode == 'heat_flux':
            A_ns = ac_rhs_3d(phi, ux, uy, uz, np.zeros_like(mdot_vol), p)
            B    = energy_rhs_3d(T, phi, ux, uy, uz, A_ns, p)
            T_new = T + p.dt * B
            if superheated_phase == 'vapour':
                T_new = np.where(phi_new < 0.5, p.T_sat, T_new)
            else:
                T_new = np.where(phi_new > 0.5, p.T_sat, T_new)
        else:
            T_new = T

        ux_new, uy_new, uz_new, pres = ns_step_3d(
            phi_new, phi, ux, uy, uz, mdot_vol, p)

        phi, ux, uy, uz, T = phi_new, ux_new, uy_new, uz_new, T_new
        t += p.dt

        if callback is not None and (step % p.save_every == 0):
            callback(step, t, dict(phi=phi, ux=ux, uy=uy, uz=uz, T=T, pres=pres))

    print(f"Done.  t_final = {t:.6f} s")
    return dict(phi=phi, ux=ux, uy=uy, uz=uz, T=T, pres=pres, t=t)
