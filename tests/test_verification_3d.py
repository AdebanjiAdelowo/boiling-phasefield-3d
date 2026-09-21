"""Basic 3-D verification (no physical validation of 3-D boiling is claimed here).

1. FFT Poisson solver against a manufactured solution in an anisotropic 3-D box
   (convergence order), complementing the discrete-inverse test in
   test_poisson_manufactured.py.
2. Projection consistency: for a smooth source, the divergence of the projected
   momentum matches the prescribed source with the O(h^2) residual expected from the
   collocated compact-Laplacian / central-gradient mismatch.  For a *sharp* bubble
   source the residual is much larger and converges slowly (~first order) at the
   resolutions used here, in 2-D as in 3-D; 3-D is not worse than 2-D, which is
   asserted below.
3. End-to-end conservation: run_3d changes the total phase mass by exactly the
   integrated vaporisation source.
4. Spherical bubble growth at a prescribed vaporisation rate, compared with the volume of
   the analytic *diffuse* profile at R0 + (mdot/rho_v) t.  The naive comparison with the
   sharp-sphere radius is dominated by a diffuse-profile offset (+25% to +72% in radius at
   N = 48..24 for R0 = 1 mm) that has nothing to do with the solver, the 3-D analogue of the
   2-D offset documented in the technical documentation.  This checks the mass-source and
   transport path in 3-D; it shows no convergence trend at these resolutions (the error
   is ~1e-3 and flat), so no convergence claim is made.
"""
import contextlib
import io

import numpy as np
import pytest

from src.flow import ns_step_2d, ns_step_3d, rho
from src.operators import div_2d, div_3d, grad_x_3d, grad_y_3d, grad_z_3d
from src.params import SimParams
from src.phase_field import mdot_volumetric
from src.pressure import solve_poisson_3d
from src.solver import run_3d


def _order(errs):
    return [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def test_poisson_3d_manufactured_solution_converges_at_second_order():
    L = (0.31, 0.27, 0.43)                        # anisotropic box: catches axis/spacing mix-ups

    def err(N):
        dx, dy, dz = (l / N for l in L)
        x, y, z = np.arange(N) * dx, np.arange(N) * dy, np.arange(N) * dz
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        kx, ky, kz = (2 * np.pi / l for l in L)
        exact = np.sin(kx * X) * np.cos(2 * ky * Y) * np.sin(kz * Z) + 0.5 * np.cos(3 * kz * Z) * np.sin(ky * Y)
        rhs = (-(kx ** 2 + (2 * ky) ** 2 + kz ** 2) * np.sin(kx * X) * np.cos(2 * ky * Y) * np.sin(kz * Z)
               - 0.5 * ((3 * kz) ** 2 + ky ** 2) * np.cos(3 * kz * Z) * np.sin(ky * Y))
        p = solve_poisson_3d(rhs, dx, dy, dz)
        return np.max(np.abs((p - p.mean()) - (exact - exact.mean())))

    e = [err(N) for N in (8, 16, 32)]
    assert all(o > 1.7 for o in _order(e))        # observed 1.78, 2.04


def test_projection_of_smooth_source_is_consistent_at_second_order():
    """w = -grad(p) with the solver's p should have div(w) = -source up to the collocated
    O(h^2) mismatch between the compact Laplacian inverted by the FFT and the central
    gradient/divergence used in the correction."""
    def resid(N):
        L = 0.01
        dx = L / N
        x = np.arange(N) * dx
        Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
        k = 2 * np.pi / L
        src = np.sin(k * X) * np.cos(k * Y) * np.sin(2 * k * Z) + 0.3 * np.cos(2 * k * X) * np.sin(k * Z)
        p = solve_poisson_3d(src, dx, dx, dx)
        w = [-grad_x_3d(p, dx), -grad_y_3d(p, dx), -grad_z_3d(p, dx)]
        return np.max(np.abs(div_3d(*w, dx, dx, dx) + src)) / np.max(np.abs(src))

    e = [resid(N) for N in (8, 16, 32)]
    assert all(o > 1.6 for o in _order(e))        # observed 1.76, 1.94


def _bubble_projection_residual(ndim, N):
    kw = dict(Nx=N, Ny=N, Lx=0.01, Ly=0.01, dt=5e-6, mode="prescribed", mdot_surf=0.1,
              rho_l=1000.0, rho_v=1.0, sigma=0.0)
    x = np.arange(N) * (0.01 / N)
    if ndim == 2:
        p = SimParams(**kw)
        Y, X = np.meshgrid(x, x, indexing="ij")
        r = np.sqrt((X - .005) ** 2 + (Y - .005) ** 2)
    else:
        p = SimParams(ndim=3, Nz=N, Lz=0.01, **kw)
        Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
        r = np.sqrt((X - .005) ** 2 + (Y - .005) ** 2 + (Z - .005) ** 2)
    phi = 0.5 * (1 - np.tanh((r - 0.002) / (2 * p.eps)))
    md = mdot_volumetric(phi, p.mdot_surf, p.eps)
    z = np.zeros_like(phi)
    src = md * (1 - p.rho_v / p.rho_l)
    if ndim == 2:
        ux, uy, _ = ns_step_2d(phi, phi, z, z, md, p)
        div = div_2d(rho(phi, p) * ux, rho(phi, p) * uy, p.dx, p.dy)
    else:
        ux, uy, uz, _ = ns_step_3d(phi, phi, z, z, z, md, p)
        div = div_3d(rho(phi, p) * ux, rho(phi, p) * uy, rho(phi, p) * uz, p.dx, p.dy, p.dz)
    return np.max(np.abs(div - src)) / np.max(np.abs(src))


def test_projection_of_sharp_bubble_source_in_3d_is_no_worse_than_2d():
    """Observed max relative divergence residual (N = 16, 32): 2-D 0.51, 0.28; 3-D 0.33, 0.15.
    Large because the interface source is only a few cells wide and the periodic box
    discards the k = 0 (net expansion) mode; the 2-D solver, which reproduces the bubble
    growth benchmark, has the same feature.  Only parity is asserted, not smallness."""
    for N in (16, 32):
        assert _bubble_projection_residual(3, N) <= 1.05 * _bubble_projection_residual(2, N)
    assert _bubble_projection_residual(3, 32) < _bubble_projection_residual(3, 16)


def test_run_3d_changes_total_phase_mass_by_exactly_the_integrated_source():
    N = 16
    p = SimParams(ndim=3, Nx=N, Ny=N, Nz=N, Lx=0.01, Ly=0.01, Lz=0.01, dt=5e-6, t_end=20 * 5e-6,
                  save_every=1, rho_l=1000.0, rho_v=1.0, sigma=0.0, mode="prescribed", mdot_surf=0.1)
    x = np.arange(N) * p.dx
    Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt((X - .005) ** 2 + (Y - .005) ** 2 + (Z - .005) ** 2)
    phi0 = 0.5 * (1 - np.tanh((r - 0.0025) / (2 * p.eps)))
    fields = [phi0]
    with contextlib.redirect_stdout(io.StringIO()):
        run_3d(p, phi0, callback=lambda step, t, st: fields.append(st["phi"].copy()))
    dV = p.dx * p.dy * p.dz
    for phi_n, phi_np in zip(fields[:-1], fields[1:]):
        assert phi_np.min() > 0 and phi_np.max() < 1, "clipping engaged; identity would not be exact"
        source = p.dt * np.sum(mdot_volumetric(phi_n, p.mdot_surf, p.eps)) / p.rho_v * dV
        assert (np.sum(phi_np) - np.sum(phi_n)) * dV == pytest.approx(source, rel=1e-9)


@pytest.mark.parametrize("N", [24, 32])
def test_3d_sphere_growth_matches_analytic_diffuse_profile_volume(N):
    """Observed V_num / V_ref - 1 = -7.9e-4 (N = 24), -1.0e-3 (N = 32), at t = 4 ms."""
    p = SimParams(ndim=3, Nx=N, Ny=N, Nz=N, Lx=0.01, Ly=0.01, Lz=0.01, dt=5e-6, t_end=0.004,
                  save_every=10 ** 9, rho_l=1000.0, rho_v=1.0, mu_l=1e-3, mu_v=1e-3, sigma=0.0,
                  mode="prescribed", mdot_surf=0.1)
    x = np.arange(N) * p.dx
    Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt((X - .005) ** 2 + (Y - .005) ** 2 + (Z - .005) ** 2)
    R0 = 0.001
    profile = lambda R: 0.5 * (1 - np.tanh((r - R) / (2 * p.eps)))
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_3d(p, profile(R0))
    v_num = np.sum(res["phi"]) * p.dx ** 3
    v_ref = np.sum(profile(R0 + p.mdot_surf / p.rho_v * res["t"])) * p.dx ** 3
    assert abs(v_num / v_ref - 1.0) < 3e-3
    # the growth increment itself is also right, not just the (large) initial volume
    v0 = np.sum(profile(R0)) * p.dx ** 3
    assert abs((v_num - v0) / (v_ref - v0) - 1.0) < 6e-3
