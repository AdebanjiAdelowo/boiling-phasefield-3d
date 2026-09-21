"""Regression test for the LEGACY 1-D Stefan scheme (superseded by src/stefan1d.py).

`examples/stefan_1d.py` used to carry its own inline scheme (copied below, unchanged):
nested central differences d/dx(alpha d/dx T) with alpha = alpha_v*phi, periodic
np.roll operators with the wall pinned on node 0, a grid-point central-difference
vaporisation gradient, source m_surf*phi(1-phi)/eps, T clamped to T_sat where phi<0.5.
That scheme fails the benchmark: about 51% interface error at t = 120 s and 66% at
t = 250 s with the current SimParams gamma default (0.1 m/s); 56% at t = 250 s with the
pre-6c114f9 gamma (= eps), which is the historical figure quoted in earlier documentation.  The failure was diagnosed and fixed by replacing the scheme with the
solver in src/stefan1d.py; see tests/test_stefan_1d.py for the verification of the new
solver and the "Stefan problem" section of
docs/Phase_Field_Boiling_Solver_Technical_Documentation.md for the diagnosis.

This test is kept as evidence that the legacy formulation is still defective, so that
the reasons for not using it stay checkable.  It is NOT a benchmark of the current
solver, and it deliberately no longer contains an xfail "target" test: the target
behaviour is now tested, against the analytic solution, in tests/test_stefan_1d.py.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from src.diagnostics import interface_position_1d
from src.operators import grad_x_2d
from src.params import SimParams
from src.phase_field import ac_rhs_2d


def _run_stefan_1d(t_end):
    p = SimParams(
        Nx=200, Ny=1, Lx=0.2, Ly=0.2 / 200, dt=5e-4, t_end=t_end,
        rho_l=1.0, rho_v=1.0, mu_l=0.01, mu_v=0.01, sigma=0.0,
        k_l=0.005, k_v=0.005, Cp_l=200.0, Cp_v=200.0, h_lv=1e4,
        T_sat=0.0, T_wall=10.0, mode="heat_flux",
    )
    alpha_v = p.k_v / (p.rho_v * p.Cp_v)
    St_num = p.Cp_v * (p.T_wall - p.T_sat) / p.h_lv
    xi = brentq(
        lambda xi: xi * np.exp(xi ** 2) * erf(xi) - St_num / np.sqrt(np.pi),
        1e-8, 10.0,
    )

    def delta_analytical(t):
        return 2 * xi * np.sqrt(alpha_v * t) if t > 0 else 0.0

    t0 = 24.7
    delta0 = delta_analytical(t0)
    x = np.linspace(0, p.Lx, p.Nx, endpoint=False)
    X2d = x[np.newaxis, :]

    phi = 0.5 * (1 - np.tanh((X2d - delta0) / (2 * p.eps)))
    T = np.where(
        X2d < delta0,
        p.T_wall - (p.T_wall - p.T_sat) * erf(X2d / (2 * np.sqrt(alpha_v * t0))) / erf(xi),
        p.T_sat,
    )
    ux = np.zeros_like(phi)
    uy = np.zeros_like(phi)
    dx = p.dx

    t = t0
    n_steps = int(round((t_end - t0) / p.dt))
    for _ in range(n_steps):
        dT_dx = grad_x_2d(T, dx)
        dphidx = grad_x_2d(phi, dx)
        mag = np.abs(dphidx) + 1e-14
        nx = dphidx / mag
        mdot_surf_local = p.k_v * dT_dx * nx / p.h_lv
        mdot_vol = mdot_surf_local * phi * (1 - phi) / p.eps

        A = ac_rhs_2d(phi, ux, uy, mdot_vol, p)
        phi_new = np.clip(phi + p.dt * A, 0.0, 1.0)

        alpha_f = p.k_v / (p.rho_v * p.Cp_v) * phi
        diff_T = grad_x_2d(alpha_f * grad_x_2d(T, dx), dx)
        T_new = T + p.dt * diff_T
        T_new = np.where(phi_new < 0.5, p.T_sat, T_new)
        T_new[0, 0] = p.T_wall

        phi, T = phi_new, T_new
        t += p.dt

    delta_num = interface_position_1d(phi[0], x)
    delta_ana = delta_analytical(t)
    return delta_num, delta_ana


def test_legacy_stefan_1d_scheme_still_fails_the_benchmark():
    delta_num, delta_ana = _run_stefan_1d(t_end=120.0)
    assert delta_num is not None
    rel_error = abs(delta_num - delta_ana) / delta_ana

    # Observed at t = 120 s: 51.5% (current default gamma = 0.1); 42% with gamma = 2e-3;
    # ~31% at t = 100 s with gamma = eps (historical).  A loose
    # lower bound confirms the legacy scheme is still defective without pinning an
    # exact figure that would be brittle to unrelated changes.
    assert rel_error > 0.15
