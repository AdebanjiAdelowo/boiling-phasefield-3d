"""Diagnostic/regression test documenting the known, unresolved failure of
the 1-D Stefan-problem benchmark described in
docs/Phase_Field_Boiling_Solver_Technical_Documentation.md ("The Stefan
problem"): the interface tracks the analytical solution only in an early
transient, then the run disintegrates (spurious secondary interfaces,
oscillatory temperature field) due to (1) a missing probe-based
vaporisation-rate method and (2) periodic boundary conditions that are
physically wrong for this problem.

Neither test below claims the benchmark is validated:

- test_stefan_1d_benchmark_diverges_as_documented is a plain regression
  check confirming the known failure is still present at roughly its
  documented magnitude, so a silent change in this behaviour (in either
  direction) would be caught.
- test_stefan_1d_benchmark_would_be_validated_if_fixed is marked xfail
  (strict) because it encodes the *target* behaviour the benchmark should
  reach once fixed, not the current behaviour. strict=True turns an
  accidental pass into a hard failure, forcing a human to consciously
  remove the marker rather than letting an undocumented fix (or a silent
  regression the other way) go unnoticed.

The physics below mirrors examples/stefan_1d.py exactly (same equations,
same parameters), reduced only in t_end (120 s of modelled physical time
instead of 250 s) to keep the test suite fast; the technical
documentation's own diagnostic table shows the failure is already
unambiguous well before t = 120 s (30.7% error and multiple spurious
interfaces by t = 100 s).
"""
import numpy as np
import pytest
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


def test_stefan_1d_benchmark_diverges_as_documented():
    delta_num, delta_ana = _run_stefan_1d(t_end=120.0)
    assert delta_num is not None
    rel_error = abs(delta_num - delta_ana) / delta_ana

    # The technical documentation reports 30.7% error at t=100s, rising
    # further by t=120-150s. A loose lower bound confirms the failure is
    # still present at roughly the documented magnitude without pinning an
    # exact figure, which would make this test brittle to unrelated
    # numerical changes.
    assert rel_error > 0.15


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known open issue, not yet fixed: the 1-D Stefan benchmark requires "
        "a probe-based vaporisation-rate method and non-periodic (wall/"
        "outlet) boundary conditions, neither implemented yet. See "
        "docs/Phase_Field_Boiling_Solver_Technical_Documentation.md, 'The "
        "Stefan problem', for the full mechanism. This test encodes the "
        "target behaviour once that work is done; it must stay marked "
        "xfail until then. strict=True turns an accidental pass into a "
        "hard failure so the marker cannot be silently forgotten."
    ),
)
def test_stefan_1d_benchmark_would_be_validated_if_fixed():
    delta_num, delta_ana = _run_stefan_1d(t_end=120.0)
    rel_error = abs(delta_num - delta_ana) / delta_ana
    assert rel_error < 0.05
