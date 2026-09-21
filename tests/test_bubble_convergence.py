"""Trend test for the 2-D bubble-growth benchmark, at reduced resolution and run time
for test-suite speed: as the grid is refined the error between the numerical and
analytical bubble radius should decrease.

The technical documentation originally reported approximately second-order
convergence for this benchmark.  An independent re-measurement (docs Section 11.1,
"Re-evaluation") reproduces the systematic decrease of the error but NOT a
justified order, so this test asserts only what is supported: the error decreases
and does so by a factor above 1.5 between N = 32 and N = 64 (thresholds unchanged).

This calls the real src.solver.run_2d pipeline end to end (not a
reimplementation of the physics), at Nx=Ny in {32, 64} and a shorter t_end,
purely to keep this test fast.  No expected error value is hard-coded here; the
golden-value test at the bottom of this file separately protects the validated
implementation from unintended change.
"""
import numpy as np
import pytest

from src.diagnostics import bubble_radius_2d
from src.params import SimParams
from src.solver import run_2d


def _bubble_growth_relative_error(N, t_end):
    p = SimParams(
        Nx=N, Ny=N, Lx=0.01, Ly=0.01,
        dt=5e-6, t_end=t_end, save_every=10 ** 9,
        rho_l=1000.0, rho_v=1.0, mu_l=1e-3, mu_v=1e-3, sigma=0.0,
        mode="prescribed", mdot_surf=0.1,
    )
    x = np.linspace(0, p.Lx, p.Nx, endpoint=False)
    y = np.linspace(0, p.Ly, p.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    R0 = 0.001
    r = np.sqrt((X - p.Lx / 2) ** 2 + (Y - p.Ly / 2) ** 2)
    phi0 = 0.5 * (1 - np.tanh((r - R0) / (2 * p.eps)))

    result = run_2d(p, phi0)

    R_num = bubble_radius_2d(result["phi"], p.dx, p.dy)
    R_ana = R0 + (p.mdot_surf / p.rho_v) * result["t"]
    return abs(R_num - R_ana) / R_ana


def test_bubble_growth_error_decreases_under_grid_refinement():
    t_end = 0.001  # short run: enough growth to measure, fast enough for a test
    err_32 = _bubble_growth_relative_error(32, t_end)
    err_64 = _bubble_growth_relative_error(64, t_end)

    assert err_64 < err_32

    # The technical documentation originally reported ratios of ~2.9-3.5 between
    # successive doublings of N at t_end = 0.008 s; an independent re-measurement gave
    # 3.7 (N = 32 -> 64) for the final-radius error at that t_end but did not
    # reproduce the documented N = 128 row, so no order is asserted.  The bound below
    # is unchanged: this shorter run (same fixed dt) mixes spatial and temporal error
    # differently, and 1.5 still fails if the decrease degrades badly or disappears.
    assert err_32 / err_64 > 1.5


# Golden values recorded from commit c9a2842 (before the Stefan-benchmark work), full 8 ms run,
# dt = 5e-6 s, N = 32 and 64.  The prescribed-vaporisation 2-D path must stay bit-identical to
# what was validated; any change to the shared operators, Allen-Cahn, projection or SimParams
# defaults that alters it should be a conscious decision, not a side effect.
GOLDEN_R_NUM = {32: 0.001986348189872809, 64: 0.0018500297037241154}


def test_bubble_growth_2d_is_unchanged_from_the_validated_baseline():
    for N, R_golden in GOLDEN_R_NUM.items():
        p = SimParams(
            Nx=N, Ny=N, Lx=0.01, Ly=0.01, dt=5e-6, t_end=0.008, save_every=10 ** 9,
            rho_l=1000.0, rho_v=1.0, mu_l=1e-3, mu_v=1e-3, sigma=0.0,
            mode="prescribed", mdot_surf=0.1,
        )
        x = np.linspace(0, p.Lx, p.Nx, endpoint=False)
        y = np.linspace(0, p.Ly, p.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt((X - p.Lx / 2) ** 2 + (Y - p.Ly / 2) ** 2)
        phi0 = 0.5 * (1 - np.tanh((r - 0.001) / (2 * p.eps)))
        result = run_2d(p, phi0)
        R_num = bubble_radius_2d(result["phi"], p.dx, p.dy)
        assert R_num == pytest.approx(R_golden, rel=1e-9), N
