"""Verification and validation tests for the 1-D Stefan solver (src/stefan1d.py).

Layout
------
1. Reference solution      : the similarity solution is checked against the Stefan
                             condition, its exact energy budget, and an independent
                             moving-boundary PDE solve (no shared code with the formula).
2. Discrete building blocks: identities that must hold to round-off (conservation,
                             energy bookkeeping), the defects of the operators this
                             solver replaced, and interface-location extraction.
3. Front dynamics          : the Allen-Cahn front driven at the Stefan speed
                             (pinning of the legacy library operators at gamma = 0.1;
                             exact-speed translation with the |grad phi| source).
4. Coupled benchmark       : convergence to the analytic solution, budgets, no
                             spurious interfaces.

Items 1-3 are verification of implementation pieces, not physical validation.  Only
item 4 compares the coupled *dedicated 1-D solver* with an analytic solution of the
physical problem it is meant to reproduce (matched-density, one-phase, 1-D Stefan
problem).  Nothing here tests the general run_2d(mode='heat_flux') pathway.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.diagnostics import count_interface_crossings_1d, interface_position_1d, vapour_thickness_1d
from src.operators import grad_x_2d
from src.params import SimParams
from src.stefan1d import (StefanOptions, StefanReference, central_gradient_1d,
                          interface_heat_flux_1d, phase_rhs_1d, run_stefan_1d,
                          temperature_rhs_1d)


def stefan_params(nx, dt=5e-4, lx=0.1):
    """Roccon (2025) Sec. 3.2 properties (St = 0.2, matched densities)."""
    return SimParams(
        Nx=nx, Ny=1, Lx=lx, Ly=lx / nx, dt=dt,
        rho_l=1.0, rho_v=1.0, mu_l=0.01, mu_v=0.01, sigma=0.0,
        k_l=0.005, k_v=0.005, Cp_l=200.0, Cp_v=200.0, h_lv=1e4,
        T_sat=0.0, T_wall=10.0, mode="heat_flux",
    )


# ── 1. Reference solution ─────────────────────────────────────────────────────

def test_reference_similarity_constant_matches_paper_value():
    ref = StefanReference(stefan_params(100))
    assert ref.St == pytest.approx(0.2)
    assert ref.alpha == pytest.approx(2.5e-5)
    # Roccon (2025) reports xi = 0.3064 for this set-up
    assert ref.xi == pytest.approx(0.3064, abs=5e-5)


@pytest.mark.parametrize("t", [24.7, 100.0, 250.0])
def test_reference_satisfies_stefan_condition_and_boundary_values(t):
    """rho h d(delta)/dt = -k dT/dx at the interface (vapour side), T(0) = T_wall,
    T(delta) = T_sat.  Checked by finite differences of the closed forms, i.e. not
    by re-using the algebra that defines xi."""
    ref = StefanReference(stefan_params(100))
    dt_ = 1e-6 * t
    ddelta_dt = (ref.delta(t + dt_) - ref.delta(t - dt_)) / (2 * dt_)
    e = 1e-9
    d = float(ref.delta(t))
    dTdx = (ref.temperature(np.array([d - e]), t)[0] - ref.temperature(np.array([d - 2 * e]), t)[0]) / e

    assert dTdx < 0.0                                         # T falls toward the interface
    assert ddelta_dt > 0.0                                    # vapour advances into the liquid
    assert ref.rho * ref.h * ddelta_dt == pytest.approx(-ref.k * dTdx, rel=1e-6)
    assert ref.rho * ref.h * ddelta_dt == pytest.approx(ref.interface_flux(t), rel=1e-6)
    assert ref.temperature(np.array([0.0]), t)[0] == pytest.approx(ref.T_wall)
    assert ref.temperature(np.array([d - 1e-12]), t)[0] == pytest.approx(ref.T_sat, abs=1e-6)


def test_reference_exact_energy_budget():
    """d/dt of sensible energy = wall flux - latent absorption, and the cumulative
    identity  E(t) - E(t0) + latent = integral of wall flux  holds exactly."""
    ref = StefanReference(stefan_params(100))
    t0 = 24.7
    for t in (30.0, 100.0, 250.0):
        h = 1e-4 * t
        dE = (ref.stored_energy(t + h) - ref.stored_energy(t - h)) / (2 * h)
        assert dE == pytest.approx(ref.wall_flux(t) - ref.interface_flux(t), rel=1e-6)
        lhs = (ref.stored_energy(t) - ref.stored_energy(t0)) + ref.latent_energy(t0, t)
        assert lhs == pytest.approx(ref.cumulative_wall_heat(t0, t), rel=1e-10)
    # sign/magnitude sanity: at St = 0.2 most of the wall heat is latent
    assert ref.interface_flux(100.0) < ref.wall_flux(100.0)


def test_reference_agrees_with_independent_moving_boundary_solve():
    """Front-fixing (Landau) transform y = x/delta(t) turns the moving-boundary problem
    into a fixed-domain PDE + an ODE for delta, solved here by a stiff integrator that
    shares no code with the similarity formula."""
    p = stefan_params(100)
    ref = StefanReference(p)
    n = 120
    y = np.linspace(0.0, 1.0, n + 1)
    dy = y[1] - y[0]
    t0, t1 = 24.7, 100.0
    d0 = float(ref.delta(t0))
    T_init = ref.temperature(y * d0, t0)
    T_init[0], T_init[-1] = p.T_wall, p.T_sat

    def rhs(t, s):
        T = np.concatenate(([p.T_wall], s[:-1], [p.T_sat]))
        d = s[-1]
        Ty1 = (3 * T[-1] - 4 * T[-2] + T[-3]) / (2 * dy)
        ddot = -(p.k_v / (p.rho_v * p.h_lv)) * Ty1 / d
        Tyy = (T[2:] - 2 * T[1:-1] + T[:-2]) / dy ** 2
        Ty = (T[2:] - T[:-2]) / (2 * dy)
        return np.concatenate((ref.alpha * Tyy / d ** 2 + y[1:-1] * (ddot / d) * Ty, [ddot]))

    sol = solve_ivp(rhs, (t0, t1), np.concatenate((T_init[1:-1], [d0])), method="BDF",
                    t_eval=[t1], rtol=1e-9, atol=1e-11)
    assert sol.y[-1, -1] == pytest.approx(float(ref.delta(t1)), rel=1e-4)


# ── 2. Discrete building blocks ───────────────────────────────────────────────

def test_phase_rhs_1d_conserves_mass_without_source():
    p = stefan_params(60)
    rng = np.random.default_rng(11)
    phi = rng.uniform(0.2, 0.8, size=p.Nx)
    rhs = phase_rhs_1d(phi, np.zeros_like(phi), gamma=0.37, p=p)
    assert abs(np.sum(rhs) * p.dx) < 1e-9 * np.sum(np.abs(rhs)) * p.dx


def test_compact_energy_stencil_damps_the_checkerboard_mode_the_nested_one_cannot_see():
    """The library's nested central difference d/dx(alpha d/dx T) has a null mode at
    2*dx (even and odd nodes decouple); the compact stencil used here damps it.
    This is the defect behind the sawtooth temperature field of the old Stefan example."""
    p = stefan_params(64)
    dx = p.dx
    alpha_v = p.k_v / (p.rho_v * p.Cp_v)
    alpha = np.full(p.Nx, alpha_v)
    checker = (-1.0) ** np.arange(p.Nx)

    nested = grad_x_2d(alpha[None, :] * grad_x_2d(checker[None, :], dx), dx)[0]
    compact = temperature_rhs_1d(checker, alpha, dx)

    assert np.max(np.abs(nested)) < 1e-9 * alpha_v / dx ** 2                        # null mode
    assert np.all(compact[1:-1] * checker[1:-1] < 0)                                 # damped
    assert compact[5] == pytest.approx(-4 * alpha_v / dx ** 2 * checker[5], rel=1e-12)


def test_energy_budget_identity_and_mass_identity_hold_to_roundoff():
    """Discrete bookkeeping: the stored sensible energy changes exactly by (wall heat
    in) - (heat removed by the saturation clamp); the phase mass changes exactly by the
    integrated source (+ reported clipping).  Round-off identities, independent of accuracy."""
    p = stefan_params(80, dt=1e-3)
    res = run_stefan_1d(p, t0=24.7, t_end=40.0, record_every=200)
    h = res["history"]
    dE = h["E_sens"] - h["E_sens"][0]
    scale = np.max(np.abs(h["E_wall"]))
    assert np.max(np.abs(h["E_wall"] - h["E_clamp"] - dE)) < 1e-10 * scale
    dM = h["mass"] - h["mass"][0]
    # clipped mass is mass that was removed by clipping, so  dM = source - clipped
    assert np.max(np.abs(dM - h["mass_source"] + h["clipped"])) < 1e-10 * np.max(np.abs(dM))


def test_probe_returns_exact_flux_for_linear_profile_regardless_of_front_position():
    """For a locally linear T the extrapolated-gradient probe is exact whatever the
    sub-cell front position, unlike a probe that assumes T_f = T_sat at the front."""
    p = stefan_params(200, lx=0.2)
    x = np.arange(p.Nx) * p.dx
    slope = -600.0
    for x_front in (0.0300, 0.03013, 0.03047, 0.03081):
        phi = 0.5 * (1 - np.tanh((x - x_front) / (2 * p.eps)))
        T = np.where(x < x_front, p.T_wall + slope * x, 0.0)
        q = interface_heat_flux_1d(phi, T, p, probe_cells=2.0)
        assert q == pytest.approx(p.k_v * abs(slope), rel=1e-9)


def test_interface_position_and_mass_thickness_for_tanh_profile():
    p = stefan_params(200, lx=0.2)
    x = np.arange(p.Nx) * p.dx
    for d in (0.0150, 0.01537, 0.0421):
        phi = 0.5 * (1 - np.tanh((x - d) / (2 * p.eps)))
        # Linear interpolation of a tanh profile is off by the cubic term of tanh: for
        # eps = 1.5 dx and a sub-cell offset of 0.37 dx that is ~2.4e-6 m (0.24% of dx),
        # worked out by hand from phi - 1/2 = -s/2 + s^3/6, s = (x - d)/(2 eps).
        assert interface_position_1d(phi, x) == pytest.approx(d, abs=5e-6)
        assert vapour_thickness_1d(phi, p.dx) == pytest.approx(d, abs=1e-6)    # odd symmetry of tanh
        assert count_interface_crossings_1d(phi) == 1


def test_central_gradient_is_zero_at_ends_and_second_order_inside():
    p = stefan_params(100)
    x = np.arange(p.Nx) * p.dx
    f = np.sin(200 * x)
    g = central_gradient_1d(f, p.dx)
    assert g[0] == 0.0 and g[-1] == 0.0
    assert np.max(np.abs(g[1:-1] - 200 * np.cos(200 * x[1:-1]))) < 200 * (200 * p.dx) ** 2 / 6 * 1.01


# ── 3. Front dynamics (Allen-Cahn front driven by a prescribed source) ────────

def _drive_front(phi_rhs, shape, gamma, v=3e-4, T=100.0, N=100, dt=2e-3, Lx=0.1):
    """Drive a phi = 0.5 front at nominal speed v with the given source shape and mobility.
    Returns (measured speed / v, 10-90 width / equilibrium width).  v = 3e-4 m/s is the
    Stefan-benchmark interface speed at t0 = 24.7 s."""
    p = SimParams(Nx=N, Ny=1, Lx=Lx, Ly=Lx / N, dt=dt, rho_v=1.0, rho_l=1.0)
    dx, eps = p.dx, p.eps
    x = np.arange(N) * dx
    phi = 0.5 * (1 - np.tanh((x - 0.02) / (2 * eps)))
    n = int(T / dt)
    pos = []
    for k in range(1, n + 1):
        phi = np.clip(phi + dt * phi_rhs(phi, shape, gamma, v, p), 0.0, 1.0)
        if k % int(round(10 / dt)) == 0:
            pos.append(interface_position_1d(phi, x))
    i, j = np.where(phi < 0.9)[0][0], np.where(phi < 0.1)[0][0]
    xs, ps = x[i - 2:j + 3], phi[i - 2:j + 3]
    width = (np.interp(0.1, ps[::-1], xs[::-1]) - np.interp(0.9, ps[::-1], xs[::-1])) / (4.39 * eps)
    t = 10.0 * np.arange(1, len(pos) + 1)
    speed = np.polyfit(t[len(t) // 2:], np.array(pos)[len(t) // 2:], 1)[0]
    return speed / v, width


def _rhs_1d_module(phi, shape, gamma, v, p):
    dx = p.dx
    shape_f = np.abs(central_gradient_1d(phi, dx)) if shape == "gradient" else phi * (1 - phi) / p.eps
    return phase_rhs_1d(phi, v * shape_f, gamma, p)


@pytest.mark.parametrize("gamma", [3e-4, 0.1])
def test_gradient_source_moves_front_at_exact_speed_for_any_mobility(gamma):
    """With m_surf*|dphi/dx| the source is a pure translation term: the production rate is
    exactly m_surf for any profile, so the front moves at the driving speed and keeps its
    width whatever gamma is (here from the interface-speed scale up to the old 0.1 m/s)."""
    speed, width = _drive_front(_rhs_1d_module, "gradient", gamma)
    assert speed == pytest.approx(1.0, abs=5e-3)
    assert width == pytest.approx(1.0, abs=0.02)


def test_legacy_nested_operator_with_equilibrium_source_pins_front_at_large_gamma():
    """Legacy formulation (library `ac_rhs_2d`: nested central differences, source
    m_surf*phi(1-phi)/eps).  At the old heat-flux-mode default gamma = 0.1 m/s a front
    driven at the Stefan speed does not move at all (propagation failure: the sharpening
    flux, ~300x the interface speed, holds the profile on the grid).  A gamma of the order
    of the interface speed lets it move.  Documents the defect; the 1-D solver does not
    use this operator/source pairing.

    The library operators are periodic, so the test uses a vapour slab in the middle of
    the box (a wall front would wrap onto the outlet) and follows its right-hand front."""
    from src.phase_field import ac_rhs_2d

    v, dt, T = 3e-4, 5e-4, 30.0

    def right_front_speed(gamma):
        p = SimParams(Nx=200, Ny=1, Lx=0.2, Ly=0.001, dt=dt, rho_v=1.0, rho_l=1.0, gamma=gamma)
        x = np.arange(p.Nx) * p.dx
        a, b = 0.06, 0.14
        phi = 0.5 * (np.tanh((x - a) / (2 * p.eps)) - np.tanh((x - b) / (2 * p.eps)))
        z = np.zeros((1, p.Nx))
        pos = []
        for k in range(1, int(T / dt) + 1):
            md = v * phi * (1 - phi) / p.eps
            phi = np.clip(phi + dt * ac_rhs_2d(phi[None, :], z, z, md[None, :], p)[0], 0.0, 1.0)
            if k % int(round(5 / dt)) == 0:
                idx = np.where(np.diff(phi >= 0.5))[0][-1]           # right-hand front
                f0, f1 = phi[idx] - 0.5, phi[idx + 1] - 0.5
                pos.append(x[idx] + f0 / (f0 - f1) * p.dx)
        t = 5.0 * np.arange(1, len(pos) + 1)
        return np.polyfit(t[len(t) // 2:], np.array(pos)[len(t) // 2:], 1)[0] / v

    assert abs(right_front_speed(0.1)) < 0.05
    assert 0.9 < right_front_speed(3e-4) < 1.1


# ── 4. Coupled benchmark against the analytic solution ────────────────────────

ALPHA_V = 2.5e-5


def _run_benchmark(nx, t_end=250.0, options=None, t_wall=10.0):
    dx = 0.1 / nx
    dt = 0.2 * dx ** 2 / ALPHA_V                    # alpha dt/dx^2 = 0.2 (diffusive stability, not accuracy, sets dt)
    p = stefan_params(nx, dt=dt)
    p.T_wall = t_wall
    res = run_stefan_1d(p, t0=24.7, t_end=t_end, options=options, record_every=max(1, int(round(1.0 / dt))))
    h = res["history"]
    da = res["ref"].delta(h["t"])
    res["err_half"] = np.abs(h["delta_half"] - da) / da
    res["err_mass"] = np.abs(h["delta_mass"] - da) / da
    return res


@pytest.fixture(scope="module")
def benchmark_runs():
    return {nx: _run_benchmark(nx) for nx in (50, 100, 200)}


def test_stefan_1d_converges_to_analytic_solution(benchmark_runs):
    """Spatial refinement with eps = 1.5 dx (dt scaled for diffusive stability).  Observed:
    final error 1.14%, 0.62%, 0.32% for dx = 2, 1, 0.5 mm, i.e. first order in dx; a
    refinement to dx = 0.125 mm (run offline) continues the trend down to 0.08%.  N >= 50
    is used because N = 25 is pre-asymptotic (interface layer under 4 cells from the wall)."""
    e = {nx: r["err_half"][-1] for nx, r in benchmark_runs.items()}
    assert e[50] < 0.015 and e[100] < 0.008 and e[200] < 0.005
    assert e[50] / e[100] > 1.6
    assert e[100] / e[200] > 1.8
    # error is bounded over the whole run, not just at the end
    assert benchmark_runs[200]["err_half"].max() < 0.006
    assert benchmark_runs[200]["err_mass"].max() < 0.006


def test_stefan_1d_energy_budget_matches_exact_solution(benchmark_runs):
    """Conservation diagnostics against the exact sharp-interface budget (N = 200)."""
    res = benchmark_runs[200]
    h, ref = res["history"], res["ref"]
    t0, t1 = h["t"][0], h["t"][-1]

    wall_exact = ref.cumulative_wall_heat(t0, t1)
    assert h["E_wall"][-1] == pytest.approx(wall_exact, rel=0.01)
    stored_exact = ref.stored_energy(t1) - ref.stored_energy(t0)
    assert h["E_sens"][-1] - h["E_sens"][0] == pytest.approx(stored_exact, rel=0.02)

    latent_from_mass = ref.h * ref.rho * (h["mass"][-1] - h["mass"][0])
    assert h["E_clamp"][-1] == pytest.approx(latent_from_mass, rel=0.005)     # sink and source consistent
    assert latent_from_mass == pytest.approx(ref.latent_energy(t0, t1), rel=0.01)


def test_stefan_1d_solution_stays_clean(benchmark_runs):
    """No spurious interfaces, no reliance on clipping, no temperature oscillation."""
    for nx, res in benchmark_runs.items():
        h = res["history"]
        assert np.all(h["n_crossings"] == 1), nx
        assert abs(h["clipped"][-1]) < 1e-9, nx
        assert h["phi_min"].min() >= 0.0 and h["phi_max"].max() <= 1.0
        T = res["T"]
        assert T.min() >= res["params"].T_sat - 1e-9 and T.max() <= res["params"].T_wall + 1e-9
        vap = T[res["phi"] > 0.5]
        assert np.all(np.diff(vap) <= 1e-9)            # monotone from the wall to the front


@pytest.mark.parametrize("t_wall", [5.0, 20.0])
def test_stefan_1d_is_not_tuned_to_one_stefan_number(t_wall):
    """St = 0.1 and 0.4 (the benchmark is St = 0.2), same solver settings."""
    res = _run_benchmark(100, t_end=150.0, t_wall=t_wall)
    assert res["err_half"].max() < 0.02
    assert res["ref"].St == pytest.approx(0.02 * t_wall)


def test_equilibrium_source_shape_is_a_necessary_fix(benchmark_runs):
    """Replacing m_surf*|dphi/dx| by m_surf*phi(1-phi)/eps (all else equal) makes the
    layer width, and hence the production rate, drift.  At N = 200 the final error is
    ~5.5% against ~0.32%; in offline runs it reached 24% (N = 400) and 30% (N = 800),
    i.e. it worsens under refinement."""
    good = benchmark_runs[200]["err_half"][-1]
    bad = _run_benchmark(200, options=StefanOptions(source_shape="equilibrium"))["err_half"][-1]
    assert bad > 5 * good
