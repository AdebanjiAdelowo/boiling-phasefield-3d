"""Verifies the mass-conservation property claimed in src/phase_field.py:

    "The divergence form of the RHS ensures mass conservation of each phase."

Both the advective term -div(u*phi) and the Allen-Cahn flux term div(J) are
written as the discrete divergence of some periodic field. For 2nd-order
central differences with periodic boundary conditions (np.roll), the sum of
a discrete derivative over the full domain is an EXACT algebraic identity,
not an approximation:

    sum_i (f[i+1] - f[i-1]) = 0   for any periodic array f,

because every element of f appears exactly once with a +1 coefficient and
once with a -1 coefficient in the telescoping sum. This holds for ANY phi,
ANY velocity field (divergence-free or not), independent of grid resolution.

So with no phase-change source (mdot_vol = 0), sum(ac_rhs) * cell_volume
must equal zero to floating-point round-off, for an arbitrary phi and an
arbitrary (even non-physical, randomly chosen) velocity field. This is checked
directly against the RHS, and separately against a short explicit-Euler
integration, without hard-coding any particular expected mass value.
"""
import numpy as np

from src.params import SimParams
from src.phase_field import ac_rhs_2d, ac_rhs_3d


def test_ac_rhs_2d_conserves_mass_with_no_source():
    p = SimParams(Nx=40, Ny=28, Lx=0.013, Ly=0.009)
    rng = np.random.default_rng(2)

    # Keep phi away from the [0,1] clip boundary and away from the eps-scale
    # interface singularity (|grad phi| -> 0) that ac_rhs_2d regularises with
    # a 1e-14 floor; a smooth random field comfortably avoids both.
    phi = rng.uniform(0.2, 0.8, size=(p.Ny, p.Nx))
    ux = rng.uniform(-2.0, 2.0, size=(p.Ny, p.Nx))  # not required to be div-free
    uy = rng.uniform(-2.0, 2.0, size=(p.Ny, p.Nx))
    mdot_vol = np.zeros_like(phi)

    A = ac_rhs_2d(phi, ux, uy, mdot_vol, p)

    total_rate = np.sum(A) * p.dx * p.dy
    scale = np.sum(np.abs(A)) * p.dx * p.dy
    assert _is_approx_zero(total_rate, scale)


def test_ac_rhs_3d_conserves_mass_with_no_source():
    p = SimParams(Nx=14, Ny=10, Nz=8, Lx=0.011, Ly=0.008, Lz=0.006)
    rng = np.random.default_rng(3)

    phi = rng.uniform(0.2, 0.8, size=(p.Nz, p.Ny, p.Nx))
    ux = rng.uniform(-2.0, 2.0, size=(p.Nz, p.Ny, p.Nx))
    uy = rng.uniform(-2.0, 2.0, size=(p.Nz, p.Ny, p.Nx))
    uz = rng.uniform(-2.0, 2.0, size=(p.Nz, p.Ny, p.Nx))
    mdot_vol = np.zeros_like(phi)

    A = ac_rhs_3d(phi, ux, uy, uz, mdot_vol, p)

    total_rate = np.sum(A) * p.dx * p.dy * p.dz
    scale = np.sum(np.abs(A)) * p.dx * p.dy * p.dz
    assert _is_approx_zero(total_rate, scale)


def test_ac_2d_total_mass_conserved_under_euler_stepping_with_no_source():
    """Complementary integration-level check: stepping ac_rhs_2d forward with
    explicit Euler and no source should hold total phase mass essentially
    constant, provided phi stays clear of the [0,1] clipping applied in
    src/solver.py::run_2d (clipping is a separate, expected nonlinearity, not
    part of the conservation property being tested here)."""
    p = SimParams(Nx=32, Ny=32, Lx=0.01, Ly=0.01, dt=1e-8)
    rng = np.random.default_rng(4)

    x = np.linspace(0, p.Lx, p.Nx, endpoint=False)
    y = np.linspace(0, p.Ly, p.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - p.Lx / 2) ** 2 + (Y - p.Ly / 2) ** 2)
    phi = 0.5 * (1 - np.tanh((r - 0.003) / (2 * p.eps)))  # smooth bubble, no clipping

    ux = rng.uniform(-0.1, 0.1, size=phi.shape)
    uy = rng.uniform(-0.1, 0.1, size=phi.shape)
    mdot_vol = np.zeros_like(phi)

    mass0 = np.sum(phi) * p.dx * p.dy
    for _ in range(50):
        A = ac_rhs_2d(phi, ux, uy, mdot_vol, p)
        phi = phi + p.dt * A
        assert phi.min() > 0.0 and phi.max() < 1.0  # confirms clipping never engages

    mass1 = np.sum(phi) * p.dx * p.dy
    assert abs(mass1 - mass0) < 1e-9 * max(abs(mass0), 1e-30)


def _is_approx_zero(value, scale, rel=1e-9, floor=1e-30):
    """True if `value` is zero relative to `scale` (or to `floor` if scale is
    itself ~0), used instead of hard-coding an absolute tolerance so the test
    scales correctly across grid sizes and field magnitudes."""
    return abs(value) < rel * max(scale, floor)
