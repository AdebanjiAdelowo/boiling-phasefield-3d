"""Independent verification of the FFT pressure-Poisson solver
(src/pressure.py) against manufactured solutions.

Two separate properties are checked, using two independently written
pieces of code so a bug in either the solver or the reference operator is
caught rather than a single implementation being checked against itself:

1. Discrete self-consistency: solve_poisson_* must exactly invert the
   discrete Laplacian implemented independently in src/operators.py, up to
   floating-point round-off. This isolates a solver bug (wrong eigenvalues,
   axis mix-up, wrong sign) from spatial discretisation error.

2. Grid convergence against the continuous manufactured solution: for a
   smooth periodic p_exact with an analytically known continuous Laplacian,
   the error between the numerical and exact solution should shrink at
   close to the 2nd order predicted by the central-difference scheme as the
   grid is refined. This is the classical method-of-manufactured-solutions
   check for a 2nd-order elliptic solver (see e.g. Roache, "Verification and
   Validation in Computational Science and Engineering", 1998).

No expected numerical value is hard-coded; only qualitative/quantitative
convergence trends are asserted.
"""
import numpy as np

from src.operators import laplacian_2d, laplacian_3d
from src.pressure import solve_poisson_2d, solve_poisson_3d


# ── 1. Discrete self-consistency (solver inverts the operator it is meant to) ─

def test_solve_poisson_2d_inverts_discrete_laplacian():
    rng = np.random.default_rng(0)
    N = 32
    Lx = Ly = 0.37  # arbitrary, non-unit domain size
    dx, dy = Lx / N, Ly / N

    # A smooth-ish periodic field built from a few Fourier modes (band-limited,
    # so it is exactly representable on this grid with no aliasing).
    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    coeffs = rng.uniform(-1, 1, size=6)
    p_exact = (
        coeffs[0] * np.sin(2 * np.pi * 1 * X / Lx)
        + coeffs[1] * np.cos(2 * np.pi * 2 * Y / Ly)
        + coeffs[2] * np.sin(2 * np.pi * 3 * X / Lx) * np.cos(2 * np.pi * 1 * Y / Ly)
        + coeffs[3] * np.cos(2 * np.pi * 2 * X / Lx) * np.sin(2 * np.pi * 4 * Y / Ly)
    )
    p_exact -= p_exact.mean()  # solver returns a zero-mean solution

    rhs = laplacian_2d(p_exact, dx, dy)  # forcing from the code's own operator
    p_num = solve_poisson_2d(rhs, dx, dy)

    assert np.max(np.abs(p_num - p_exact)) < 1e-8 * np.max(np.abs(p_exact))


def test_solve_poisson_3d_inverts_discrete_laplacian():
    rng = np.random.default_rng(1)
    N = 12
    Lx = Ly = Lz = 0.22
    dx = dy = dz = Lx / N

    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    z = np.linspace(0, Lz, N, endpoint=False)
    # indexing='ij' with inputs ordered (z, y, x) gives arrays shaped
    # (Nz, Ny, Nx), matching the (Nz, Ny, Nx) layout used throughout the solver.
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    coeffs = rng.uniform(-1, 1, size=3)
    p_exact = (
        coeffs[0] * np.sin(2 * np.pi * 1 * X / Lx) * np.cos(2 * np.pi * 1 * Y / Ly)
        + coeffs[1] * np.cos(2 * np.pi * 2 * Z / Lz)
        + coeffs[2] * np.sin(2 * np.pi * 1 * Y / Ly) * np.sin(2 * np.pi * 2 * Z / Lz)
    )
    p_exact -= p_exact.mean()

    rhs = laplacian_3d(p_exact, dx, dy, dz)
    p_num = solve_poisson_3d(rhs, dx, dy, dz)

    assert np.max(np.abs(p_num - p_exact)) < 1e-8 * np.max(np.abs(p_exact))


# ── 2. Grid convergence against the continuous manufactured solution ─────────

def _poisson_2d_manufactured_error(N):
    Lx = Ly = 1.0
    dx, dy = Lx / N, Ly / N
    x = np.linspace(0, Lx, N, endpoint=False)
    y = np.linspace(0, Ly, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx, ky = 2, 3  # well below Nyquist even at the coarsest N tested below
    p_exact = np.sin(2 * np.pi * kx * X / Lx) * np.sin(2 * np.pi * ky * Y / Ly)
    rhs = -((2 * np.pi * kx / Lx) ** 2 + (2 * np.pi * ky / Ly) ** 2) * p_exact

    p_num = solve_poisson_2d(rhs, dx, dy)
    return np.max(np.abs(p_num - p_exact))


def test_poisson_2d_converges_at_second_order():
    e16 = _poisson_2d_manufactured_error(16)
    e32 = _poisson_2d_manufactured_error(32)
    e64 = _poisson_2d_manufactured_error(64)

    assert e32 < e16
    assert e64 < e32

    # Clean 2nd-order central differencing predicts ~4x error reduction per
    # doubling of N; a conservative threshold keeps this robust to the
    # specific manufactured solution chosen while still catching a solver
    # that has regressed to 1st order (ratio -> ~2) or worse.
    assert e16 / e32 > 3.0
    assert e32 / e64 > 3.0
