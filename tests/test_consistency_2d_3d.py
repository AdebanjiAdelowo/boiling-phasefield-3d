"""2-D / 3-D consistency: a 3-D run whose fields are invariant along one axis must
reproduce the 2-D run in the remaining plane.

This is a verification of the dimensional extension (operators, projection, viscous
and advective terms, axis conventions), NOT a physical validation of 3-D boiling.

The comparison is repeated for extrusion along each of the three axes, with different
in-plane sizes/spacings and a viscosity contrast (mu_v != mu_l), so that
  * an axis mix-up in any 3-D operator,
  * a spacing (dx/dy/dz) mix-up,
  * a term present in 2-D but missing in 3-D (the grad(mu).grad(u) viscous terms were
    missing in 3-D; with mu_v == mu_l they are ~1e-19 and the mismatch is invisible)
shows up as a mismatch of order 1e-3 rather than round-off.

Array layouts: 2-D (Ny, Nx); 3-D (Nz, Ny, Nx).  Mapping of the 2-D plane onto 3-D:
  extrude along z : 2-D (y, x) -> 3-D (y, x)     velocities (ux, uy) -> (ux, uy)
  extrude along y : 2-D (y, x) -> 3-D (z, x)     velocities (ux, uy) -> (ux, uz)
  extrude along x : 2-D (y, x) -> 3-D (z, y)     velocities (ux, uy) -> (uy, uz)
"""
import contextlib
import io

import numpy as np
import pytest

from src.params import SimParams
from src.solver import run_2d, run_3d

NX2, NY2 = 24, 20                    # in-plane grid (deliberately unequal)
LX2, LY2 = 0.011, 0.008              # in-plane box (deliberately unequal, dx != dy)
N_EXTRUDED = 6                       # points along the invariant axis (arbitrary)
L_EXTRUDED = 0.0093                  # box length along the invariant axis (arbitrary)


def _initial_fields():
    x = np.arange(NX2) * (LX2 / NX2)
    y = np.arange(NY2) * (LY2 / NY2)
    Y, X = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt((X - 0.45 * LX2) ** 2 + (Y - 0.55 * LY2) ** 2)
    eps = 1.5 * LX2 / NX2
    phi = 0.5 * (1 - np.tanh((r - 0.0011) / (2 * eps)))
    ux = 0.05 * np.sin(2 * np.pi * X / LX2) * np.cos(2 * np.pi * Y / LY2)
    uy = -0.03 * np.cos(2 * np.pi * X / LX2) * np.sin(4 * np.pi * Y / LY2)
    return phi, ux, uy


def _run_pair(axis, mu_v, steps=40):
    kw2 = dict(Nx=NX2, Ny=NY2, Lx=LX2, Ly=LY2, dt=5e-6, t_end=steps * 5e-6, save_every=10 ** 9,
               rho_l=1000.0, rho_v=1.0, mu_l=1e-3, mu_v=mu_v, sigma=0.0,
               mode="prescribed", mdot_surf=0.1)
    p2 = SimParams(**kw2)
    phi2, ux2, uy2 = _initial_fields()

    n, L = N_EXTRUDED, L_EXTRUDED
    if axis == "z":
        dims = dict(Nx=NX2, Ny=NY2, Nz=n, Lx=LX2, Ly=LY2, Lz=L)
        ex = lambda f: np.repeat(f[None, :, :], n, axis=0)
        zero = np.zeros((n, NY2, NX2))
        ics = dict(ux0=ex(ux2), uy0=ex(uy2), uz0=zero)
        pick = lambda a: a[0]
        mapping = {"ux": "ux", "uy": "uy"}
    elif axis == "y":
        dims = dict(Nx=NX2, Ny=n, Nz=NY2, Lx=LX2, Ly=L, Lz=LY2)
        ex = lambda f: np.repeat(f[:, None, :], n, axis=1)
        zero = np.zeros((NY2, n, NX2))
        ics = dict(ux0=ex(ux2), uy0=zero, uz0=ex(uy2))
        pick = lambda a: a[:, 0, :]
        mapping = {"ux": "ux", "uy": "uz"}
    else:  # "x"
        dims = dict(Nx=n, Ny=NX2, Nz=NY2, Lx=L, Ly=LX2, Lz=LY2)
        ex = lambda f: np.repeat(f[:, :, None], n, axis=2)
        zero = np.zeros((NY2, NX2, n))
        ics = dict(ux0=zero, uy0=ex(ux2), uz0=ex(uy2))
        pick = lambda a: a[:, :, 0]
        mapping = {"ux": "uy", "uy": "uz"}

    kw3 = {k: v for k, v in kw2.items() if k not in ("Nx", "Ny", "Lx", "Ly")}
    # SimParams derives eps (1.5 dx) and gamma (max(mdot/rho_v, dx/dt*1e-3)) from the *x*-spacing,
    # which for extrusion along x is the arbitrary extruded-direction spacing; pass the 2-D
    # values explicitly so both runs solve the same problem.
    p3 = SimParams(ndim=3, eps=p2.eps, gamma=p2.gamma, **dims, **kw3)
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = run_2d(p2, phi2, ux0=ux2, uy0=uy2)
        r3 = run_3d(p3, ex(phi2), **ics)
    return r2, r3, pick, mapping


@pytest.mark.parametrize("axis", ["z", "y", "x"])
@pytest.mark.parametrize("mu_v", [1e-3, 1e-5], ids=["matched-viscosity", "viscosity-contrast"])
def test_extruded_3d_run_reproduces_2d_run(axis, mu_v):
    r2, r3, pick, m = _run_pair(axis, mu_v)
    pairs = {"phi": ("phi", "phi"), "pres": ("pres", "pres"),
             "ux": ("ux", m["ux"]), "uy": ("uy", m["uy"])}
    for name, (k2, k3) in pairs.items():
        a2, a3 = r2[k2], pick(r3[k3])
        scale = np.max(np.abs(a2))
        assert scale > 1e-8, f"{name} is trivially zero; the test would not detect anything"
        # ~1e-16 relative in practice (bit-identical on the reference platform); 1e-10 leaves
        # room for different FFT back-ends but is ~6 orders below the 1e-3 mismatch that a
        # missing viscous term produces.
        assert np.max(np.abs(a3 - a2)) < 1e-10 * scale, (axis, mu_v, name)


@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_extruded_3d_solution_stays_invariant_and_transverse_velocity_stays_zero(axis):
    _, r3, _, _ = _run_pair(axis, 1e-5)
    inv_axis = {"z": 0, "y": 1, "x": 2}[axis]
    transverse = {"z": "uz", "y": "uy", "x": "ux"}[axis]
    for name in ("phi", "pres"):
        a = r3[name]
        assert np.max(np.abs(a - np.take(a, [0], axis=inv_axis))) < 1e-12 * np.max(np.abs(a))
    assert np.max(np.abs(r3[transverse])) < 1e-12
