"""Instrumented 1-D lab that reproduces the repository's Stefan scheme EXACTLY
(default options) and lets ONE numerical/model choice at a time be varied.

Nothing in the repo is modified.  `verify_against_repo()` checks bit-level
agreement with the repo's own ac_rhs_2d / inline scheme before any result from
this lab is trusted.
"""
import sys
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

# ---------------------------------------------------------------- benchmark
K, RHO, CP, H, TS, TW = 0.005, 1.0, 200.0, 1e4, 0.0, 10.0
ALPHA = K / (RHO * CP)
ST = CP * (TW - TS) / H
XI = brentq(lambda x: x * np.exp(x * x) * erf(x) - ST / np.sqrt(np.pi), 1e-8, 10.0)


def delta_ana(t):
    return 2.0 * XI * np.sqrt(ALPHA * t)


def T_ana(x, t):
    d = delta_ana(t)
    return np.where(x < d, TW - (TW - TS) * erf(x / (2 * np.sqrt(ALPHA * t))) / erf(XI), TS)


# ---------------------------------------------------------------- operators
def d1(f, h):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * h)


def interface_first(phi, x):
    f = phi - 0.5
    idx = np.where(np.diff(np.sign(f)))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return x[i] + (-f[i] / (f[i + 1] - f[i])) * (x[i + 1] - x[i])


def n_crossings(phi):
    return int(np.sum(np.diff(np.sign(phi - 0.5)) != 0))


class Cfg:
    """Every field is a single experimental knob.  Defaults == repo at HEAD."""
    def __init__(self, **kw):
        self.N = 200
        self.L = 0.2
        self.dt = 5e-4
        self.t0 = 24.7
        self.t_end = 250.0
        self.eps_over_dx = 1.5
        self.gamma = 0.1          # HEAD default via SimParams (max(mdot_surf/rho_v, ...)) = 0.1
        self.clip = True
        self.alpha_mode = "phi"   # 'phi' -> alpha_v*phi (repo) ; 'one' -> alpha_v
        self.clamp = "phi_new<0.5"  # repo: T=Ts where phi_new<0.5 ; 'none'
        self.wall = "node0"       # repo: T[0]=Tw only (periodic wrap left at Ts)
        self.normal = "sign"      # repo: nx = dphi/(|dphi|+1e-14)
        self.mdot_form = "grid"   # repo: k*dTdx*nx/h * phi(1-phi)/eps at grid nodes
        self.out_every = 500
        self.diag = True
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)

    @property
    def dx(self):
        return self.L / self.N

    @property
    def eps(self):
        return self.eps_over_dx * self.dx


def init_state(c):
    x = np.arange(c.N) * c.dx
    d0 = delta_ana(c.t0)
    phi = 0.5 * (1 - np.tanh((x - d0) / (2 * c.eps)))
    T = np.where(x < d0, TW - (TW - TS) * erf(x / (2 * np.sqrt(ALPHA * c.t0))) / erf(XI), TS)
    return x, phi, T


def rhs_phi(phi, mdot_vol, c):
    dx, eps, g = c.dx, c.eps, c.gamma
    dphi = d1(phi, dx)
    mag = np.sqrt(dphi ** 2 + 1e-14)      # repo: sqrt(dphix^2+dphiy^2+1e-14), dphiy=0
    J = g * (eps * dphi - phi * (1 - phi) / mag * dphi)
    return d1(J, dx) + mdot_vol / RHO       # ux=uy=0; source = mdot_vol/rho_v


def mdot_vol_of(phi, T, c):
    dx = c.dx
    dT = d1(T, dx)
    dphi = d1(phi, dx)
    if c.normal == "sign":
        nx = dphi / (np.abs(dphi) + 1e-14)
    else:
        raise ValueError
    return (K * dT * nx / H) * phi * (1 - phi) / c.eps


def step(phi, T, c, acct=None):
    dx = c.dx
    md = mdot_vol_of(phi, T, c)
    A = rhs_phi(phi, md, c)
    phi_new = phi + c.dt * A
    if c.clip:
        clipped = phi_new - np.clip(phi_new, 0.0, 1.0)
        phi_new = np.clip(phi_new, 0.0, 1.0)
    else:
        clipped = np.zeros_like(phi)
    a = ALPHA * phi if c.alpha_mode == "phi" else ALPHA * np.ones_like(phi)
    Tn = T + c.dt * d1(a * d1(T, dx), dx)
    e_clamp = 0.0
    if c.clamp == "phi_new<0.5":
        m = phi_new < 0.5
        e_clamp = RHO * CP * np.sum(Tn[m] - TS) * dx
        Tn = np.where(m, TS, Tn)
    e_wall = 0.0
    if c.wall == "node0":
        e_wall = RHO * CP * (TW - Tn[0]) * dx
        Tn[0] = TW
    if acct is not None:
        acct["src"] += c.dt * np.sum(md) * dx / RHO
        acct["clip"] += np.sum(clipped) * dx
        acct["e_clamp"] += e_clamp
        acct["e_wall"] += e_wall
    return phi_new, Tn


def run(c, snapshots=(), verbose=False):
    x, phi, T = init_state(c)
    n = int(round((c.t_end - c.t0) / c.dt))
    acct = dict(src=0.0, clip=0.0, e_clamp=0.0, e_wall=0.0)
    rec = []
    snaps = {}
    t = c.t0
    M0 = np.sum(phi) * c.dx
    E0 = RHO * CP * np.sum(T - TS) * c.dx
    snap_steps = {int(round((s - c.t0) / c.dt)): s for s in snapshots}

    def record(k):
        tt = c.t0 + k * c.dt
        M = np.sum(phi) * c.dx
        E = RHO * CP * np.sum(T - TS) * c.dx
        rec.append(dict(t=tt, d=interface_first(phi, x), da=delta_ana(tt), M=M, E=E,
                        Msrc=M0 + acct["src"], clip=acct["clip"], eclamp=acct["e_clamp"],
                        ewall=acct["e_wall"], ncross=n_crossings(phi),
                        phimin=phi.min(), phimax=phi.max(), Tmax=T.max(), Tmin=T.min()))

    record(0)
    for k in range(1, n + 1):
        phi, T = step(phi, T, c, acct)
        if k in snap_steps:
            snaps[snap_steps[k]] = (phi.copy(), T.copy())
        if k % c.out_every == 0:
            record(k)
    return dict(rec=rec, snaps=snaps, x=x, phi=phi, T=T, cfg=c, M0=M0, E0=E0)


def verify_against_repo(nsteps=3000):
    """Bit-level check of the lab (default cfg) against the repo's own code path."""
    from src.params import SimParams
    from src.phase_field import ac_rhs_2d
    from src.operators import grad_x_2d
    p = SimParams(Nx=200, Ny=1, Lx=0.2, Ly=0.2 / 200, dt=5e-4, t_end=250.0,
                  rho_l=1.0, rho_v=1.0, mu_l=0.01, mu_v=0.01, sigma=0.0,
                  k_l=0.005, k_v=0.005, Cp_l=200.0, Cp_v=200.0, h_lv=1e4,
                  T_sat=0.0, T_wall=10.0, mode="heat_flux")
    c = Cfg()
    p.gamma = 0.1   # the heat_flux-mode default at HEAD c9a2842 (inherited from mdot_surf); pinned explicitly
    assert abs(p.gamma - c.gamma) < 1e-15 and abs(p.eps - c.eps) < 1e-15, (p.gamma, p.eps)
    x, phi, T = init_state(c)
    phiR, TR = phi[None, :].copy(), T[None, :].copy()
    ux = np.zeros_like(phiR); uy = np.zeros_like(phiR); dx = p.dx
    maxd = 0.0
    for _ in range(nsteps):
        # ---- verbatim from examples/stefan_1d.py
        dT_dx = grad_x_2d(TR, dx); dphidx = grad_x_2d(phiR, dx)
        mag = np.abs(dphidx) + 1e-14; nx = dphidx / mag
        mdot_surf_local = p.k_v * dT_dx * nx / p.h_lv
        mdot_vol = mdot_surf_local * phiR * (1 - phiR) / p.eps
        A = ac_rhs_2d(phiR, ux, uy, mdot_vol, p)
        phi_new = np.clip(phiR + p.dt * A, 0.0, 1.0)
        alpha_f = p.k_v / (p.rho_v * p.Cp_v) * phiR
        diff_T = grad_x_2d(alpha_f * grad_x_2d(TR, dx), dx)
        T_new = TR + p.dt * diff_T
        T_new = np.where(phi_new < 0.5, p.T_sat, T_new)
        T_new[0, 0] = p.T_wall
        phiR, TR = phi_new, T_new
        phi, T = step(phi, T, c)
        maxd = max(maxd, np.abs(phi - phiR[0]).max(), np.abs(T - TR[0]).max())
    return maxd


if __name__ == "__main__":
    print("xi =", XI, " St =", ST, " alpha =", ALPHA)
    print("max |lab - repo| over 3000 steps:", verify_against_repo(3000))
