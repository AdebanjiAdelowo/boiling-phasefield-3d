"""lab2: coupled 1-D Stefan lab with independent switches (each = ONE numerical/model choice).
Defaults reproduce the repo at HEAD exactly (checked in check_lab2_vs_repo).  Nothing in the repo is touched."""
import sys
import numpy as np
from stefan_lab import (K, RHO, CP, H, TS, TW, ALPHA, ST, XI, delta_ana, T_ana,
                        interface_first, n_crossings)


class C:
    def __init__(self, **kw):
        self.N = 200; self.L = 0.2; self.dt = 5e-4; self.t0 = 24.7; self.t_end = 250.0
        self.eps_over_dx = 1.5
        self.gamma = 0.1                 # 'adaptive' or a number.  0.1 = HEAD default
        self.gamma_c = 1.0               # adaptive: gamma = gamma_c * v_front
        self.bc = "periodic"             # 'periodic' (np.roll, repo) | 'wall' (reflect ghosts at both ends, no wrap)
        self.tstencil = "wide"           # 'wide' = D1(alpha*D1(T)) (repo) | 'compact' = face-centred conservative
        self.wall_ghost = "Ts"           # periodic only: value of the wrapped neighbour of node 0 (repo: Ts, i.e. left alone)
        self.closure = "central_sign"    # 'central_sign' (repo) | 'backward_sign' | 'probe'
        self.dprobe = 3.0                # probe distance in cells
        self.dprobe2 = 2.0               # probe2 distance in cells
        self.src = "phi1mphi"            # mass-source shape: 'phi1mphi' (repo, equilibrium approx) | 'gradphi' (exact |grad phi|)
        self.acop = "nested"             # AC flux divergence: 'nested' D1(J) (repo) | 'compact' face fluxes
        self.clamp_on = "phi_new"        # repo: T=Ts where phi_new<0.5
        self.alpha_mode = "phi"          # repo: alpha_v*phi
        self.out_every = 500
        for k, v in kw.items():
            if not hasattr(self, k): raise AttributeError(k)
            setattr(self, k, v)

    dx = property(lambda s: s.L / s.N)
    eps = property(lambda s: s.eps_over_dx * s.dx)


# ------------------------------------------------------------ neighbour access
def nbr(f, c, sign_T=False):
    """returns (f_{i-1}, f_{i+1}).  periodic: roll.  wall: even reflection about the end nodes."""
    if c.bc == "periodic":
        return np.roll(f, 1), np.roll(f, -1)
    fm = np.concatenate(([f[1]], f[:-1]))
    fp = np.concatenate((f[1:], [f[-2]]))
    return fm, fp


def D1(f, c):
    fm, fp = nbr(f, c)
    return (fp - fm) / (2 * c.dx)


def front_speed_scale(phi, T, c):
    return None


def probe_flux(phi, T, c):
    """k|T_x|/(rho h) style: returns msurf [kg/m2/s] from a 2nd-order one-sided probe behind phi=0.5 with T_f=Ts."""
    x = np.arange(c.N) * c.dx
    xf = interface_first(phi, x)
    if np.isnan(xf): return 0.0
    h = c.dprobe * c.dx
    T1 = np.interp(xf - h, x, T); T2 = np.interp(xf - 2 * h, x, T)
    g = (3 * TS - 4 * T1 + T2) / (2 * h)
    return max(-K * g / H, 0.0)


def probe2_flux(phi, T, c):
    """extrapolated central-difference gradient: g(xf) ~ 2 g(xf-h) - g(xf-2h), h=dprobe2*dx, gradients from interpolated T at +-dx."""
    x = np.arange(c.N) * c.dx
    xf = interface_first(phi, x)
    if np.isnan(xf): return 0.0
    h = c.dprobe2 * c.dx
    gc = lambda xp: (np.interp(xp + c.dx, x, T) - np.interp(xp - c.dx, x, T)) / (2 * c.dx)
    g0 = 2 * gc(xf - h) - gc(xf - 2 * h)
    return max(-K * g0 / H, 0.0)


def mdot_vol_of(phi, T, c):
    dx, eps = c.dx, c.eps
    dphi = D1(phi, c)
    if c.closure in ("central_sign", "backward_sign"):
        nx = dphi / (np.abs(dphi) + 1e-14)
        if c.closure == "central_sign":
            g = D1(T, c)
        else:
            Tm, _ = nbr(T, c); g = (T - Tm) / dx
        return (K * g * nx / H) * phi * (1 - phi) / eps
    shape = phi * (1 - phi) / eps if c.src == "phi1mphi" else np.abs(dphi)
    if c.closure == "probe":
        return probe_flux(phi, T, c) * shape
    if c.closure == "probe2":
        return probe2_flux(phi, T, c) * shape
    raise ValueError(c.closure)


def rhs_phi(phi, md, c, gam):
    dx, eps = c.dx, c.eps
    dphi = D1(phi, c)
    mag = np.sqrt(dphi ** 2 + 1e-14)
    if c.acop == "nested":
        J = gam * (eps * dphi - phi * (1 - phi) / mag * dphi)
        return D1(J, c) + md / RHO
    # compact: fluxes on faces i+1/2, normal from the face gradient
    fm, fp = nbr(phi, c)
    dphif = (fp - phi) / dx
    phif = 0.5 * (phi + fp)
    magf = np.sqrt(dphif ** 2 + 1e-14)
    Jf = gam * (eps * dphif - phif * (1 - phif) / magf * dphif)
    if c.bc == "wall":
        Jf[-1] = 0.0
        Jfm = np.concatenate(([0.0], Jf[:-1]))
    else:
        Jfm = np.roll(Jf, 1)
    return (Jf - Jfm) / dx + md / RHO


def T_diffuse(phi, T, c):
    dx = c.dx
    a = ALPHA * phi if c.alpha_mode == "phi" else ALPHA * np.ones_like(phi)
    if c.tstencil == "wide":
        return D1(a * D1(T, c), c)
    Tm, Tp = nbr(T, c)
    am, ap = nbr(a, c)
    afp = 0.5 * (a + ap); afm = 0.5 * (a + am)
    out = (afp * (Tp - T) - afm * (T - Tm)) / dx ** 2
    if c.bc == "wall":
        out[-1] = -afm[-1] * (T[-1] - Tm[-1]) / dx ** 2 * 0.0  # outlet: zero flux; alpha=0 there anyway
    return out


def step(phi, T, c, acct):
    dx = c.dx
    if c.closure == "probe":
        ms = probe_flux(phi, T, c)
    elif c.closure == "probe2":
        ms = probe2_flux(phi, T, c)
    else:
        ms = None
    md = mdot_vol_of(phi, T, c)
    if c.gamma == "adaptive":
        vscale = ms / RHO if ms is not None else np.max(np.abs(md)) * c.eps / 0.25 / RHO
        gam = max(c.gamma_c * vscale, 1e-9)
    else:
        gam = c.gamma
    A = rhs_phi(phi, md, c, gam)
    pn = phi + c.dt * A
    clipped = pn - np.clip(pn, 0, 1)
    pn = np.clip(pn, 0, 1)
    Tn = T + c.dt * T_diffuse(phi, T, c)
    m = pn < 0.5
    e_clamp = RHO * CP * np.sum(Tn[m] - TS) * dx
    Tn = np.where(m, TS, Tn)
    if c.bc == "periodic" and c.wall_ghost == "Tw":
        Tn[-1] = TW
    e_wall = RHO * CP * (TW - Tn[0]) * dx
    Tn[0] = TW
    acct["src"] += c.dt * np.sum(md) * dx / RHO
    acct["clip"] += np.sum(clipped) * dx
    acct["e_clamp"] += e_clamp
    acct["e_wall"] += e_wall
    acct["gam"] = gam
    acct["ms"] = ms if ms is not None else np.nan
    return pn, Tn


def init(c):
    x = np.arange(c.N) * c.dx
    d0 = delta_ana(c.t0)
    phi = 0.5 * (1 - np.tanh((x - d0) / (2 * c.eps)))
    T = np.where(x < d0, T_ana(x, c.t0), TS).astype(float)
    T[0] = TW
    return x, phi, T


def run(c, snapshots=()):
    x, phi, T = init(c)
    n = int(round((c.t_end - c.t0) / c.dt))
    acct = dict(src=0.0, clip=0.0, e_clamp=0.0, e_wall=0.0, gam=np.nan, ms=np.nan)
    rec = []; snaps = {}
    snap_steps = {int(round((s - c.t0) / c.dt)): s for s in snapshots}
    M0 = phi.sum() * c.dx; E0 = RHO * CP * np.sum(T - TS) * c.dx

    def record(k):
        tt = c.t0 + k * c.dt
        rec.append(dict(t=tt, d=interface_first(phi, x), da=delta_ana(tt), M=phi.sum() * c.dx - M0,
                        E=RHO * CP * np.sum(T - TS) * c.dx - E0, Msrc=acct["src"], clip=acct["clip"],
                        eclamp=acct["e_clamp"], ewall=acct["e_wall"], ncross=n_crossings(phi),
                        gam=acct["gam"], ms=acct["ms"], Tmin=T.min(), Tmax=T.max(), qw=-K*(-3*T[0]+4*T[1]-T[2])/(2*c.dx),
                        saw=float(np.max(np.abs(T[2:40] - 0.5 * (T[1:39] + T[3:41]))))))
    record(0)
    for k in range(1, n + 1):
        phi, T = step(phi, T, c, acct)
        if k in snap_steps: snaps[snap_steps[k]] = (phi.copy(), T.copy())
        if k % c.out_every == 0: record(k)
    return dict(rec=rec, snaps=snaps, x=x, phi=phi, T=T, cfg=c)


def errs_at(r, times=(25, 26, 30, 60, 100, 250)):
    out = []
    for tq in times:
        R = min(r["rec"], key=lambda R: abs(R["t"] - tq))
        out.append(abs(R["d"] - R["da"]) / R["da"] * 100 if not np.isnan(R["d"]) else np.nan)
    return out


def summary(name, r):
    e = errs_at(r); R = r["rec"][-1]
    print(f"{name:<44}" + " ".join(f"{v:7.2f}" for v in e) + f" | maxsaw={max(q['saw'] for q in r['rec']):6.3f} ncross_end={R['ncross']:3d} clipΣ={R['clip']:+.1e}", flush=True)


def check_lab2_vs_repo(nsteps=3000):
    from stefan_lab import Cfg, step as step1, init_state
    c1 = Cfg(); x, p1, T1 = init_state(c1); a1 = dict(src=0.0, clip=0.0, e_clamp=0.0, e_wall=0.0)
    c2 = C(); x, p2, T2 = init(c2); a2 = dict(src=0.0, clip=0.0, e_clamp=0.0, e_wall=0.0, gam=0, ms=0)
    m = 0.0
    for _ in range(nsteps):
        p1, T1 = step1(p1, T1, c1, a1); p2, T2 = step(p2, T2, c2, a2)
        m = max(m, np.abs(p1 - p2).max(), np.abs(T1 - T2).max())
    return m


if __name__ == "__main__":
    print("lab2 default vs lab1 (== repo) max diff over 3000 steps:", check_lab2_vs_repo())
