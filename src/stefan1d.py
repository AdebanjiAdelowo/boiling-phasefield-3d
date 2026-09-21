"""1-D Stefan problem with matched densities: analytic reference + non-periodic solver.

Physical problem (Roccon 2025, Section 3.2 set-up)
--------------------------------------------------
A vapour layer 0 < x < delta(t) sits against a wall held at T_wall > T_sat; the
liquid (x > delta) is uniformly at T_sat.  With rho_v = rho_l there is no flow.
Heat conducted through the vapour is consumed as latent heat at the interface:

    rho h_lv d(delta)/dt = -k_v dT/dx |_(delta^-)          (Stefan condition)

which has the similarity solution

    delta(t) = 2 xi sqrt(alpha_v t),   xi exp(xi^2) erf(xi) = St / sqrt(pi),
    T(x,t)   = T_wall - (T_wall - T_sat) erf(x / (2 sqrt(alpha_v t))) / erf(xi),
    St       = Cp_v (T_wall - T_sat) / h_lv .

`StefanReference` implements this and its exact energy budget.  It has been
checked independently of the similarity formula (see tests/test_stefan_1d.py).

Numerical scheme (what `run_stefan_1d` does, and why)
-----------------------------------------------------
Each choice below was examined by a controlled experiment (necessary in the tested
configuration; none is claimed as the unique root cause of the legacy failure); see the
"Stefan problem" section of docs/Phase_Field_Boiling_Solver_Technical_Documentation.md.

* Non-periodic grid: node 0 is the wall (Dirichlet T_wall), the last node is an
  outlet held at T_sat.  The periodic `np.roll` operators used elsewhere in the
  library wrap the wall onto the outlet, which is wrong for this problem.
* Energy equation: compact, conservative, face-centred diffusion stencil with
  alpha = alpha_mix(phi) (Roccon Eq. R.10).  The nested central difference
  D1(alpha D1(T)) used by the periodic library decouples even and odd nodes.
* The non-superheated phase is held at T_sat (Roccon's saturation argument):
  T = T_sat wherever phi < 0.5 after the phase-field update.  No latent-heat
  source S_t is used; latent heat enters only through the vaporisation rate.
* Vaporisation rate: k_v |dT/dx| / h_lv at the phi = 0.5 iso-point, with the
  gradient linearly extrapolated to the front from central differences taken
  `probe_cells` and 2 x `probe_cells` behind it (a 1-D probe method).  The
  liquid side has zero gradient (liquid at T_sat).
* Mass source: m_surf * |dphi/dx|, NOT m_surf * phi(1-phi)/eps.  The two agree
  only while the layer has exactly its equilibrium width; the latter makes the
  production rate proportional to the current layer width, which is unstable
  (the layer widens, produces more vapour, and the front runs ahead).
  The integral of |dphi/dx| is exactly 1 for a monotone profile, so the
  production rate is exactly m_surf whatever the profile width.
* Allen-Cahn mobility gamma = gamma_factor * m_surf / rho_v, i.e. tied to the
  interface speed (Mirjalili et al. 2020 boundedness needs gamma of the order
  of the interface speed; gamma_factor = 0.25 < 0.5 produced trace clipping at
  eps = 1.5 dx).  With this |grad phi| source the result was insensitive to gamma
  over 1e-4 .. 0.1 m/s; a much larger gamma pinned the front only in the legacy
  pairing of nested operator and phi(1-phi)/eps source.
* Conservative Allen-Cahn fluxes are evaluated at cell faces (compact), with
  zero flux through both ends.

Scope and limits
----------------
Matched densities (rho_l == rho_v), 1-D, one superheated phase, first-order
explicit Euler in time.  Not a validation of the 2-D/3-D heat-flux closure
(`energy.mdot_from_heatflux_2d`), which this module does not use.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from .diagnostics import count_interface_crossings_1d, interface_position_1d, vapour_thickness_1d
from .energy import alpha_mix
from .params import SimParams


# ── Analytic reference ────────────────────────────────────────────────────────

class StefanReference:
    """Similarity solution of the one-phase Stefan problem for the given params."""

    def __init__(self, p: SimParams):
        if p.rho_l != p.rho_v:
            raise ValueError("StefanReference assumes matched densities (rho_l == rho_v)")
        self.k, self.rho, self.Cp, self.h = p.k_v, p.rho_v, p.Cp_v, p.h_lv
        self.T_sat, self.T_wall = p.T_sat, p.T_wall
        self.dT = p.T_wall - p.T_sat
        self.alpha = self.k / (self.rho * self.Cp)
        self.St = self.Cp * self.dT / self.h
        self.xi = brentq(lambda s: s * np.exp(s * s) * erf(s) - self.St / np.sqrt(np.pi),
                         1e-12, 10.0, xtol=1e-15)

    def delta(self, t):
        """Interface position [m]."""
        return 2.0 * self.xi * np.sqrt(self.alpha * np.asarray(t, dtype=float))

    def speed(self, t):
        """d(delta)/dt [m/s]."""
        return self.xi * np.sqrt(self.alpha / np.asarray(t, dtype=float))

    def temperature(self, x, t):
        """Vapour temperature profile; T_sat for x >= delta(t)."""
        x = np.asarray(x, dtype=float)
        T = self.T_wall - self.dT * erf(x / (2.0 * np.sqrt(self.alpha * t))) / erf(self.xi)
        return np.where(x < self.delta(t), T, self.T_sat)

    def wall_flux(self, t):
        """Heat flux entering from the wall, -k dT/dx|_0 [W/m^2]."""
        return self.k * self.dT / (erf(self.xi) * np.sqrt(np.pi * self.alpha * np.asarray(t, dtype=float)))

    def interface_flux(self, t):
        """Heat flux arriving at the interface, -k dT/dx|_(delta^-) = rho h d(delta)/dt [W/m^2]."""
        return self.rho * self.h * self.speed(t)

    def cumulative_wall_heat(self, t0, t):
        """Integral of wall_flux from t0 to t [J/m^2]."""
        return 2.0 * self.k * self.dT * (np.sqrt(t) - np.sqrt(t0)) / (erf(self.xi) * np.sqrt(np.pi * self.alpha))

    def stored_energy(self, t):
        """Sensible energy rho Cp int_0^delta (T - T_sat) dx [J/m^2] (closed form)."""
        s = 2.0 * np.sqrt(self.alpha * t)
        xi = self.xi
        integral_erf = s * (xi * erf(xi) + (np.exp(-xi * xi) - 1.0) / np.sqrt(np.pi))
        return self.rho * self.Cp * self.dT * (self.delta(t) - integral_erf / erf(xi))

    def latent_energy(self, t0, t):
        """Latent heat absorbed between t0 and t, rho h (delta(t) - delta(t0)) [J/m^2]."""
        return self.rho * self.h * (self.delta(t) - self.delta(t0))


# ── Discrete building blocks (non-periodic, node 0 = wall, node N-1 = outlet) ──

def phase_rhs_1d(phi, mdot_vol, gamma, p: SimParams):
    """RHS of the conservative Allen-Cahn equation (Roccon Eq. 1, u = 0) in face-flux
    form with zero flux through both ends.  With mdot_vol = 0 it conserves
    sum(phi) exactly (algebraic identity)."""
    dx, eps = p.dx, p.eps
    dphi_f = (phi[1:] - phi[:-1]) / dx
    phi_f = 0.5 * (phi[1:] + phi[:-1])
    mag_f = np.sqrt(dphi_f ** 2 + 1e-14)
    J_f = gamma * (eps * dphi_f - phi_f * (1.0 - phi_f) / mag_f * dphi_f)
    rhs = np.zeros_like(phi)
    rhs[:-1] += J_f / dx
    rhs[1:] -= J_f / dx
    return rhs + mdot_vol / p.rho_v


def temperature_rhs_1d(T, alpha, dx):
    """dT/dt = d/dx(alpha dT/dx) with face-averaged alpha (compact, conservative).
    Node 0 is a half-cell wall node (mirror ghost: rhs = 2 F_{1/2} / dx); its value is
    overwritten by the Dirichlet condition.  The last node is an outlet held fixed."""
    alpha_f = 0.5 * (alpha[1:] + alpha[:-1])
    flux = alpha_f * (T[1:] - T[:-1]) / dx
    rhs = np.zeros_like(T)
    rhs[1:-1] = (flux[1:] - flux[:-1]) / dx
    rhs[0] = 2.0 * flux[0] / dx
    return rhs


def central_gradient_1d(f, dx):
    """Central difference in the interior, zero at both ends."""
    g = np.zeros_like(f)
    g[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx)
    return g


def interface_heat_flux_1d(phi, T, p: SimParams, probe_cells=2.0):
    """Heat flux k_v |dT/dx| arriving at the phi = 0.5 front from the vapour side [W/m^2].

    Central-difference gradients (dx-wide, from linearly interpolated T) are taken at
    h and 2h behind the front, h = probe_cells * dx, and linearly extrapolated to the
    front: g(x_f) ~ 2 g(x_f - h) - g(x_f - 2h).  Sampling behind the front keeps the
    stencil clear of the saturation clamp at the front.  Requires x_f > 2h + dx.
    Returns 0 when no front exists."""
    dx = p.dx
    x = np.arange(phi.size) * dx
    x_f = interface_position_1d(phi, x)
    if x_f is None:
        return 0.0
    h = probe_cells * dx

    def grad_at(xp):
        return (np.interp(xp + dx, x, T) - np.interp(xp - dx, x, T)) / (2.0 * dx)

    g_front = 2.0 * grad_at(x_f - h) - grad_at(x_f - 2.0 * h)
    return max(-p.k_v * g_front, 0.0)


# ── Solver ────────────────────────────────────────────────────────────────────

@dataclass
class StefanOptions:
    gamma_factor: float = 1.0        # gamma = gamma_factor * m_surf / rho_v
    gamma_min: float = 1e-9          # floor so the sharpening term never vanishes exactly
    probe_cells: float = 2.0         # probe distance h in cells
    source_shape: str = "gradient"   # 'gradient' (|dphi/dx|, exact) | 'equilibrium' (phi(1-phi)/eps)
    clip: bool = True                # clip phi to [0,1]; the clipped mass is reported, not hidden


def initial_state(p: SimParams, ref: StefanReference, t0):
    """phi, T on the grid at time t0 from the similarity solution."""
    x = np.arange(p.Nx) * p.dx
    d0 = float(ref.delta(t0))
    phi = 0.5 * (1.0 - np.tanh((x - d0) / (2.0 * p.eps)))
    T = np.asarray(ref.temperature(x, t0), dtype=float)
    T[0] = p.T_wall
    return x, phi, T


def run_stefan_1d(p: SimParams, t0=24.7, t_end=250.0, options: StefanOptions = None,
                  record_every=500, snapshot_times=()):
    """Advance the 1-D Stefan problem from the similarity state at t0 to t_end.

    p.Nx, p.Lx, p.dt, p.eps and the fluid/thermal properties are used; p.gamma,
    p.mode and p.mdot_surf are NOT (gamma follows the interface speed).  t0 > 0
    avoids the t -> 0 singularity of the similarity solution.

    Returns dict(history=dict of arrays, x, phi, T, snapshots, ref).
    history keys: t, delta_half, delta_mass, m_surf, gamma, mass (sum(phi) dx), mass_source
    (cumulative integral of the source), clipped (cumulative clipped mass), E_sens, E_wall
    (cumulative wall heat), E_clamp (cumulative heat removed by the saturation clamp),
    q_wall, n_crossings, phi_min, phi_max.  Energies are per unit area [J/m^2].
    """
    opts = options or StefanOptions()
    ref = StefanReference(p)
    dx, dt = p.dx, p.dt
    alpha_max = max(p.k_v / (p.rho_v * p.Cp_v), p.k_l / (p.rho_l * p.Cp_l))
    if alpha_max * dt / dx ** 2 > 0.5:
        raise ValueError(f"explicit diffusion unstable: alpha*dt/dx^2 = {alpha_max * dt / dx ** 2:.3f} > 0.5")

    x, phi, T = initial_state(p, ref, t0)
    rho_cp = p.rho_v * p.Cp_v
    n_steps = int(round((t_end - t0) / dt))
    hist = {k: [] for k in ("t", "delta_half", "delta_mass", "m_surf", "gamma", "mass", "mass_source",
                            "clipped", "E_sens", "E_wall", "E_clamp", "q_wall", "n_crossings",
                            "phi_min", "phi_max")}
    acc = dict(src=0.0, clip=0.0, E_wall=0.0, E_clamp=0.0)
    snaps = {}
    snap_steps = {int(round((s - t0) / dt)): s for s in snapshot_times}
    m_surf = gamma = 0.0

    def stored_energy(T):
        return rho_cp * (np.sum(T[1:] - p.T_sat) + 0.5 * (T[0] - p.T_sat)) * dx

    def record(k):
        d = interface_position_1d(phi, x)
        hist["t"].append(t0 + k * dt)
        hist["delta_half"].append(np.nan if d is None else d)
        hist["delta_mass"].append(vapour_thickness_1d(phi, dx))
        hist["m_surf"].append(m_surf)
        hist["gamma"].append(gamma)
        hist["mass"].append(np.sum(phi) * dx)
        hist["mass_source"].append(acc["src"])
        hist["clipped"].append(acc["clip"])
        hist["E_sens"].append(stored_energy(T))
        hist["E_wall"].append(acc["E_wall"])
        hist["E_clamp"].append(acc["E_clamp"])
        hist["q_wall"].append(-p.k_v * (-3.0 * T[0] + 4.0 * T[1] - T[2]) / (2.0 * dx))
        hist["n_crossings"].append(count_interface_crossings_1d(phi))
        hist["phi_min"].append(phi.min())
        hist["phi_max"].append(phi.max())

    record(0)
    for k in range(1, n_steps + 1):
        q_int = interface_heat_flux_1d(phi, T, p, opts.probe_cells)
        m_surf = q_int / p.h_lv
        shape = np.abs(central_gradient_1d(phi, dx)) if opts.source_shape == "gradient" \
            else phi * (1.0 - phi) / p.eps
        mdot_vol = m_surf * shape
        gamma = max(opts.gamma_factor * m_surf / p.rho_v, opts.gamma_min)

        phi_new = phi + dt * phase_rhs_1d(phi, mdot_vol, gamma, p)
        if opts.clip:
            acc["clip"] += np.sum(phi_new - np.clip(phi_new, 0.0, 1.0)) * dx
            phi_new = np.clip(phi_new, 0.0, 1.0)
        acc["src"] += dt * np.sum(mdot_vol) * dx / p.rho_v

        T_new = T + dt * temperature_rhs_1d(T, alpha_mix(phi, p), dx)
        cold = phi_new < 0.5
        acc["E_clamp"] += rho_cp * np.sum(T_new[cold] - p.T_sat) * dx
        T_new = np.where(cold, p.T_sat, T_new)
        acc["E_wall"] += rho_cp * (p.T_wall - T_new[0]) * 0.5 * dx      # half-cell wall node
        T_new[0] = p.T_wall

        phi, T = phi_new, T_new
        if k in snap_steps:
            snaps[snap_steps[k]] = (phi.copy(), T.copy())
        if k % record_every == 0:
            record(k)
    if n_steps > 0 and n_steps % record_every != 0:
        record(n_steps)                     # always report the final state

    return dict(history={k: np.array(v) for k, v in hist.items()}, x=x, phi=phi, T=T,
                snapshots=snaps, ref=ref, t0=t0, params=p, options=opts)
