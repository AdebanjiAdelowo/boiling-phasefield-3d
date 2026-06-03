"""
1-D Stefan problem: superheated vapour drives vaporisation of liquid at the wall.

Reproduces the benchmark in Roccon (2025), Section 3.2.

Analytical solution
-------------------
Interface position:  δ(t) = 2ξ √(αᵥ t)
where ξ satisfies:   ξ exp(ξ²) erf(ξ) = St / √π
and the Stefan number is:  St = Cₚ,ᵥ (T_wall − T_sat) / h_lv

Run
---
    cd boiling-phasefield-3d
    python examples/stefan_1d.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import brentq

from src.params      import SimParams
from src.phase_field import ac_rhs_2d, mdot_volumetric
from src.operators   import grad_x_2d
from src.diagnostics import interface_position_1d

# ── Parameters ────────────────────────────────────────────────────────────────
# Matched viscosities and densities (ρᵥ=ρₗ → no flow expansion) so that only
# the thermal problem needs solving — matching Roccon (2025) Section 3.2 setup.
p = SimParams(
    Nx=200, Ny=1,
    Lx=0.2,  Ly=0.2/200,   # effectively 1-D (Ny=1 row)
    dt=5e-4,
    t_end=250.0,     # run from t0=24.7 s to t_f=250 s  (matching paper Section 3.2)
    save_every=500,
    rho_l=1.0,   rho_v=1.0,     # matched densities
    mu_l=0.01,   mu_v=0.01,
    sigma=0.0,                   # no surface tension
    k_l=0.005,   k_v=0.005,
    Cp_l=200.0,  Cp_v=200.0,
    h_lv=1e4,
    T_sat=0.0,
    T_wall=10.0,                 # Stefan number St = Cp*(Tw-Ts)/h_lv = 0.2
    mode='heat_flux',
)

alpha_v = p.k_v / (p.rho_v * p.Cp_v)
St_num  = p.Cp_v * (p.T_wall - p.T_sat) / p.h_lv
print(f"Stefan number  St = {St_num:.4f}")
print(f"αᵥ = {alpha_v:.4e} m²/s")

# Analytical: solve  ξ exp(ξ²) erf(ξ) = St/√π
xi = brentq(lambda xi: xi * np.exp(xi**2) * erf(xi) - St_num / np.sqrt(np.pi),
            1e-8, 10.0)
print(f"ξ  = {xi:.6f}")

def delta_analytical(t):
    return 2 * xi * np.sqrt(alpha_v * t) if t > 0 else 0.0

# ── Initial condition ─────────────────────────────────────────────────────────
# Start at t0 = 24.7 s to avoid the t→0 singularity (same as paper)
t0     = 24.7
delta0 = delta_analytical(t0)
print(f"t₀ = {t0} s  →  δ(t₀) = {delta0:.4f} m")

x   = np.linspace(0, p.Lx, p.Nx, endpoint=False)
X2d = x[np.newaxis, :]           # shape (1, Nx) for 2-D ops with Ny=1

# φ=1 (vapour) for x < δ₀,  φ=0 (liquid) for x > δ₀
phi0 = 0.5 * (1 - np.tanh((X2d - delta0) / (2 * p.eps)))

# Temperature: linear profile in vapour, T_sat in liquid (Roccon 2025 Eq. 27)
T0 = np.where(
    X2d < delta0,
    p.T_wall - (p.T_wall - p.T_sat) * erf(X2d / (2*np.sqrt(alpha_v*t0))) / erf(xi),
    p.T_sat
)

# ── Time integration (1-D: uy = 0, pure diffusion + phase-change) ─────────────
phi = phi0.copy()
T   = T0.copy()
ux  = np.zeros_like(phi)
uy  = np.zeros_like(phi)
dx  = p.dx

t        = t0
n_steps  = int(round((p.t_end - t0) / p.dt))
times    = [t0]
d_num    = [delta0]
d_ana    = [delta0]

print(f"\nRunning 1-D Stefan problem: {p.Nx} points, {n_steps} steps …")

for step in range(n_steps):

    # Vaporisation rate from heat-flux balance (Eq. 12, simplified).
    # In the Stefan problem the liquid is at T_sat (∇T_liquid = 0), so only
    # the vapour-side heat flux drives vaporisation: ṁ = kv (∇T·n̂) / h_lv
    dT_dx  = grad_x_2d(T, dx)
    dphidx = grad_x_2d(phi, dx)
    mag    = np.abs(dphidx) + 1e-14
    nx     = dphidx / mag                         # interface normal
    mdot_surf_local = p.k_v * dT_dx * nx / p.h_lv
    mdot_vol = mdot_surf_local * phi * (1 - phi) / p.eps

    # Allen-Cahn step (no flow)
    A       = ac_rhs_2d(phi, ux, uy, mdot_vol, p)
    phi_new = np.clip(phi + p.dt * A, 0.0, 1.0)

    # Energy: diffusion only in vapour, liquid held at T_sat
    alpha_f = p.k_v / (p.rho_v * p.Cp_v) * phi   # only in vapour
    diff_T  = grad_x_2d(alpha_f * grad_x_2d(T, dx), dx)
    T_new   = T + p.dt * diff_T
    T_new   = np.where(phi_new < 0.5, p.T_sat, T_new)   # liquid at T_sat
    T_new[0, 0] = p.T_wall                              # wall BC

    phi, T = phi_new, T_new
    t += p.dt

    if step % p.save_every == 0:
        d_n = interface_position_1d(phi[0], x)
        d_a = delta_analytical(t)
        if d_n is not None:
            times.append(t)
            d_num.append(d_n)
            d_ana.append(d_a)
            err = abs(d_n - d_a) / d_a * 100
            print(f"  t={t:6.1f} s  δ_num={d_n:.4f} m  δ_ana={d_a:.4f} m  err={err:.2f}%")

print("Done.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(times, np.array(d_ana)*100, 'k-',  lw=2,   label='Analytical  δ=2ξ√(αᵥt)')
axes[0].plot(times, np.array(d_num)*100, 'C0o', ms=5,   label='Numerical')
axes[0].set_xlabel('t  [s]')
axes[0].set_ylabel('Interface position δ  [cm]')
axes[0].set_title('Stefan problem: interface position vs time')
axes[0].legend()

axes[1].plot(x*100, phi[0],         'C0',  lw=2, label='φ  (phase-field)')
axes[1].plot(x*100, T[0]/p.T_wall,  'C3--', lw=2, label='T / T_wall')
axes[1].axvline(delta_analytical(t)*100, color='gray', ls=':', lw=1, label='δ(t) analytical')
axes[1].set_xlabel('x  [cm]')
axes[1].set_title('Phase-field and temperature at t = {:.0f} s'.format(t))
axes[1].legend()

fig.suptitle('1-D Stefan Problem — Phase-Field Validation\n'
             'Roccon (2025) Section 3.2 benchmark', fontweight='bold')
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'stefan_1d_result.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Plot saved to {out}")
plt.show()
