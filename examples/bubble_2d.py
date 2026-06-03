"""
2-D vapour bubble growth at a constant prescribed vaporisation rate.

Reproduces the benchmark in Roccon (2025), Section 3.4.

Analytical solution
-------------------
When ṁ (per unit surface) is constant and uniform, the bubble radius grows as:
    R(t) = R₀ + (ṁ / ρᵥ) t

Run
---
    cd boiling-phasefield-3d
    python examples/bubble_2d.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.params      import SimParams
from src.solver      import run_2d
from src.diagnostics import bubble_radius_2d

# ── Parameters (matching Roccon 2025 Section 3.4) ─────────────────────────────
p = SimParams(
    Nx=64, Ny=64,
    Lx=0.01, Ly=0.01,
    dt=5e-6,
    t_end=0.008,
    save_every=200,
    rho_l=1000.0, rho_v=1.0,      # density ratio 0.001 (matching Roccon 2025)
    mu_l=1e-3,   mu_v=1e-3,
    sigma=0.0,    # sigma=0 here: analytical R(t) is independent of surface tension.
                  # Non-zero sigma with explicit CSF requires dt ~ sqrt(rho_v*dx³/sigma)
                  # ~ 1e-9 s — implicit surface tension is PhD Year-2 work.
    mode='prescribed',
    mdot_surf=0.1,                 # prescribed surface vaporisation rate [kg/m²s]
)

# ── Initial condition: circular bubble at domain centre ───────────────────────
x = np.linspace(0, p.Lx, p.Nx, endpoint=False)
y = np.linspace(0, p.Ly, p.Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

R0 = 0.001   # initial bubble radius [m]
r  = np.sqrt((X - p.Lx/2)**2 + (Y - p.Ly/2)**2)

# Smooth tanh interface profile: φ=1 inside bubble (vapour), φ=0 outside (liquid)
phi0 = 0.5 * (1 - np.tanh((r - R0) / (2 * p.eps)))

# ── Diagnostics storage ───────────────────────────────────────────────────────
times, R_num, R_ana = [], [], []

def callback(step, t, state):
    R = bubble_radius_2d(state['phi'], p.dx, p.dy)
    Ra = R0 + (p.mdot_surf / p.rho_v) * t
    times.append(t)
    R_num.append(R)
    R_ana.append(Ra)
    err = abs(R - Ra) / Ra * 100
    print(f"  step {step:5d}  t={t:.5f} s  "
          f"R_num={R*1e3:.4f} mm  R_ana={Ra*1e3:.4f} mm  err={err:.2f}%")

# ── Run simulation ────────────────────────────────────────────────────────────
result = run_2d(p, phi0, callback=callback)

# ── Plot results ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(np.array(times)*1e3, np.array(R_ana)*1e3, 'k-',  lw=2,   label='Analytical')
axes[0].plot(np.array(times)*1e3, np.array(R_num)*1e3, 'C0o-', ms=4, lw=1.5, label='Numerical')
axes[0].set_xlabel('t  [ms]')
axes[0].set_ylabel('Bubble radius  [mm]')
axes[0].set_title('R(t) = R₀ + (ṁ/ρᵥ) t')
axes[0].legend()

im = axes[1].imshow(result['phi'], cmap='RdBu_r', origin='lower',
                    extent=[0, p.Lx*1e3, 0, p.Ly*1e3], vmin=0, vmax=1)
axes[1].set_xlabel('x  [mm]')
axes[1].set_ylabel('y  [mm]')
axes[1].set_title('Phase-field φ at t = {:.1f} ms'.format(result['t']*1e3))
plt.colorbar(im, ax=axes[1], label='φ  (0=liquid, 1=vapour)')

fig.suptitle('2-D Vapour Bubble Growth — Phase-Field Method\n'
             'Roccon (2025) Section 3.4 benchmark', fontweight='bold')
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'bubble_2d_result.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out}")
plt.show()
