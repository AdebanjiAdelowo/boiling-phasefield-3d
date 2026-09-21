"""
1-D Stefan problem: superheated vapour drives vaporisation of liquid at the wall.

Set-up follows Roccon (2025), Section 3.2 (matched densities, St = 0.2).  The
solver is `src/stefan1d.py`; see that module's docstring for the scheme and for
why each choice was made.

Analytical solution
-------------------
Interface position:  delta(t) = 2 xi sqrt(alpha_v t)
where xi satisfies:  xi exp(xi^2) erf(xi) = St / sqrt(pi)
and the Stefan number is:  St = Cp_v (T_wall - T_sat) / h_lv

The run starts at t0 = 24.7 s from the similarity state (a start away from the
t -> 0 singularity) and ends at t = 250 s.

Run
---
    cd boiling-phasefield-3d
    python examples/stefan_1d.py                 # single run, N = 200 (dx = 1 mm)
    python examples/stefan_1d.py --refine        # N = 50, 100, 200, 400 convergence study
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from src.params import SimParams
from src.stefan1d import run_stefan_1d


def make_params(nx, dt=5e-4, lx=0.2):
    return SimParams(
        Nx=nx, Ny=1, Lx=lx, Ly=lx / nx,
        dt=dt, t_end=250.0,
        rho_l=1.0, rho_v=1.0,          # matched densities: no expansion flow
        mu_l=0.01, mu_v=0.01,
        sigma=0.0,
        k_l=0.005, k_v=0.005,
        Cp_l=200.0, Cp_v=200.0,
        h_lv=1e4,
        T_sat=0.0, T_wall=10.0,        # St = Cp (T_wall - T_sat) / h_lv = 0.2
        mode='heat_flux',
    )


def report(res, label):
    h, ref = res['history'], res['ref']
    print(f"\n{label}:  xi = {ref.xi:.6f}, St = {ref.St:.3f}, alpha_v = {ref.alpha:.3e} m^2/s")
    print("    t [s]   delta_0.5 [m]  delta_mass [m]  delta_ana [m]   err_0.5 [%]  err_mass [%]")
    for tq in (25, 30, 60, 100, 150, 200, 250):
        i = int(np.argmin(np.abs(h['t'] - tq)))
        t, da = h['t'][i], float(ref.delta(h['t'][i]))
        print(f"  {t:7.2f}   {h['delta_half'][i]:12.6f}  {h['delta_mass'][i]:13.6f}  {da:12.6f}"
              f"   {abs(h['delta_half'][i] - da) / da * 100:10.3f}  {abs(h['delta_mass'][i] - da) / da * 100:11.3f}")
    i = -1
    print(f"  crossings of phi=0.5 at end: {h['n_crossings'][i]},  phi in [{h['phi_min'].min():.4f}, {h['phi_max'].max():.4f}],"
          f"  clipped mass: {h['clipped'][i]:.2e} m")
    dE = h['E_sens'][i] - h['E_sens'][0]
    print(f"  energy budget [J/m^2]: wall input {h['E_wall'][i]:.3f} (exact {ref.cumulative_wall_heat(h['t'][0], h['t'][i]):.3f}),"
          f" clamp removal {h['E_clamp'][i]:.3f}, stored change {dE:.3f} (exact {ref.stored_energy(h['t'][i]) - ref.stored_energy(h['t'][0]):.3f}),"
          f"\n    identity  wall - clamp - stored = {h['E_wall'][i] - h['E_clamp'][i] - dE:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=200)
    ap.add_argument('--refine', action='store_true')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    p = make_params(args.nx)
    print(f"1-D Stefan: N = {p.Nx}, dx = {p.dx * 1e3:.3f} mm, eps = {p.eps * 1e3:.3f} mm, dt = {p.dt:g} s")
    res = run_stefan_1d(p, t0=24.7, t_end=250.0, snapshot_times=(250.0,))
    report(res, f"N = {p.Nx}")

    if args.refine:
        print("\nSpatial refinement (eps = 1.5 dx, dt = 5e-4 s), final-time interface error:")
        print("    N     dx [mm]   err_0.5 [%]   err_mass [%]")
        prev = None
        for nx in (50, 100, 200, 400):
            q = make_params(nx)
            r = run_stefan_1d(q, t0=24.7, t_end=250.0, record_every=10 ** 9)
            hh = r['history']
            da = float(r['ref'].delta(hh['t'][-1]))
            e5, em = (abs(hh['delta_half'][-1] - da) / da * 100, abs(hh['delta_mass'][-1] - da) / da * 100)
            rate = "" if prev is None else f"   observed order {np.log2(prev / e5):.2f}"
            print(f"  {nx:4d}   {q.dx * 1e3:7.3f}   {e5:10.4f}   {em:11.4f}{rate}")
            prev = e5

    if args.no_plot:
        return
    h, ref, x = res['history'], res['ref'], res['x']
    phi, T = res['snapshots'][250.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(h['t'], ref.delta(h['t']) * 100, 'k-', lw=2, label='Analytical  delta = 2 xi sqrt(alpha_v t)')
    axes[0].plot(h['t'][::8], h['delta_half'][::8] * 100, 'C0o', ms=4, label='Numerical (phi = 0.5)')
    axes[0].set_xlabel('t  [s]'); axes[0].set_ylabel('Interface position  [cm]')
    axes[0].set_title('Interface position'); axes[0].legend()
    da = ref.delta(h['t'])
    axes[1].semilogy(h['t'], np.abs(h['delta_half'] - da) / da * 100 + 1e-12, label='phi = 0.5')
    axes[1].semilogy(h['t'], np.abs(h['delta_mass'] - da) / da * 100 + 1e-12, label='mass  int(phi) dx')
    axes[1].set_xlabel('t  [s]'); axes[1].set_ylabel('relative error  [%]')
    axes[1].set_title('Interface-position error'); axes[1].legend()
    axes[2].plot(x * 100, phi, 'C0', lw=2, label='phi')
    axes[2].plot(x * 100, T / p.T_wall, 'C3--', lw=2, label='T / T_wall')
    axes[2].plot(x * 100, ref.temperature(x, 250.0) / p.T_wall, 'k:', lw=1.5, label='T / T_wall analytical')
    axes[2].set_xlim(0, 8); axes[2].set_xlabel('x  [cm]')
    axes[2].set_title('Fields at t = 250 s'); axes[2].legend()
    fig.suptitle(f'1-D Stefan problem, N = {p.Nx}', fontweight='bold')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'stefan_1d_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out}")


if __name__ == '__main__':
    main()
