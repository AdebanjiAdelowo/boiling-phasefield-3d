# boiling-phasefield-3d

Python prototype of a phase-field solver for boiling heat transfer, extending
the validated 1D/2D method of **Roccon (2025)** toward full 3D DNS.

> Roccon A. (2025). *Boiling heat transfer by phase-field method.*
> Acta Mechanica 236, 5623–5638. https://doi.org/10.1007/s00707-024-04122-7

---

## System requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | Tested on 3.12 (Anaconda) |
| NumPy, SciPy, Matplotlib | See `requirements.txt` |
| Mac / Linux | Works on Apple Silicon (M1/M2/M3), no CUDA needed |
| GPU / Fortran | **Not required** for this prototype |

The production HPC code will require NVIDIA GPU + Fortran + MPI (not available
on Apple Silicon). This prototype runs entirely in Python/NumPy.

---

## Physics

The solver couples four governing equations on a uniform Cartesian grid,
following Roccon (2025) Section 2:

| Equation | Paper ref | What it does |
|---|---|---|
| Conservative Allen-Cahn | Eq. 1 | Tracks the vapour-liquid interface via φ ∈ [0,1] without geometric reconstruction |
| Mass conservation | Eq. 3 | Accounts for density jump across the interface during vaporisation |
| Navier-Stokes (one-fluid) | Eq. 4 | Flow field; density and viscosity vary continuously with φ |
| Energy equation | Eq. 9 | Temperature field with latent-heat source term |

**Key numerical property:** the one-fluid NS formulation yields a
*constant-coefficient* Poisson equation for pressure (Eq. 18), which is solved
via FFT-based direct solvers. This property holds in 3D and makes the code
directly GPU-portable, which is the core motivation for choosing this method.

**Vaporisation rate:** computed in two ways (selectable via `SimParams.mode`):
- `'prescribed'`: surface rate ṁ is given directly (bubble growth benchmark)
- `'heat_flux'`: ṁ computed from Rankine-Hugoniot heat-flux balance (Eq. 12)
  at the interface (Stefan problem, turbulent boiling)

---

## Project structure

```
boiling-phasefield-3d/
│
├── src/                        Core solver library
│   ├── params.py               SimParams dataclass, all physical and numerical
│   │                           parameters in one place with sensible defaults
│   │
│   ├── operators.py            2nd-order central-difference spatial operators
│   │                           (∂/∂x, ∂/∂y, ∂/∂z, ∇, ∇·, ∇²) for both 2-D
│   │                           and 3-D. Periodic BCs via np.roll.
│   │
│   ├── phase_field.py          Conservative Allen-Cahn RHS (Eq. 1) with
│   │                           vaporisation source. Provides ac_rhs_2d and
│   │                           ac_rhs_3d. Also mdot_volumetric() for the
│   │                           prescribed-rate mode (Eq. 2).
│   │
│   ├── pressure.py             FFT-based solver for the constant-coefficient
│   │                           Poisson equation ∇²p = rhs (Eq. 18). Uses
│   │                           discrete Laplacian eigenvalues. Provides
│   │                           solve_poisson_2d and solve_poisson_3d.
│   │
│   ├── flow.py                 Navier-Stokes projection-correction method
│   │                           (Eqs. 16-20). Mixture density/viscosity (Eqs.
│   │                           5-6). Surface tension via CSF model (Eq. 7).
│   │                           Provides ns_step_2d and ns_step_3d.
│   │
│   ├── energy.py               Energy equation RHS (Eq. 9) with latent-heat
│   │                           source Sₜ (Eq. 11). Simplified heat-flux
│   │                           vaporisation rate mdot_from_heatflux_2d.
│   │                           Mixture thermal diffusivity (Eq. 10).
│   │
│   ├── solver.py               Main time-integration loop. Couples Allen-Cahn,
│   │                           energy, and Navier-Stokes in the correct order
│   │                           (Roccon 2025 Section 2.5). Provides run_2d and
│   │                           run_3d. Accepts a callback for diagnostics.
│   │
│   └── diagnostics.py          Post-processing: bubble_radius_2d/3d,
│                               interface_position_1d, nusselt_number_2d,
│                               lp_norm, kinetic_energy_2d.
│
├── examples/
│   ├── bubble_2d.py            2-D vapour bubble growth at constant
│   │                           vaporisation rate. Validates R(t) = R₀ +
│   │                           (ṁ/ρᵥ)t against the analytical solution.
│   │                           Reproduces Roccon (2025) Section 3.4.
│   │
│   └── stefan_1d.py            1-D Stefan problem: superheated vapour drives
│                               vaporisation. Validates δ(t) = 2ξ√(αᵥt)
│                               against the analytical solution.
│                               Reproduces Roccon (2025) Section 3.2.
│
├── requirements.txt            Python dependencies
└── README.md                   This file
```

---

## Quick start

```bash
cd boiling-phasefield-3d
pip install -r requirements.txt

# Benchmark 1: 2-D bubble growth (runs in ~30 s on a laptop)
python examples/bubble_2d.py
# Output: bubble_2d_result.png (R(t) numerical vs analytical + final φ field)

# Benchmark 2: 1-D Stefan problem (runs in ~5 min on a laptop)
python examples/stefan_1d.py
# Output: stefan_1d_result.png (δ(t) numerical vs analytical + final T, φ profiles)
```

---

## Extending to 3-D

The `run_3d` function and all `*_3d` operator variants are already implemented.
The 3-D extension is the main Year-1 PhD task:

```python
from src import SimParams, run_3d
import numpy as np

p = SimParams(ndim=3, Nx=32, Ny=32, Nz=32,
              Lx=0.01, Ly=0.01, Lz=0.01,
              dt=5e-6, t_end=0.001,
              mode='prescribed', mdot_surf=0.1)

# Spherical bubble at domain centre
x = np.linspace(0, p.Lx, p.Nx, endpoint=False)
y = np.linspace(0, p.Ly, p.Ny, endpoint=False)
z = np.linspace(0, p.Lz, p.Nz, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')   # shape (Nx, Ny, Nz)
r    = np.sqrt((X - p.Lx/2)**2 + (Y - p.Ly/2)**2 + (Z - p.Lz/2)**2)
phi0 = 0.5 * (1 - np.tanh((r - 0.001) / (2 * p.eps)))

result = run_3d(p, phi0)
```

---

## PhD roadmap

| Year | Task | Status in this repo |
|---|---|---|
| 1 | 3-D spherical bubble benchmark | `run_3d` implemented, needs testing |
| 1 | Full probe-based vaporisation rate (Eq. 12) | Simplified version in `energy.py`; full probe method is TODO |
| 1 | Wall + outlet boundary conditions | Currently periodic only; wall BCs are TODO |
| 2 | GPU port (Fortran + NVIDIA HPC-SDK) | FLOW36 fork, separate production repo |
| 2 | DNS of nucleate boiling in turbulent channel | Requires HPC cluster |
| 3 | Parameter studies: Re, ΔT, density ratio | Post-processing scripts |
| 3 | Nusselt number scaling laws | `diagnostics.py::nusselt_number_2d` stub ready |

---

## Known limitations of this prototype

1. **Surface tension instability with large density ratios.**
   The explicit CSF model requires time step Δt ~ √(ρᵥ dx³/σ) ~ 10⁻⁹ s for
   water (ρᵥ/ρₗ = 0.001, σ = 0.07 N/m). This is impractical for explicit
   Euler. The bubble benchmark therefore uses σ = 0. Implicit surface tension
   treatment is a Year-2 implementation task.

2. **Simplified vaporisation rate (no probe interpolation).**
   The heat-flux vaporisation rate in `energy.py` evaluates temperature
   gradients at grid-point locations rather than at probe points ±Δ from the
   interface iso-contour (Roccon 2025, Section 2.4). This introduces growing
   error in the Stefan problem at late times. The full probe method is a
   Year-1 implementation task.

3. **Periodic boundary conditions only.**
   The pressure FFT solver assumes periodicity on all boundaries. Wall-bounded
   turbulence (the main PhD simulation) requires Dirichlet/Neumann BCs, which
   need a different solver (e.g., sine/cosine transform or tridiagonal solver).

4. **Explicit time stepping.**
   All equations use explicit Euler, which is first-order accurate in time.
   The paper uses the same scheme, but higher-order RK methods will be needed
   for turbulent DNS accuracy.

5. **Python performance.**
   NumPy is sufficient for 2-D prototype runs (64×64 in seconds). A 3-D
   turbulent simulation at 512³ requires Fortran + GPU: this prototype is
   a validation tool only, not a production solver.

---

## Connection to Roccon's existing codebase

Roccon's public repositories at [MultiphaseFlowLab](https://github.com/MultiphaseFlowLab):

| Repo | Language | What it does | Phase-change? |
|---|---|---|---|
| FLOW36 | Fortran + GPU | 3-D DNS of drop-laden turbulence (NS + Allen-Cahn) | **No** |
| MHIT36 | Fortran + GPU | Multi-GPU turbulence DNS, scales to 4096³ | **No** |

Neither existing code includes boiling physics. The PhD project bridges this
gap by adding the energy equation and vaporisation source term from Roccon
(2025) into the FLOW36 framework. This Python prototype validates those
additions before the Fortran port.

---

## References

- Roccon A. (2025). Boiling heat transfer by phase-field method. *Acta Mech.*
  236, 5623–5638.
- Mirjalili S., Ivey C.B., Mani A. (2020). A conservative diffuse interface
  method for two-phase flows. *J. Comput. Phys.* 401, 109006.
- Roccon A., Zonta F., Soldati A. (2023). Phase-field modeling of complex
  interface dynamics in drop-laden turbulence. *Phys. Rev. Fluids* 8, 090501.
