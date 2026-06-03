"""Simulation parameters — physical and numerical."""
from dataclasses import dataclass


@dataclass
class SimParams:
    """All parameters for the boiling phase-field solver.

    Physical defaults correspond to water at saturation (standard pressure),
    matching Roccon (2025) Section 3.4 bubble-growth benchmark.
    """

    # ── Geometry ──────────────────────────────────────────────────────────────
    ndim: int   = 2       # 2 = 2-D simulation  |  3 = 3-D simulation
    Nx:   int   = 64
    Ny:   int   = 64
    Nz:   int   = 1       # ignored when ndim == 2
    Lx:   float = 0.01    # domain length [m]
    Ly:   float = 0.01
    Lz:   float = 0.01

    # ── Time integration ──────────────────────────────────────────────────────
    dt:         float = 5e-6   # time step [s]
    t_end:      float = 0.01   # end time [s]
    save_every: int   = 200    # save diagnostics every N steps

    # ── Fluid properties ──────────────────────────────────────────────────────
    rho_l: float = 1000.0   # liquid density [kg/m³]
    rho_v: float = 1.0      # vapour density [kg/m³]
    mu_l:  float = 1e-3     # liquid dynamic viscosity [Pa·s]
    mu_v:  float = 1e-3     # vapour dynamic viscosity [Pa·s]
    sigma: float = 0.07     # surface tension [N/m]

    # ── Thermal properties ────────────────────────────────────────────────────
    k_l:   float = 0.677    # liquid thermal conductivity [W/(m·K)]
    k_v:   float = 0.024    # vapour thermal conductivity [W/(m·K)]
    Cp_l:  float = 4216.0   # liquid specific heat [J/(kg·K)]
    Cp_v:  float = 2077.0   # vapour specific heat [J/(kg·K)]
    h_lv:  float = 2.25e6   # latent heat of vaporisation [J/kg]
    T_sat: float = 373.15   # saturation temperature [K]
    T_wall: float = 383.15  # heated-wall temperature [K]  (ΔT = 10 K)

    # ── Phase-field ───────────────────────────────────────────────────────────
    eps:   float = None   # interface width [m]  (default: 1.5 × dx)
    gamma: float = None   # Allen-Cahn mobility  (default: eps)

    # ── Vaporisation mode ─────────────────────────────────────────────────────
    # 'prescribed' : ṁ (per unit surface) is constant — used for bubble benchmark
    # 'heat_flux'  : ṁ computed from temperature gradient at interface (Eq. 12)
    mode:       str   = 'prescribed'
    mdot_surf:  float = 0.1   # [kg/(m²·s)]  — only used when mode='prescribed'

    def __post_init__(self):
        if self.eps is None:
            self.eps = 1.5 * self.dx
        if self.gamma is None:
            self.gamma = self.eps

    # ── Derived grid spacing ──────────────────────────────────────────────────
    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def dz(self) -> float:
        return self.Lz / max(self.Nz, 1)
