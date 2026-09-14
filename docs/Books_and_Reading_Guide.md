---
title: "Books and Reading Guide"
subtitle: "A Structured Study Roadmap for Phase-Field Modelling of Boiling Heat Transfer in Turbulent Flows"
author: "Adebanji Oluwatimileyin Adelowo"
date: "2026"
toc: true
toc-depth: 2
number-sections: true
geometry: margin=1in
fontsize: 11pt
linkcolor: blue
urlcolor: blue
monofont: "Menlo"
monofontoptions:
  - Scale=0.85
---

\newpage

# How to Use This Guide

## Purpose

This is a study roadmap, not a bibliography. Its companion document —
*A Phase-Field Solver for Boiling Heat Transfer: Governing Equations, Numerical Methods,
and a Python Prototype Toward 3D DNS* — cites 41 papers and derives roughly a hundred
equations. Those papers assume a body of background that they do not supply. This guide
identifies the books that supply it, says exactly which chapters matter and why, and puts
them in an order that builds rather than jumps.

Every entry answers the same six questions:

1. **What is it** — full bibliographic detail, verified edition and ISBN.
2. **Why it matters *here*** — the specific connection to this project, not a generic
   endorsement.
3. **Which sections of the technical document it supports** — a direct cross-reference.
4. **What to actually read** — named chapters, not "the whole book".
5. **Level and classification** — foundational, supplementary, or advanced.
6. **Relationship to the cited papers** — especially Roccon (2025), Mirjalili et al.
   (2020), and Jacqmin (1999).

Twenty-one books are included. Titles that are merely adjacent — general dispersed
multiphase flow, general thermodynamics, general programming — have been deliberately left
out. Everything here connects to something the project actually does.

## The three classifications

**Foundational** — material the project's mathematics rests on directly. If a derivation in
the technical document is unclear, the answer is in one of these. Ten books.

**Supplementary** — deepens or broadens a topic that is used but not developed from scratch.
Read selectively, driven by need. Seven books.

**Advanced** — needed for the later phases of the PhD (turbulence, spectral wall-bounded
solvers, GPU porting) rather than for understanding the current prototype. Four books.

## An honest caveat about coverage

No book covers this project's actual subject. The specific combination — conservative
Allen–Cahn phase field, coupled to one-fluid Navier–Stokes with a phase-change mass source,
closed by a Rankine–Hugoniot vaporisation rate — exists only in the journal literature, and
principally in Roccon (2025) itself. Section 9 of this guide states precisely where the
books stop and the papers must take over. The books are here to make those papers readable,
not to replace them.

A second caveat on the phase-field literature specifically. The standard texts (Provatas &
Elder, Emmerich) come from **materials science**, where phase-field methods were invented to
model solidification and microstructure. Their $\phi$ is an order parameter for a
crystallographic phase, not a volume fraction, and their equations are usually written on
$[-1,+1]$ rather than $[0,1]$. The *mathematics* — free-energy functionals, variational
derivatives, double-well potentials, sharp-interface asymptotics — is identical and is
exactly what Section 4 of the technical document needs. The *application* is not. Read them
for the theory, and translate.

\newpage

# Reading Roadmap

The ordering below is by dependency, not by topic. Each phase assumes the previous one.

| Phase | Goal | Books | Weeks |
|---|---|---|---|
| 0 | Fluid mechanics on a rigorous footing; the projection method at its source | Chorin & Marsden; Bird et al. (skim) | 4–6 |
| 1 | Heat transfer and the physics of boiling | Incropera et al.; Carey | 5–7 |
| 2 | PDE theory and finite-difference analysis | Evans (selective); LeVeque; Strikwerda | 6–8 |
| 3 | CFD practice: discretisation, pressure–velocity coupling | Ferziger, Perić & Street; Anderson; Moin | 6–8 |
| 4 | The FFT Poisson solver and spectral methods | Trefethen (*Spectral Methods*); Trefethen & Bau | 3–4 |
| 5 | Interface methods: VOF, level set, front tracking, phase field | Prosperetti & Tryggvason; Tryggvason, Scardovelli & Zaleski | 6–8 |
| 6 | Phase-field theory from the free energy down | Provatas & Elder; Emmerich (selective) | 5–7 |
| 7 | Multiphase flow context | Brennen | 2–3 |
| 8 | Turbulence and DNS | Pope | 8–10 |
| 9 | Wall-bounded spectral solvers (the FLOW36 architecture) | Peyret; Boyd | 4–6 |
| 10 | GPU porting | Kirk, Hwu & El Hajj | 4–6 |

Phases 0–6 cover everything needed to fully understand the current prototype and the
Roccon (2025) method. Phases 7–10 are the PhD's later years: turbulence, the wall-bounded
spectral formulation, and the Fortran/CUDA port.

**If time is short.** The four books that pay back fastest, in order: LeVeque (Chapters
9–10) for why the Stefan benchmark oscillates; Provatas & Elder (Chapters 2–5) for the
free-energy derivations behind Section 4 of the technical document; Prosperetti &
Tryggvason (Chapters 2–4) for the interface-method landscape; and Ferziger, Perić & Street
(Chapters 7–8) for pressure–velocity coupling and the projection method.

\newpage

# Part I — Fluid Mechanics and Transport Foundations

## Chorin & Marsden, *A Mathematical Introduction to Fluid Mechanics*

| | |
|---|---|
| **Authors** | Alexandre J. Chorin, Jerrold E. Marsden |
| **Edition** | 3rd edition |
| **Publisher** | Springer-Verlag, New York (Texts in Applied Mathematics, vol. 4) |
| **Year** | 1993 |
| **ISBN** | 978-0-387-97918-2 |
| **DOI** | 10.1007/978-1-4612-0883-9 |
| **Level** | Advanced undergraduate / early graduate |
| **Classification** | **Foundational** |

**Why it matters here.** Chorin is the author of the projection method — Chorin (1968),
*Numerical solution of the Navier–Stokes equations*, which is reference [59] in Roccon
(2025) and the direct ancestor of Eqs. (R.16)–(R.20), the time-stepping scheme implemented
in `src/flow.py::ns_step_2d`. This book is where he sets out the mathematical framework the
method sits in. It is short, dense, and unusually clear about *why* incompressible flow
requires a pressure Poisson solve at all: pressure is not a thermodynamic variable but a
Lagrange multiplier enforcing the divergence constraint. Once that is understood, the
structure of Section 8.2 of the technical document — and the reason the constant-coefficient
property is such a prize — stops being a trick and becomes inevitable.

**Supports.** Technical document Sections 2.1 (sharp-interface Navier–Stokes), 8.2 (why the
one-fluid formulation gives a constant-coefficient Poisson equation), 8.4 (the
projection–correction scheme).

**What to read.**

- Chapter 1, §1.1–1.3 — derivation of the Euler and Navier–Stokes equations from
  conservation principles. The cleanest short derivation available.
- Chapter 1, §1.3 — the role of pressure and the Hodge/Helmholtz decomposition. **This is
  the single most important section in the book for this project.** The projection method is
  a discrete Hodge decomposition: split the intermediate momentum field into a
  divergence-constrained part and a gradient, and discard the gradient. The correction
  step — Eqs. (R.17) and (R.19) in the technical document — is exactly this.
- Chapter 3, §3.1–3.2 — the numerical treatment, including Chorin's own account of the
  projection method.

**Connection to the implementation.** The pair

```python
pres   = solve_poisson_2d(rhs_p, dx, dy)          # project
wx_new = wx_star - dt * grad_x_2d(pres, dx)       # subtract the gradient part
```

in `src/flow.py` is a Hodge decomposition performed with FFTs. Chapter 1 §1.3 explains why
subtracting a gradient is guaranteed to produce a field satisfying the constraint, and why
the pressure that does it is unique up to a constant — which is exactly the null-space
handling (`p_hat[0,0,0] = 0.0`) in `src/pressure.py`.

**Relationship to the cited papers.** Direct: Chorin (1968) is cited in the technical
document as the origin of the projection method, and Roccon (2025) §2.5 cites it as
reference [59] for the "first-order projection-correction method" used there.

## Bird, Stewart & Lightfoot, *Transport Phenomena*

| | |
|---|---|
| **Authors** | R. Byron Bird, Warren E. Stewart, Edwin N. Lightfoot |
| **Edition** | Revised 2nd edition |
| **Publisher** | John Wiley & Sons, Hoboken, NJ |
| **Year** | 2007 |
| **ISBN** | 978-0-470-11539-8 |
| **Level** | Undergraduate / graduate |
| **Classification** | **Foundational** (reference rather than cover-to-cover) |

**Why it matters here.** The canonical unified treatment of momentum, energy, and mass
transport. Its value for this project is specific and easy to state: it is the clearest
source for the **interphase transport balances** and for the general structure of a
conservation equation with a source term, which is the form every one of the four governing
equations in Section 7 of the technical document takes. It is also the standard reference
for the property data used throughout (thermal conductivity, specific heat, viscosity of
water and steam at saturation).

**Supports.** Technical document Sections 2.1 (the sharp-interface equations), 7.2 (mass
conservation with a phase-change source), 7.5 (energy equation).

**What to read.** Use it as a reference, not a text. The relevant parts:

- Chapters 1–3 — momentum transport: viscosity, the momentum flux tensor, and the
  Navier–Stokes equations. Chapter 3's treatment of the equations of change is the reference
  for the conservative-versus-non-conservative form distinction that Defect 2 (technical
  document Section 9.5) turns on.
- Chapters 9–11 — energy transport: thermal conductivity, the energy equation, and its
  derivation including the terms Roccon (2025) neglects (viscous dissipation, pressure
  work). Chapter 11 is the source for understanding *why* those terms are negligible here.
- Chapter 11, §11.4 and the interphase-transport sections — boundary conditions at phase
  boundaries.
- Appendix A — the equations of change in all coordinate systems, and Appendix E for
  property tables.

**Connection to the implementation.** The distinction between $\nabla\cdot(\rho\mathbf{uu})$
and $\rho(\mathbf{u}\cdot\nabla)\mathbf{u}$, which Chapter 3 develops carefully, is exactly
the distinction that `src/flow.py` gets wrong: the code comment claims the two are equal,
which holds only when $\nabla\cdot(\rho\mathbf{u}) = 0$ — precisely what phase change
violates.

**Relationship to the cited papers.** Background rather than direct. Roccon (2025) Eq. (9)
is a standard energy equation with the dissipation and pressure-work terms dropped; Bird et
al. Chapter 11 is where that starting equation and the justification for dropping them live.

\newpage

# Part II — Heat Transfer and Boiling Physics

## Incropera, DeWitt, Bergman & Lavine, *Fundamentals of Heat and Mass Transfer*

| | |
|---|---|
| **Authors** | Theodore L. Bergman, Adrienne S. Lavine, Frank P. Incropera, David P. DeWitt |
| **Edition** | 8th edition |
| **Publisher** | John Wiley & Sons, Hoboken, NJ |
| **Year** | 2017 |
| **ISBN** | 978-1-119-35388-1 |
| **Level** | Undergraduate |
| **Classification** | **Foundational** |

**Why it matters here.** This is where the vocabulary of the project's *engineering* context
is defined: Nusselt number, Stefan number, Jakob number, the boiling curve, critical heat
flux, and the empirical correlations that the technical document's introduction argues are
inadequate. It is impossible to explain why DNS of boiling is worth doing without first
knowing precisely what the correlations claim and where they come from. Chapter 10 is a
compact, authoritative account of the boiling curve and the nucleate/transition/film regimes.

**Supports.** Technical document Sections 1.1 (why boiling matters, CHF), 10.2 (the Stefan
number and the Stefan problem), 13.6 (Nusselt-number scaling laws, the target of the
eventual parameter study).

**What to read.**

- Chapter 2 — conduction: Fourier's law and the heat equation. The source of $\nabla\cdot(k\nabla T)$
  and of the mixture-diffusivity form implemented in `src/energy.py::alpha_mix`.
- Chapter 5 — transient conduction, including the semi-infinite solid with an
  error-function solution. **Read this before the Stefan problem**: it is the same
  similarity solution, and Eq. (R.27) in the technical document is its direct descendant.
- Chapter 6 — convection fundamentals and the definition of the Nusselt number, which
  `src/diagnostics.py::nusselt_number_2d` computes.
- **Chapter 10 — boiling and condensation.** The core chapter. §10.1–10.4 give the boiling
  curve, nucleate boiling, critical heat flux, and film boiling; §10.5 gives the Rohsenow
  and Zuber correlations that DNS aims to replace or explain.

**Connection to the implementation.** The Stefan number $\mathrm{St} = C_{p,v}\Delta T/h_{lv}$
computed in `examples/stefan_1d.py` and the Nusselt number in `src/diagnostics.py` are both
defined here. Chapter 5's error-function solutions are the analytical machinery of the
Stefan benchmark.

**Relationship to the cited papers.** Provides the engineering framing for Kharangate &
Mudawar (2017), the boiling/condensation computational review cited in the technical
document's Section 5.2.

## Carey, *Liquid–Vapor Phase-Change Phenomena*

| | |
|---|---|
| **Author** | Van P. Carey |
| **Subtitle** | *An Introduction to the Thermophysics of Vaporization and Condensation Processes in Heat Transfer Equipment* |
| **Edition** | 3rd edition |
| **Publisher** | CRC Press / Taylor & Francis, Boca Raton, FL |
| **Year** | 2020 |
| **ISBN** | 978-1-4987-1661-1 |
| **Level** | Graduate |
| **Classification** | **Foundational** — the single most important physics book for this project |

**Why it matters here.** Carey is cited *by Roccon (2025)* — reference [52] — as the source
for kinetic-theory models of interfacial mass transfer. It is the definitive graduate text
on the thermophysics of vaporisation and condensation, and it is the book that supplies the
physical content the numerical papers assume without deriving. In particular, it is the
proper source for the two competing vaporisation-rate closures that Section 7.6 of the
technical document contrasts: the energy/heat-conduction model (Eq. R.12) that this project
implements, and the kinetic model (Eq. R.13, Tanasawa) that it does not.

**Supports.** Technical document Sections 2.1 (interfacial jump conditions), 7.6
(Rankine–Hugoniot balance and the kinetic alternative), 10.2 (Stefan problem physics), 12.1
(why the vaporisation-rate closure is the highest-priority fix).

**What to read.**

- Chapter 2 — interfacial thermodynamics: surface tension from a molecular viewpoint, the
  Young–Laplace equation, and the Kelvin equation. Background for Section 4.1 of the
  technical document, where surface tension emerges from the Ginzburg–Landau free energy.
- **Chapter 4 — transport at interfaces and the kinetic theory of interfacial mass
  transfer.** This is the derivation behind Schrage's and Tanasawa's models, i.e. Eq. (R.13).
  Essential for understanding *why* Roccon chose the energy-based model instead.
- Chapter 5 — the Rankine–Hugoniot / interfacial energy balance in its physical setting.
- Chapters 7–9 — pool boiling: nucleation, bubble growth, the boiling curve, and critical
  heat flux. Chapter 7's treatment of bubble growth is the physical counterpart to the
  `examples/bubble_2d.py` benchmark.
- Chapter 11 — external and internal flow boiling; the closest the book comes to the PhD's
  eventual turbulent-channel target.

**Connection to the implementation.** `src/energy.py::mdot_from_heatflux_2d` implements the
energy-based closure; Carey Chapter 4 is where the alternative kinetic closure is derived
and where the assumptions of each are laid out. If the probe method (technical document
Section 13.1, priority 1) proves difficult to parallelise, a kinetic closure is the fallback,
and this chapter is where that decision would be made.

**Relationship to the cited papers.** Directly cited by Roccon (2025) as [52]. Also the
background for Schrage (1953) and Tanasawa (1991), references [51] and [53] in that paper.

\newpage

# Part III — PDE Theory and Finite-Difference Analysis

## Evans, *Partial Differential Equations*

| | |
|---|---|
| **Author** | Lawrence C. Evans |
| **Edition** | 2nd edition |
| **Publisher** | American Mathematical Society (Graduate Studies in Mathematics, vol. 19) |
| **Year** | 2010 |
| **ISBN** | 978-0-8218-4974-3 |
| **Level** | Graduate (mathematics) |
| **Classification** | **Supplementary** — read selectively |

**Why it matters here.** This is a pure-mathematics PDE text and most of it is not needed.
Two things in it are. First, the treatment of **second-order parabolic equations** — the
maximum principle, and the notion of well-posedness — is what makes the boundedness claim of
Mirjalili et al. (2020) meaningful: "$\phi$ remains bounded in $[0,1]$" is a discrete
maximum principle, and knowing what the continuous version says is what tells you whether
the discrete one is plausible. Second, the **calculus of variations** chapter is the rigorous
version of the variational derivative $\delta\mathcal{F}/\delta\phi$ that Section 4.2 of the
technical document computes informally.

**Supports.** Technical document Sections 4.2 (chemical potential as a variational
derivative), 4.5 (the boundedness criterion), 12.4 (oscillation control).

**What to read.** Only these:

- Chapter 2, §2.3 — the heat equation: fundamental solution, mean-value property, and the
  maximum principle.
- Chapter 7, §7.1 — second-order parabolic equations, existence and uniqueness.
- **Chapter 8, §8.1–8.2 — the calculus of variations**: Euler–Lagrange equations and the
  first variation. This is the formal justification for the derivation in Section 4.2 of the
  technical document, and for the Euler–Lagrange equation $\lambda\phi'' = f_0'(\phi)$ whose
  first integral gives the equilibrium tanh profile.

Skip the rest unless a specific need arises. This book is a reference, not a course, for
this project.

**Relationship to the cited papers.** Underpins the boundedness argument of Mirjalili, Ivey
& Mani (2020) and the free-energy formulation of Cahn & Hilliard (1958).

## LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*

| | |
|---|---|
| **Author** | Randall J. LeVeque |
| **Subtitle** | *Steady-State and Time-Dependent Problems* |
| **Publisher** | SIAM (Society for Industrial and Applied Mathematics), Philadelphia |
| **Year** | 2007 |
| **ISBN** | 978-0-89871-629-0 |
| **Level** | Graduate |
| **Classification** | **Foundational** — the highest-value book in this guide for debugging |

**Why it matters here.** Every spatial operator in `src/operators.py` is a second-order
central difference, and every time integrator in the repository is explicit Euler. LeVeque
is the book that explains what that choice costs: consistency, stability, convergence, the
modified-equation analysis, and — critically for this project — **why central differences
have no dissipation at the grid scale**. Section 11.2 of the technical document diagnoses the
Stefan-problem breakdown as a $2\Delta x$ odd–even oscillation that nothing in the scheme
damps. LeVeque Chapters 9–10 are where that failure mode is named and analysed.

**Supports.** Technical document Sections 8.1 (spatial discretisation), 8.5 (stability of
the explicit scheme), 11.2 (the Stefan-problem failure analysis), 12.4 (no oscillation
control), 13.4 (higher-order time integration).

**What to read.**

- Chapter 1 — finite-difference approximations and truncation error. The formal basis for
  the $\mathcal{O}(h^2)$ claims about `_d1` and `_d2` in `src/operators.py`.
- Chapter 2 — boundary value problems and the discrete Laplacian, including its
  eigenvalues. **Directly relevant**: the eigenvalue formula
  $\lambda_k = (2\cos(2\pi k/N) - 2)/h^2$ implemented in
  `src/pressure.py::_eig_laplacian_1d` is derived here.
- Chapter 5 — the method of lines; how a spatial discretisation plus a time integrator
  combine, and where the stability restriction comes from.
- Chapter 7 — absolute stability regions for explicit Euler and Runge–Kutta methods. The
  basis for the roadmap item on RK3 (technical document Section 13.4).
- Chapter 8 — stiff ODEs; why the viscous term is the binding constraint (technical
  document Section 8.5) and what implicit treatment would buy.
- **Chapters 9–10 — diffusion and advection equations.** Chapter 10 in particular covers
  numerical dissipation, dispersion, modified equations, and the behaviour of central
  differences at the Nyquist mode. **This is the direct explanation of the Stefan-problem
  breakdown.**

**Connection to the implementation.** LeVeque Chapter 10's modified-equation analysis shows
that the leading error of a central difference is dispersive, not diffusive — the reason
`src/operators.py` is a good choice for DNS (no artificial dissipation to contaminate the
turbulent cascade) and simultaneously the reason the Stefan benchmark disintegrates once the
saturation clamp excites the $2\Delta x$ mode.

**Relationship to the cited papers.** Provides the analytical framework behind Mirjalili,
Ivey & Mani (2020), whose entire contribution is a boundedness proof for *central*
differencing, chosen precisely for its non-dissipative character.

## Strikwerda, *Finite Difference Schemes and Partial Differential Equations*

| | |
|---|---|
| **Author** | John C. Strikwerda |
| **Edition** | 2nd edition |
| **Publisher** | SIAM, Philadelphia |
| **Year** | 2004 |
| **ISBN** | 978-0-89871-639-9 |
| **Level** | Graduate |
| **Classification** | **Supplementary** |

**Why it matters here.** Overlaps LeVeque but is stronger on one thing this project needs:
**von Neumann stability analysis**, developed systematically and applied to a wide range of
schemes. The technical document's Section 8.5 states four stability restrictions —
convective, viscous, phase-field sharpening, and surface tension — and each is a von Neumann
result. Strikwerda is where the method is set out carefully enough to derive them rather
than quote them.

**Supports.** Technical document Sections 8.5 (stability restrictions), 12.5 (first-order
explicit time integration).

**What to read.**

- Chapters 2–3 — consistency, stability, convergence, and the Lax–Richtmyer equivalence
  theorem.
- **Chapter 4 — the von Neumann analysis in full.** The tool for deriving every restriction
  in Section 8.5.
- Chapter 6 — parabolic PDEs and their stability restrictions; the $\Delta t \sim \Delta x^2$
  viscous limit.
- Chapter 9 — elliptic PDEs and their solution, including transform methods; useful
  background for the FFT Poisson solver.
- Chapter 13 — the incompressible Navier–Stokes equations, including projection methods and
  the pressure Poisson problem.

**Connection to the implementation.** The claim that the bubble benchmark runs at 82% of the
viscous stability limit — and that the $N = 128$ convergence run needed
$\Delta t = 10^{-6}$ s to stay inside it (technical document Section 11.1) — is a von
Neumann result of exactly the kind Chapter 4 teaches.

## Moin, *Fundamentals of Engineering Numerical Analysis*

| | |
|---|---|
| **Author** | Parviz Moin |
| **Edition** | 2nd edition |
| **Publisher** | Cambridge University Press |
| **Year** | 2010 |
| **ISBN** | 978-0-521-71123-4 |
| **Level** | Graduate (engineering) |
| **Classification** | **Supplementary** — the fastest route in |

**Why it matters here.** Short, practical, and written by a leading DNS practitioner. Where
LeVeque and Strikwerda are thorough, Moin is fast: it covers interpolation, numerical
integration, ODE integration, finite differences, and — unusually for a book this size —
**discrete Fourier transforms and their use in solving PDEs**, which is precisely the content
of `src/pressure.py`. If the mathematics in Part III feels heavy, start here and return to
LeVeque afterwards.

**Supports.** Technical document Sections 8.1 (spatial discretisation), 8.3 (FFT Poisson
solve), 8.5 (stability).

**What to read.**

- Chapter 2 — numerical solution of ODEs: explicit Euler, Runge–Kutta, and stability.
  §2.4's treatment of RK methods is the concise reference for the RK3 upgrade in technical
  document Section 13.4.
- Chapter 5 — numerical solution of PDEs, including the finite-difference discretisation of
  the heat and advection equations.
- **Chapter 6 — discrete Fourier transforms.** Covers the DFT, aliasing, and — directly —
  the use of transforms to solve PDEs. The clearest short account of the technique
  `src/pressure.py` uses.

**Relationship to the cited papers.** Moin's own DNS work with Jain and others (Jain, Mani &
Moin 2020, cited in the technical document) sits in the same computational tradition;
Chapter 6 explains the transform machinery that FLOW36 and MHIT36 are built on.

\newpage

# Part IV — Computational Fluid Dynamics

## Ferziger, Perić & Street, *Computational Methods for Fluid Dynamics*

| | |
|---|---|
| **Authors** | Joel H. Ferziger, Milovan Perić, Robert L. Street |
| **Edition** | 4th edition |
| **Publisher** | Springer, Cham |
| **Year** | 2020 |
| **ISBN** | 978-3-319-99691-2 |
| **DOI** | 10.1007/978-3-319-99693-6 |
| **Level** | Graduate |
| **Classification** | **Foundational** — the main CFD reference for this project |

**Why it matters here.** The standard graduate CFD text, and the best single source for the
two things this project's flow solver depends on: **pressure–velocity coupling** and
**staggered versus collocated grids**. Technical document Section 8.1 notes that Roccon
(2025) uses a staggered arrangement (velocities at faces, scalars at centres) whereas this
prototype is collocated, and Section 12.7 flags that divergence as a structural defect
implicated in the Stefan failure. Ferziger, Perić & Street Chapter 7 is where the
odd–even pressure–velocity decoupling of collocated grids is explained and where the
remedies are set out.

**Supports.** Technical document Sections 8.1 (spatial discretisation, staggered vs
collocated), 8.2 (projection method for variable density), 8.4 (time stepping), 12.7
(collocated grid limitation), 13.3 (wall and outlet boundary conditions).

**What to read.**

- Chapters 2–3 — the conservation equations and an overview of numerical methods.
- Chapter 4 — finite-difference and finite-volume discretisation.
- **Chapter 7 — solution of the Navier–Stokes equations.** The central chapter. Covers the
  pressure Poisson equation, the SIMPLE family, fractional-step/projection methods, and
  §7.2's treatment of staggered versus collocated variable arrangements. Read this before
  attempting the wall-boundary-condition work in technical document Section 13.3.
- Chapter 8 — complex geometries and boundary condition treatment.
- Chapter 9 — turbulent flows: DNS, LES, RANS, and the resolution requirements that make
  the technical document's $\mathcal{O}(10^8)$ grid-point estimate concrete.
- Chapter 12 — efficiency and parallelisation; domain decomposition, relevant to the
  eventual MPI port.

**Connection to the implementation.** `src/flow.py::ns_step_2d` is a fractional-step method
of the type Chapter 7 describes; the choice to solve for pressure from the divergence of the
intermediate field, and the requirement that the correction use $\rho^{n+1}$, are both
discussed there.

## Anderson, *Computational Fluid Dynamics: The Basics with Applications*

| | |
|---|---|
| **Author** | John D. Anderson, Jr. |
| **Publisher** | McGraw-Hill, New York |
| **Year** | 1995 |
| **ISBN** | 978-0-07-001685-9 |
| **Level** | Advanced undergraduate / early graduate |
| **Classification** | **Supplementary** — an on-ramp, not a destination |

**Why it matters here.** Anderson is the gentlest serious introduction to CFD in print. Its
value for this project is pedagogical: it derives the governing equations in every form
(conservation/non-conservation, differential/integral) side by side and is explicit about
what distinguishes them. That distinction is exactly what Defect 2 in the technical document
(Section 9.5) turns on — the momentum advection in `src/flow.py` uses the non-conservative
form while the docstring claims the conservative one, and the two differ by
$\mathbf{u}\,\nabla\cdot(\rho\mathbf{u})$, which phase change makes non-zero.

**Supports.** Technical document Sections 2.1, 7.3 (one-fluid Navier–Stokes), 9.5 (Defect 2).

**What to read.**

- **Chapter 2 — the governing equations of fluid dynamics.** Especially §2.5–2.10, which
  present conservation and non-conservation forms in parallel and explain when they coincide.
- Chapter 4 — basic aspects of discretisation.
- Chapter 6 — some simple CFD techniques, including the Lax and MacCormack schemes; useful
  for building intuition about numerical dissipation before reading LeVeque Chapter 10.

Anderson's compressible-flow emphasis limits its usefulness beyond this; treat it as an
entry point and move to Ferziger, Perić & Street.

## Pope, *Turbulent Flows*

| | |
|---|---|
| **Author** | Stephen B. Pope |
| **Publisher** | Cambridge University Press |
| **Year** | 2000 |
| **ISBN** | 978-0-521-59886-6 (paperback) |
| **Level** | Graduate |
| **Classification** | **Advanced** — needed for PhD Years 2–3 |

**Why it matters here.** The PhD's target simulation is nucleate boiling in a **wall-bounded
turbulent channel**. Nothing in the current prototype is turbulent, and nothing in the other
books on this list covers turbulence properly. Pope is the standard graduate text: the energy
cascade, Kolmogorov scaling, the resolution requirements of DNS, and — in Chapter 7 — the
structure of wall-bounded turbulent flow, including the friction Reynolds number that sets
the grid requirement quoted in technical document Section 1.2.

**Supports.** Technical document Sections 1.2 (why DNS, and its cost), 12.5 (why first-order
time integration is inadequate for turbulence), 13.6 (nucleate boiling in a turbulent
channel; Nusselt-number scaling).

**What to read.**

- Chapters 1–3 — the statistical description of turbulence; means, fluctuations, and the
  Reynolds decomposition.
- **Chapter 6 — the scales of turbulent motion.** Kolmogorov's hypotheses, the energy
  cascade, and the $\mathrm{Re}^{9/4}$ scaling of DNS cost. This is the quantitative basis
  for the claim that a boiling DNS needs GPU-scale hardware.
- **Chapter 7 — wall flows.** Channel and boundary-layer turbulence, the law of the wall,
  and near-wall resolution requirements. Essential before designing the channel simulation.
- Chapter 9, §9.1 — direct numerical simulation: what it resolves, what it costs, and how
  it is validated.

**Connection to the project.** Chapter 6's cascade argument is why technical document
Section 12.5 insists that first-order explicit Euler must be replaced before turbulent runs:
temporal accuracy directly affects the resolved spectrum. Chapter 7's near-wall scaling is
why the periodic-boundary limitation (Section 12.3) is blocking rather than cosmetic.

**Relationship to the cited papers.** The turbulence background for Roccon, Zonta & Soldati
(2023), the drop-laden-turbulence paper that is the direct non-boiling ancestor of this
project.

\newpage

# Part V — Multiphase Flow and Interface Methods

## Prosperetti & Tryggvason (eds.), *Computational Methods for Multiphase Flow*

| | |
|---|---|
| **Editors** | Andrea Prosperetti, Gretar Tryggvason |
| **Publisher** | Cambridge University Press |
| **Year** | 2007 |
| **ISBN** | 978-0-521-84764-3 |
| **Level** | Graduate / research |
| **Classification** | **Foundational** |

**Why it matters here.** This edited volume is the single best map of the territory that
Section 3.4 and Section 5 of the technical document survey. Each chapter is written by a
leading practitioner of one method — front tracking, volume of fluid, level set, and
diffuse-interface/phase-field — which makes it the natural companion to the comparison table
in technical document Section 3.4. Reading the VOF and level-set chapters is what makes the
argument for choosing conservative Allen–Cahn convincing rather than merely asserted.

**Supports.** Technical document Sections 3.4 (survey of interface methods), 5.1–5.4
(classical approaches to boiling simulation), 6 (research evolution).

**What to read.**

- Chapter 1 — introduction and the classification of methods (interface tracking versus
  interface capturing), the same taxonomy Roccon (2025) opens with.
- **Chapter 2 — direct numerical simulations of finite Reynolds number flows**, covering
  the one-fluid formulation with singular interfacial source terms. This is the conceptual
  parent of Eq. (R.4) and of the CSF force in Eq. (R.7).
- **Chapter 3 — immersed boundary and front-tracking methods** (Tryggvason). The background
  for Juric & Tryggvason (1998) and for the probe method of technical document Section 7.6.
- **Chapter 4 — volume of fluid and level-set methods.** The strengths and weaknesses that
  Section 5.2 and 5.3 of the technical document summarise.
- Chapter 5 — diffuse-interface methods; the closest thing in book form to what this
  project does, though it predates the conservative Allen–Cahn formulation.

**Relationship to the cited papers.** Direct lineage. Chapter 3's authors are the authors of
Unverdi & Tryggvason (1992) and Juric & Tryggvason (1998); Chapter 4 covers the methods of
Hirt & Nichols (1981), Osher & Sethian (1988), Welch & Wilson (2000) and Son & Dhir (1998) —
all cited in the technical document.

## Tryggvason, Scardovelli & Zaleski, *Direct Numerical Simulations of Gas–Liquid Multiphase Flows*

| | |
|---|---|
| **Authors** | Grétar Tryggvason, Ruben Scardovelli, Stéphane Zaleski |
| **Publisher** | Cambridge University Press |
| **Year** | 2011 |
| **ISBN** | 978-0-521-78240-1 |
| **DOI** | 10.1017/CBO9780511975264 |
| **Level** | Graduate / research |
| **Classification** | **Foundational** |

**Why it matters here.** **This book is cited by Roccon (2025) as reference [18]** — it is
the reference he gives for interface-resolved simulation as a class. It is the most directly
relevant book in this guide to the technical document's Sections 2 and 8: it develops the
one-fluid formulation, the treatment of singular interfacial forces, the projection method
for variable-density flow, and the numerical treatment of surface tension, all in the
specific setting of gas–liquid DNS.

**Supports.** Technical document Sections 2.2 (what makes this hard), 7.3 (one-fluid
Navier–Stokes), 7.4 (CSF surface tension), 8.2 (constant-coefficient Poisson equation),
8.5 (stability, including the surface-tension time-step restriction).

**What to read.**

- Chapter 1 — introduction and the one-fluid formulation.
- Chapter 2 — the governing equations for multiphase flow with singular interfacial terms.
  The delta-function formulation that $|\nabla\phi|$ approximates in Eq. (R.2).
- **Chapter 3 — numerical solutions of the Navier–Stokes equations for multiphase flow.**
  The projection method with variable density; this is where the difference between
  correcting *velocity* (variable-coefficient Poisson) and correcting *momentum*
  (constant-coefficient Poisson) is laid out — the core argument of technical document
  Section 8.2.
- **Chapter 4 — advecting the interface**, covering marker methods and the general problem.
- Chapter 5 — surface tension: the CSF model, curvature estimation, and **spurious
  currents**. The reference for technical document Section 12.2's discussion of why
  $\sigma = 0$ in the bubble benchmark.
- Chapter 6 — VOF methods (Scardovelli and Zaleski's specialty).
- Chapter 9 — phase change and boiling, including the mass-transfer source terms. The most
  direct book-length treatment of the physics in Eq. (R.3) and Eq. (R.12).

**Connection to the implementation.** Chapter 5's discussion of curvature estimation from a
smoothed indicator function is exactly what `src/flow.py::_curvature_2d` does, and its
analysis of spurious currents explains why they do not vanish under time-step refinement — a
point technical document Section 12.2 makes in correcting the README's stated justification
for $\sigma = 0$.

**Relationship to the cited papers.** Cited in the technical document's own reference list,
and cited by Roccon (2025) as [18]. Its authors wrote Unverdi & Tryggvason (1992), Juric &
Tryggvason (1998), and Esmaeeli & Tryggvason (2004), all cited in the technical document.

## Brennen, *Fundamentals of Multiphase Flow*

| | |
|---|---|
| **Author** | Christopher E. Brennen |
| **Publisher** | Cambridge University Press |
| **Year** | 2005 |
| **ISBN** | 978-0-521-84804-6 |
| **Level** | Graduate |
| **Classification** | **Supplementary** |

**Why it matters here.** Brennen supplies the *physical* context that the numerical books
assume: flow regimes, the dynamics of a single bubble, and — in Chapter 15 — boiling
phenomena including nucleation and the boiling crisis. Its treatment of the Rayleigh–Plesset
equation and of bubble growth is the physics counterpart to the constant-rate bubble
benchmark in `examples/bubble_2d.py`, and it is the natural place to learn what happens when
surface tension and inertia are *not* switched off.

**Supports.** Technical document Sections 2.2 (large property ratios), 10.1 (bubble-growth
benchmark), 13.6 (film boiling, nucleate boiling).

**What to read.**

- Chapters 1–2 — single-phase and multiphase flow fundamentals; the notion of a flow regime.
- **Chapter 4 — bubble growth and collapse**, including the Rayleigh–Plesset equation and
  thermally-controlled growth. Directly relevant to the bubble benchmark, and the source of
  the $R \propto \sqrt{t}$ thermally-limited growth law that contrasts with the linear
  $R(t)$ of the prescribed-rate case.
- Chapter 6 — boiling and condensation basics.
- **Chapter 15 — boiling phenomena**: nucleation sites, the boiling curve, and the boiling
  crisis. Complements Carey with a fluid-dynamics rather than thermodynamics emphasis.

The book is freely available from the author's institutional page as well as in print, which
makes it convenient as a quick reference.

\newpage

# Part VI — Phase-Field Theory

## Provatas & Elder, *Phase-Field Methods in Materials Science and Engineering*

| | |
|---|---|
| **Authors** | Nikolas Provatas, Ken Elder |
| **Publisher** | Wiley-VCH, Weinheim |
| **Year** | 2010 |
| **ISBN** | 978-3-527-40747-7 |
| **Level** | Graduate |
| **Classification** | **Foundational** — the standard phase-field text |

**Why it matters here.** This is the book behind Section 4 of the technical document. It is
the standard reference for phase-field methodology, and it derives, carefully and from first
principles, every piece of machinery that section uses informally: the Ginzburg–Landau free
energy functional, the double-well potential, the variational derivative and chemical
potential, the distinction between conserved (Cahn–Hilliard) and non-conserved (Allen–Cahn)
dynamics, the equilibrium hyperbolic-tangent profile, the relation between the model
parameters and surface tension, and — most importantly — the **matched asymptotic expansions**
that establish the sharp-interface limit.

Read the caveat in Section 1.3 of this guide before starting: the applications are
solidification and microstructure, not two-phase flow, and the order parameter convention
differs. The mathematics transfers exactly; the physics does not.

**Supports.** Technical document Sections 3.1–3.3 (diffuse interface, order parameter,
interface thickness), **4.1–4.5 in their entirety** (free energy, chemical potential,
Cahn–Hilliard vs Allen–Cahn, sharp-interface limit, conservative Allen–Cahn).

**What to read.**

- Chapter 2 — mean-field theory of phase transformations; where the double-well potential
  of technical document Section 4.1 comes from physically rather than by fiat.
- **Chapter 3 — spatial variations and interfaces.** The Ginzburg–Landau functional, the
  equilibrium interface profile, and the calculation of surface tension as the excess free
  energy. This is the direct source for the derivation of $\varepsilon = \sqrt{\lambda/2\beta}$
  and $\sigma = \lambda/(6\varepsilon)$ in technical document Section 4.1.
- **Chapter 4 — phase-field dynamics.** Model A (non-conserved, i.e. Allen–Cahn) and Model B
  (conserved, i.e. Cahn–Hilliard), derived from the free energy by variational principles.
  This is Section 4.3 of the technical document, done rigorously.
- **Chapter 5 — sharp-interface limits of phase-field models.** The matched asymptotic
  expansion that Section 4.4 of the technical document sketches in outline. If only one
  chapter of this book is read, make it this one: it is what justifies the entire modelling
  approach, and it is the source of the $\mathcal{O}(\varepsilon)$ and
  $\mathcal{O}(\varepsilon^2)$ error scalings that the convergence study in technical
  document Section 11.1 measures.
- Chapter 6 — numerical implementation: explicit schemes, stability, and grid resolution of
  the interface. The origin of the $\varepsilon \gtrsim \Delta x$ requirement.
- Chapter 7 — advanced topics, including adaptive mesh refinement.

**Connection to the implementation.** The tanh initial condition in
`examples/bubble_2d.py`, the identity $|\nabla\phi| = \phi(1-\phi)/\varepsilon$ used by
`src/phase_field.py::mdot_volumetric`, and the factor of 6 in
`src/flow.py::surface_tension_2d` are all consequences of the equilibrium profile derived in
Chapter 3.

**Relationship to the cited papers.** The book-length treatment of Cahn & Hilliard (1958) and
Allen & Cahn (1979). It does **not** cover the conservative Allen–Cahn formulation of Chiu &
Lin (2011) or Mirjalili et al. (2020) — that is a fluid-dynamics development postdating it,
and for that the papers are required. Read Provatas & Elder first, then Mirjalili et al.
(2020) will read as a natural modification rather than an unmotivated one.

## Emmerich, *The Diffuse Interface Approach in Materials Science*

| | |
|---|---|
| **Author** | Heike Emmerich |
| **Subtitle** | *Thermodynamic Concepts and Applications of Phase-Field Models* |
| **Publisher** | Springer, Berlin (Lecture Notes in Physics Monographs, vol. 73) |
| **Year** | 2003 |
| **ISBN** | 978-3-540-00416-5 |
| **Level** | Graduate / research |
| **Classification** | **Supplementary** |

**Why it matters here.** A shorter, more thermodynamically-focused companion to Provatas &
Elder. Its particular value is the emphasis on **thermodynamic consistency** — showing that
a phase-field model does not merely produce plausible interfaces but respects the second law,
with free energy decreasing monotonically under the dynamics. That question matters directly
for this project: the conservative Allen–Cahn equation (Eq. R.1) is *not* derived from a free
energy in the way Model A is, and understanding what is given up in exchange for discrete
mass conservation is worth being explicit about.

**Supports.** Technical document Sections 4.1–4.3 (free energy and relaxation dynamics),
4.5 (what the conservative reformulation trades away).

**What to read.**

- Chapter 2 — thermodynamic concepts and the derivation of phase-field models from
  irreversible thermodynamics.
- Chapter 3 — the sharp-interface limit, from a different angle than Provatas & Elder;
  useful as a second pass.
- Chapter 4 — applications, for a sense of the method's range.

Treat as a second opinion rather than a primary source. If Provatas & Elder Chapter 5 is
clear, this book is optional.

\newpage

# Part VII — Spectral Methods and Numerical Linear Algebra

## Trefethen, *Spectral Methods in MATLAB*

| | |
|---|---|
| **Author** | Lloyd N. Trefethen |
| **Publisher** | SIAM, Philadelphia (Software, Environments and Tools, vol. 10) |
| **Year** | 2000 |
| **ISBN** | 978-0-89871-465-4 |
| **Level** | Graduate |
| **Classification** | **Foundational** for the pressure solver |

**Why it matters here.** `src/pressure.py` solves the pressure Poisson equation by
diagonalising the discrete Laplacian in the Fourier basis. This book is the clearest
explanation in print of why that works and what its limits are. It is short — forty short
chapters, each a few pages with a complete runnable program — and it is unusually good at
building intuition rather than formalism.

Its relevance extends beyond the current prototype. FLOW36, the production framework
(Roccon, Soligo & Soldati 2025), is pseudo-spectral: Fourier series in the periodic
directions and **Chebyshev polynomials in the wall-normal direction**. Trefethen's Chebyshev
chapters are the entry point to that architecture.

**Supports.** Technical document Sections 8.3 (FFT solution of the Poisson equation), 13.3
(wall boundary conditions and alternative transform solvers), 13.5 (the FLOW36 port).

**What to read.**

- Chapters 1–4 — differentiation matrices and their spectral accuracy.
- **Chapters 5–8 — the discrete Fourier transform and its use for periodic problems.**
  Chapter 8 in particular covers Chebyshev and Fourier spectral differentiation side by
  side. Directly relevant to `_eig_laplacian_1d`.
- **Chapter 12 — polar coordinates and Poisson solvers.** Solving $\nabla^2 u = f$ by
  transform.
- Chapters 6–7 — aliasing and the Nyquist limit. Relevant to technical document Section 8.1's
  point that central differences have zero modified wavenumber at $k = \pi/h$, and to the
  Stefan-problem oscillation of Section 11.2.
- Chapters 11–14 — boundary conditions in spectral methods; how Dirichlet and Neumann
  conditions are imposed. This is the reading for the sine/cosine-transform option in
  technical document Section 13.3.

**Connection to the implementation.** The three-line core of `solve_poisson_3d` — transform,
divide by eigenvalues, inverse transform — is a textbook application of the technique
Chapters 5–8 develop. Trefethen also makes clear the distinction that
`src/pressure.py` gets right and that matters: dividing by the eigenvalues of the *discrete*
operator rather than the continuous $-k^2$, so that the solve inverts exactly the operator
the rest of the code differentiates with.

## Trefethen & Bau, *Numerical Linear Algebra*

| | |
|---|---|
| **Authors** | Lloyd N. Trefethen, David Bau III |
| **Publisher** | SIAM, Philadelphia |
| **Year** | 1997 |
| **ISBN** | 978-0-89871-361-9 |
| **Level** | Graduate |
| **Classification** | **Supplementary** |

**Why it matters here.** The technical document argues in Sections 1.3 and 8.2 that the
constant-coefficient Poisson equation is valuable because it avoids **iterative** solvers,
which scale badly on distributed GPUs. Making that argument properly requires knowing what
iterative solvers actually cost: conditioning, convergence rates, and why a coefficient jump
of 1000 across an interface wrecks them. Trefethen & Bau is the standard text, and its
treatment of conditioning and of Krylov methods is what turns "iterative solvers are slower"
into a quantitative claim.

**Supports.** Technical document Sections 1.3 (why the constant-coefficient property
matters), 8.2 (variable- vs constant-coefficient Poisson), 13.3 (choosing a replacement
solver for non-periodic boundaries).

**What to read.** Selectively:

- Lectures 1–7 — matrices, orthogonality, the SVD; the language.
- **Lectures 12–15 — conditioning and stability.** The framework for understanding why a
  1000:1 density ratio makes the variable-coefficient pressure equation of technical
  document Section 8.2 so badly conditioned.
- **Lectures 32–40 — iterative methods**: Arnoldi, GMRES, and conjugate gradients, including
  convergence rates as a function of the condition number. This is the quantitative basis
  for preferring a direct FFT solve.

Not needed to understand the current code, which contains no iterative solver at all. Needed
to defend the design choice, and to evaluate the tridiagonal option in Section 13.3.

## Boyd, *Chebyshev and Fourier Spectral Methods*

| | |
|---|---|
| **Author** | John P. Boyd |
| **Edition** | 2nd edition, revised |
| **Publisher** | Dover Publications, Mineola, NY |
| **Year** | 2001 |
| **ISBN** | 978-0-486-41183-5 |
| **Level** | Graduate / research |
| **Classification** | **Advanced** |

**Why it matters here.** Where Trefethen's *Spectral Methods in MATLAB* is an intuition
builder, Boyd is the comprehensive reference — and it is freely available from the author's
university page as well as in a cheap Dover print edition. It becomes relevant at the point
where the project moves to FLOW36's pseudo-spectral, wall-bounded formulation, which is
where the subtleties Boyd catalogues (aliasing control, the choice of basis for non-periodic
directions, boundary-condition treatment) start to matter.

**Supports.** Technical document Sections 13.3 (alternative pressure solvers), 13.5 (the
FLOW36 port).

**What to read.**

- Chapters 1–2 — introduction, Chebyshev and Fourier series, and convergence theory.
- **Chapter 5 — Chebyshev polynomials and their use for non-periodic problems.** The basis
  FLOW36 uses in the wall-normal direction.
- Chapter 6 — pseudospectral methods and aliasing; the 2/3 rule.
- Chapter 15 — matrix-solving methods for spectral discretisations, including the
  tridiagonal structure that arises when transforming only the periodic directions. This is
  the theoretical background for the second option in technical document Section 13.3.

## Peyret, *Spectral Methods for Incompressible Viscous Flow*

| | |
|---|---|
| **Author** | Roger Peyret |
| **Publisher** | Springer, New York (Applied Mathematical Sciences, vol. 148) |
| **Year** | 2002 |
| **ISBN** | 978-0-387-95221-5 |
| **Level** | Research |
| **Classification** | **Advanced** |

**Why it matters here.** The most specific book on this list. It treats exactly the problem
FLOW36 solves: incompressible Navier–Stokes by spectral methods, including the pressure
problem in wall-bounded geometries, Fourier–Chebyshev discretisation of a channel, and the
influence-matrix and projection techniques for handling pressure boundary conditions. When
the project reaches the Fortran port and the wall-bounded formulation, this is the reference
that closes the gap between the periodic FFT solver in `src/pressure.py` and a production
channel-flow solver.

**Supports.** Technical document Sections 13.3 (wall/outlet boundary conditions and the
FFT-plus-tridiagonal solver), 13.5 (the FLOW36 port).

**What to read.**

- Chapter 2 — the fundamentals of spectral approximation.
- **Chapter 3 — the Fourier and Chebyshev methods for the Navier–Stokes equations.**
- **Chapter 6 — solution of the Stokes and Navier–Stokes problems**, including the pressure
  boundary-condition difficulty and the influence-matrix method. This is the technically
  hardest part of moving from periodic to wall-bounded, and it is where the pressure solver
  redesign will actually be decided.
- Chapter 7 — applications to channel flow.

Read after Trefethen and Boyd, and only when the wall-bounded work begins.

\newpage

# Part VIII — High-Performance and GPU Computing

## Kirk, Hwu & El Hajj, *Programming Massively Parallel Processors*

| | |
|---|---|
| **Authors** | Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj |
| **Subtitle** | *A Hands-on Approach* |
| **Edition** | 4th edition |
| **Publisher** | Morgan Kaufmann / Elsevier |
| **Year** | 2022 |
| **ISBN** | 978-0-323-91231-0 |
| **Level** | Graduate / practitioner |
| **Classification** | **Advanced** — PhD Year 2 |

**Why it matters here.** The production target is a boiling-capable fork of FLOW36, in
Fortran with CUDA and MPI, running on GPU clusters. This is the standard text on GPU
programming, and its relevance is not generic: the specific patterns it teaches — memory
coalescing, tiling, halo exchange for stencil computations, and the parallel primitives
needed for irregular work — map directly onto the two computational kernels this project
depends on.

The stencil chapters cover exactly the structure of `src/operators.py`: a fixed
neighbourhood computation over a regular grid, whose GPU performance is bounded by memory
bandwidth rather than arithmetic. And the chapters on irregular parallelism are the relevant
background for the one genuinely awkward part of the port, identified in technical document
Section 13.5: the **probe search** for the vaporisation rate, which is the only non-local
operation in the algorithm and the only one that does not reduce to a halo exchange.

**Supports.** Technical document Sections 1.3 (GPU scalability of FFT vs iterative solvers),
8.3 (portability of the FFT solver), 13.5 (the Fortran/CUDA/MPI port).

**What to read.**

- Chapters 1–4 — the CUDA execution model, memory hierarchy, and the fundamentals of
  writing a kernel.
- **Chapter 5 — memory architecture and data locality**, and **Chapter 6 — performance
  considerations.** Together these explain why a stencil code's performance is set by
  memory access patterns, which is the first thing that will need attention in the port.
- **Chapter 8 — stencil computation.** The direct analogue of `src/operators.py`; tiling
  strategies and halo handling.
- Chapter 7 — convolution, closely related to stencils.
- Chapters 12–14 — merge, sorting, and sparse computation; the patterns needed for the
  irregular probe search.
- Chapter 20 — programming a heterogeneous cluster with MPI and CUDA together. The exact
  configuration FLOW36 and MHIT36 use.

**Relationship to the cited papers.** The implementation background for Roccon, Soligo &
Soldati (2025) (FLOW36, MPI + OpenACC + cuFFT) and Roccon et al. (2025) (MHIT36, cuDecomp +
cuFFT, scaling to 1024 GPUs). Note that both production codes use **OpenACC directives**
rather than raw CUDA; this book teaches CUDA, which is the more transferable foundation, and
the directive-based approach is a simplification on top of the same execution model.

\newpage

# Where the Books Stop

It is worth being explicit about the boundary of what the literature above covers, because
mistaking a gap for a gap in one's own understanding wastes time.

**Covered well by books.** Diffuse-interface theory from a free-energy functional
(Provatas & Elder); the sharp-interface limit by matched asymptotics (Provatas & Elder
Chapter 5); the one-fluid formulation and CSF surface tension (Tryggvason, Scardovelli &
Zaleski); projection methods for variable-density flow (Ferziger, Perić & Street Chapter 7;
Tryggvason et al. Chapter 3); the stability and dissipation properties of finite differences
(LeVeque, Strikwerda); spectral and transform-based Poisson solvers (Trefethen, Boyd,
Peyret); the physics of boiling (Carey, Brennen, Incropera et al.); turbulence and DNS cost
(Pope); GPU stencil computation (Kirk, Hwu & El Hajj).

**Not covered by any book on this list — papers required.**

| Topic | Where it lives | Technical doc section |
|---|---|---|
| Conservative Allen–Cahn equation | Chiu & Lin (2011); Olsson & Kreiss (2005) | 4.5, 6 |
| Provable boundedness of $\phi$ with central differences | **Mirjalili, Ivey & Mani (2020)**; Jain et al. (2020) | 4.5, 9.1 |
| Parameter criteria $\gamma \geq \lvert u\rvert_{\max}$, $\varepsilon > 0.5\Delta x$ | Jain (2022) | 4.5, 9.1, 13.1 |
| Phase-field + Navier–Stokes coupling | **Jacqmin (1999)** | 4.3, 6 |
| Conservative AC for drop-laden turbulence | Roccon, Zonta & Soldati (2023) | 6 |
| Phase change in a conservative AC framework | **Roccon (2025)**; Haghani-Hassan-Abadi et al. (2021); Tamura & Katono (2022) | 7 (all) |
| Probe method for one-sided interfacial gradients | Udaykumar et al. (1996); Irfan & Muradoglu (2017) | 7.6, 13.1 |
| Diffuse-interface velocity-profile asymmetry | Sun & Beckermann (2004) | 5.4, 11.1 |
| The 2D bubble-growth benchmark | Tanguy et al. (2014) | 10.1 |
| Stefan-problem reference data | Sun, Xu & Chen (2014); Welch & Wilson (2000) | 10.2 |

The single most important gap is the second row. The **conservative Allen–Cahn formulation
with provable boundedness is the reason this project's method works at density ratio 1000**,
and it exists only in Mirjalili, Ivey & Mani (2020) and its successors. No textbook covers
it. Provatas & Elder will explain everything about Allen–Cahn *except* the modification that
makes it usable here.

\newpage

# Cross-Reference Matrix

Book to technical-document section, for use in the other direction — when a section of the
technical document is unclear, this table says which book to open.

| Technical document section | Primary book | Secondary |
|---|---|---|
| 1.1 Why boiling matters | Incropera et al. Ch. 10 | Carey Ch. 7–9 |
| 1.2 Why DNS | Pope Ch. 6 | Ferziger et al. Ch. 9 |
| 1.3 Why phase-field | Prosperetti & Tryggvason Ch. 1 | Tryggvason et al. Ch. 1 |
| 2.1 Sharp-interface equations | Tryggvason et al. Ch. 2 | Chorin & Marsden Ch. 1 |
| 2.2 What makes it hard | Tryggvason et al. Ch. 5 | Brennen Ch. 4 |
| 3.1–3.3 Diffuse interface, $\phi$, $\varepsilon$ | Provatas & Elder Ch. 3 | Emmerich Ch. 2 |
| 3.4 Survey of interface methods | Prosperetti & Tryggvason Ch. 3–5 | Tryggvason et al. Ch. 4, 6 |
| 4.1 Ginzburg–Landau functional | **Provatas & Elder Ch. 3** | Emmerich Ch. 2 |
| 4.2 Chemical potential | Provatas & Elder Ch. 4 | Evans Ch. 8 |
| 4.3 Cahn–Hilliard vs Allen–Cahn | **Provatas & Elder Ch. 4** | Emmerich Ch. 2 |
| 4.4 Sharp-interface limit | **Provatas & Elder Ch. 5** | Emmerich Ch. 3 |
| 4.5 Conservative Allen–Cahn | *(papers — see Section 9)* | Provatas & Elder Ch. 4 |
| 5 Classical boiling methods | Prosperetti & Tryggvason Ch. 3–4 | Tryggvason et al. Ch. 6, 9 |
| 7.1–7.2 Allen–Cahn, mass conservation | Tryggvason et al. Ch. 2, 9 | Bird et al. Ch. 3 |
| 7.3 One-fluid Navier–Stokes | **Tryggvason et al. Ch. 2** | Anderson Ch. 2 |
| 7.4 CSF surface tension | **Tryggvason et al. Ch. 5** | Provatas & Elder Ch. 3 |
| 7.5 Energy equation | Bird et al. Ch. 11 | Incropera et al. Ch. 2 |
| 7.6 Rankine–Hugoniot closure | **Carey Ch. 4–5** | Tryggvason et al. Ch. 9 |
| 8.1 Spatial discretisation | **LeVeque Ch. 1–2** | Ferziger et al. Ch. 4 |
| 8.2 Constant-coefficient Poisson | **Tryggvason et al. Ch. 3** | Chorin & Marsden Ch. 1.3 |
| 8.3 FFT Poisson solver | **Trefethen, *Spectral Methods* Ch. 5–8** | Moin Ch. 6 |
| 8.4 Projection–correction | Ferziger et al. Ch. 7 | Chorin & Marsden Ch. 3 |
| 8.5 Explicit stability | **Strikwerda Ch. 4** | LeVeque Ch. 7–8 |
| 9 Code implementation | LeVeque Ch. 1–2 | Moin Ch. 5 |
| 10.1 Bubble benchmark | Brennen Ch. 4 | Carey Ch. 7 |
| 10.2 Stefan problem | **Incropera et al. Ch. 5** | Carey Ch. 5 |
| 11.1 Convergence study | Provatas & Elder Ch. 5 | LeVeque Ch. 1 |
| 11.2 Stefan breakdown | **LeVeque Ch. 10** | Strikwerda Ch. 4 |
| 12.2 Surface-tension time step | **Tryggvason et al. Ch. 5** | Strikwerda Ch. 4 |
| 12.3 Periodic BCs only | Ferziger et al. Ch. 8 | Peyret Ch. 6 |
| 12.4 No oscillation control | **LeVeque Ch. 10** | Ferziger et al. Ch. 7 |
| 12.7 Collocated vs staggered | **Ferziger et al. Ch. 7** | Moin Ch. 5 |
| 13.3 Wall/outlet BCs | **Peyret Ch. 6** | Boyd Ch. 5, 15 |
| 13.4 Higher-order time integration | LeVeque Ch. 7 | Moin Ch. 2 |
| 13.5 Fortran/CUDA/MPI port | **Kirk et al. Ch. 8, 20** | Ferziger et al. Ch. 12 |
| 13.6 Turbulent boiling, Nu scaling | **Pope Ch. 7** | Incropera et al. Ch. 10 |

\newpage

# Summary Table

| # | Book | Classification | Phase | Core chapters |
|---|---|---|---|---|
| 1 | Chorin & Marsden, *Math. Intro. to Fluid Mechanics* | Foundational | 0 | 1.1–1.3, 3 |
| 2 | Bird, Stewart & Lightfoot, *Transport Phenomena* | Foundational | 0 | 1–3, 9–11 |
| 3 | Bergman et al., *Fundamentals of Heat and Mass Transfer* | Foundational | 1 | 2, 5, 6, 10 |
| 4 | Carey, *Liquid–Vapor Phase-Change Phenomena* | Foundational | 1 | 2, 4, 5, 7–9, 11 |
| 5 | Evans, *Partial Differential Equations* | Supplementary | 2 | 2.3, 7.1, 8.1–8.2 |
| 6 | LeVeque, *Finite Difference Methods for ODEs and PDEs* | Foundational | 2 | 1, 2, 5, 7–10 |
| 7 | Strikwerda, *Finite Difference Schemes and PDEs* | Supplementary | 2 | 2–4, 6, 9, 13 |
| 8 | Moin, *Fundamentals of Engineering Numerical Analysis* | Supplementary | 3 | 2, 5, 6 |
| 9 | Ferziger, Perić & Street, *Comp. Methods for Fluid Dynamics* | Foundational | 3 | 4, 7, 8, 9, 12 |
| 10 | Anderson, *CFD: The Basics with Applications* | Supplementary | 3 | 2, 4, 6 |
| 11 | Trefethen, *Spectral Methods in MATLAB* | Foundational | 4 | 1–8, 11–14 |
| 12 | Trefethen & Bau, *Numerical Linear Algebra* | Supplementary | 4 | 12–15, 32–40 |
| 13 | Prosperetti & Tryggvason, *Comp. Methods for Multiphase Flow* | Foundational | 5 | 1–5 |
| 14 | Tryggvason, Scardovelli & Zaleski, *DNS of Gas–Liquid Multiphase Flows* | Foundational | 5 | 1–6, 9 |
| 15 | Provatas & Elder, *Phase-Field Methods in Mat. Sci. & Eng.* | Foundational | 6 | 2–7 |
| 16 | Emmerich, *The Diffuse Interface Approach in Materials Science* | Supplementary | 6 | 2–4 |
| 17 | Brennen, *Fundamentals of Multiphase Flow* | Supplementary | 7 | 1–2, 4, 6, 15 |
| 18 | Pope, *Turbulent Flows* | Advanced | 8 | 1–3, 6, 7, 9.1 |
| 19 | Boyd, *Chebyshev and Fourier Spectral Methods* | Advanced | 9 | 1–2, 5, 6, 15 |
| 20 | Peyret, *Spectral Methods for Incompressible Viscous Flow* | Advanced | 9 | 2, 3, 6, 7 |
| 21 | Hwu, Kirk & El Hajj, *Programming Massively Parallel Processors* | Advanced | 10 | 1–8, 12–14, 20 |

**Ten foundational, seven supplementary, four advanced.**

\newpage

# Full Bibliographic Details

Anderson, J. D., Jr. (1995). *Computational Fluid Dynamics: The Basics with Applications*.
McGraw-Hill, New York. ISBN 978-0-07-001685-9.

Bergman, T. L., Lavine, A. S., Incropera, F. P., and DeWitt, D. P. (2017). *Fundamentals of
Heat and Mass Transfer*, 8th edition. John Wiley & Sons, Hoboken, NJ.
ISBN 978-1-119-35388-1.

Bird, R. B., Stewart, W. E., and Lightfoot, E. N. (2007). *Transport Phenomena*, Revised 2nd
edition. John Wiley & Sons, Hoboken, NJ. ISBN 978-0-470-11539-8.

Boyd, J. P. (2001). *Chebyshev and Fourier Spectral Methods*, 2nd edition (revised). Dover
Publications, Mineola, NY. ISBN 978-0-486-41183-5.

Brennen, C. E. (2005). *Fundamentals of Multiphase Flow*. Cambridge University Press,
Cambridge. ISBN 978-0-521-84804-6.

Carey, V. P. (2020). *Liquid–Vapor Phase-Change Phenomena: An Introduction to the
Thermophysics of Vaporization and Condensation Processes in Heat Transfer Equipment*, 3rd
edition. CRC Press / Taylor & Francis, Boca Raton, FL. ISBN 978-1-4987-1661-1.

Chorin, A. J., and Marsden, J. E. (1993). *A Mathematical Introduction to Fluid Mechanics*,
3rd edition. Texts in Applied Mathematics, vol. 4. Springer-Verlag, New York.
ISBN 978-0-387-97918-2. DOI: 10.1007/978-1-4612-0883-9.

Emmerich, H. (2003). *The Diffuse Interface Approach in Materials Science: Thermodynamic
Concepts and Applications of Phase-Field Models*. Lecture Notes in Physics Monographs,
vol. 73. Springer, Berlin. ISBN 978-3-540-00416-5.

Evans, L. C. (2010). *Partial Differential Equations*, 2nd edition. Graduate Studies in
Mathematics, vol. 19. American Mathematical Society, Providence, RI.
ISBN 978-0-8218-4974-3.

Ferziger, J. H., Perić, M., and Street, R. L. (2020). *Computational Methods for Fluid
Dynamics*, 4th edition. Springer, Cham. ISBN 978-3-319-99691-2.
DOI: 10.1007/978-3-319-99693-6.

Hwu, W. W., Kirk, D. B., and El Hajj, I. (2022). *Programming Massively Parallel Processors:
A Hands-on Approach*, 4th edition. Morgan Kaufmann / Elsevier, Cambridge, MA.
ISBN 978-0-323-91231-0.

LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential
Equations: Steady-State and Time-Dependent Problems*. SIAM, Philadelphia.
ISBN 978-0-89871-629-0.

Moin, P. (2010). *Fundamentals of Engineering Numerical Analysis*, 2nd edition. Cambridge
University Press, Cambridge. ISBN 978-0-521-71123-4.

Peyret, R. (2002). *Spectral Methods for Incompressible Viscous Flow*. Applied Mathematical
Sciences, vol. 148. Springer, New York. ISBN 978-0-387-95221-5.

Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press, Cambridge.
ISBN 978-0-521-59886-6.

Prosperetti, A., and Tryggvason, G. (eds.) (2007). *Computational Methods for Multiphase
Flow*. Cambridge University Press, Cambridge. ISBN 978-0-521-84764-3.

Provatas, N., and Elder, K. (2010). *Phase-Field Methods in Materials Science and
Engineering*. Wiley-VCH, Weinheim. ISBN 978-3-527-40747-7.

Strikwerda, J. C. (2004). *Finite Difference Schemes and Partial Differential Equations*,
2nd edition. SIAM, Philadelphia. ISBN 978-0-89871-639-9.

Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. Software, Environments and Tools,
vol. 10. SIAM, Philadelphia. ISBN 978-0-89871-465-4.

Trefethen, L. N., and Bau, D., III (1997). *Numerical Linear Algebra*. SIAM, Philadelphia.
ISBN 978-0-89871-361-9.

Tryggvason, G., Scardovelli, R., and Zaleski, S. (2011). *Direct Numerical Simulations of
Gas–Liquid Multiphase Flows*. Cambridge University Press, Cambridge.
ISBN 978-0-521-78240-1. DOI: 10.1017/CBO9780511975264.

## Note on verification

Every entry above was checked against publisher records, the OpenLibrary catalogue, or
Crossref before inclusion; editions and ISBNs correspond to the specific printings named.
Where a book exists in several editions, the most recent has been listed except where an
earlier one is the standard citation.

Three of these books are cited in the technical document's own reference list or in Roccon
(2025): Tryggvason, Scardovelli & Zaleski (2011) appears in both; Carey (2020) is Roccon's
reference [52]; and Chorin's 1968 projection-method paper — the subject of Chorin & Marsden
Chapter 3 — is Roccon's reference [59].
