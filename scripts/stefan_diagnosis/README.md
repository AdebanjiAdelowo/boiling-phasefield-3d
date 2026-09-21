# Stefan-benchmark diagnosis harness

Exploratory, self-contained 1-D harness used to isolate why the legacy Stefan benchmark
failed and to demonstrate which changes were necessary. It produced the tables in
"The Stefan problem" of `docs/Phase_Field_Boiling_Solver_Technical_Documentation.md`.
It is **not** the production solver (that is `src/stefan1d.py`, which reproduces this
harness's corrected configuration to 3e-18) and is not part of the test suite.

Run from this directory (`cd scripts/stefan_diagnosis`); scripts write `*.pkl` files here.

* `stefan_lab.py`, `lab2.py`: switchable scheme. **Defaults reproduce the legacy inline scheme of
  `examples/stefan_1d.py` bit-for-bit** (`python stefan_lab.py` and `python lab2.py` print
  `max |lab - repo| = 0.0`; the check pins `gamma = 0.1`, the `SimParams` heat_flux-mode default at HEAD). Each switch is one numerical/model choice.
* `verify_analytical.py`: independent check of the analytic reference (Stefan condition, exact
  energy budget, front-fixing PDE solve). Superseded by `tests/test_stefan_1d.py`.
* `run_baseline_diag.py`: the instrumented legacy run (mass, energy, clamp budgets).
* `s1_thermal_only.py`, `s1b_interface_flux.py`: energy equation alone, interface prescribed exactly.
* `s2_interface_only.py`, `s2b_front_speed.py`, `s2c_width.py`: interface side alone (exact T; constant-speed
  front; layer-width stability of the two mass-source forms).
* `ablate.py`, `late_ablate.py`, `more_ablate.py`: forward and leave-one-out ablations.
* `coupled_refine.py`, `coupled2.py`, `studies.py`: coupled refinement; time-step, interface-thickness,
  gamma and probe-distance studies.

Caveats:
* `s1c_measure.py` (sub-grid Dirichlet "ghost" front) is **unstable for N >= 400** (tiny cut fractions give a huge
  ghost value). Only its N = 100, 200 rows are meaningful; the N >= 400 output is not a result.
* Cases run on shared, loaded hardware; wall-clock times are not meaningful.
* `coupled_refine.py` uses a superseded closure/source combination and is kept because the docs cite its
  finding that the source form, not the closure, caused the late-time error growth under refinement.
