# Step 05: radiation verification coverage and findings

## Provenance and scope

Started at clean Step 04 commit `1fd4502a9d186b3b496d477c64c905143214b7c9`.
Verified archive SHA-256:
`360d2fffc9947face4bd4e9d7afb22c9899ac4da4b35d4bf100cb684a90ee70c`.
The existing S&O serial production-method baseline and opacity limiting cases
passed before changes; their stdout remains unchanged afterwards.

The suite executes production `rad_grid` transport and source methods and the
production `compute_radiation` and `all_rad_bounds` orchestration methods. Mesh,
material fixtures, and communication endpoints are serial stand-ins. This is
not a full Octo-TIGER executable, distributed validation, or a replacement for
its hydro solver. No application physics, opacity, timestep, scheduler, hydro,
gravity, or boundary source file was modified.

One explicit verification-only exception is the piecewise-constant adapter: a
test wrapper supplies `(center,center)` in place of `M1::reconstruct`. All other
methods and the VL predictor/corrector are production code. This does not add
or claim a production command-line reconstruction switch.

## Definition and reference rules

Each descriptor records physical c=2.99792458e10 cm/s, geometry in cm, final time
in seconds, initial condition, opacity in inverse cm, density and energy units,
boundary condition, refinement levels, visualization products, and diagnostic
policy. Native runs use a 6e10 cm periodic cube with x-only data, N=8,16,32 per
direction. These are explicitly **diagnostic variants**, not substitutions for
published problem definitions. The legacy four tests retain their original
scaled configs and application commands, including their 4-second final time.

The existing `equilibrium_sphere` is a Gaussian luminosity distribution with an
analytic diffusion-limit solution. It is neither a uniform emitting sphere nor
an exact nonlinear M1 equilibrium. It must not be relabeled as a reproduced
Ensman benchmark. Its identity is preserved under `skinner_ostriker` for command
compatibility, with this limitation recorded in its descriptor.

The damped wave uses the equations in Skinner & Ostriker (2013), section 4.1.2:
https://arxiv.org/abs/1306.0010 . It retains amplitude 1e-6 and optical depth 0.1
per wavelength. This implementation evolves the complete radiation mean and
perturbation under absorption, so its exact reference includes decay of both.
It is explicitly an equation-(57) variant, not a reproduction of their Figure 4.
Physical c and the chosen wavelength fix the period; no light-speed reduction
or reference rescaling is hidden.

Streaming references are exact finite-volume averages. The Gaussian is an
exact periodized streaming beam, distinct from the legacy isotropic Gaussian.
Diffusion references use an independent matrix exponential of the linearized
finite-speed moment equations, rather than treating the parabolic limit as
exact M1. The 1/chi diffusion asymptote is checked separately. Thermal and moving
source references use independent high-accuracy DOP853 ODE integration. They do
not call the production implicit solver to generate reference trajectories.

No verified primary Ensman sphere definition or radiative-shock reference data
was available. Attempts to retrieve the primary material were unsuccessful.
Their descriptors have explicit null physical parameters and conditional status;
no invented domain, curve, tolerance, or successful numerical result is supplied.
Completing these cases requires verified source definitions and appropriate
application boundary/initial-condition support. Full dynamic-diffusion advection
is also not established by the homogeneous moving-source diagnostic.

## Status from three-resolution sweeps

The numerical values below were measured during precommit verification. The
handoff also runs the finalized suite from the committed tree, whose exact
commit and clean state are recorded in the generated report.

| Test | Finest normalized L1 | Last order | Status / meaning |
|---|---:|---:|---|
| 1-D streaming wave | 9.659058e-03 | 1.924280 | Passed; F=cE, phase speed, bounds, conservation |
| 1-D streaming front | 6.254313e-02 | 0.681175 | Passed; discontinuity uses reduced convergence order |
| Thin streaming Gaussian | 1.823631e-02 | 1.638262 | Passed; exact periodized beam |
| Static diffusion, chi L=30 | 2.283386e-03 | 2.688585 | Passed; finite-speed and diffusion-limit checks |
| Static diffusion, chi L=100 | 4.627008e-03 | 2.298956 | Passed; thicker-medium comparison |
| Thermal relaxation | 8.765976e-04 | 1.001324 | Passed; expected first-order source time convergence |
| Moving scattering | 1.409767e-08 | 0.927055 | Passed; homogeneous work, bounds, conservation |
| Piecewise-constant wave | 9.076141e-02 | 0.885171 | Passed exact discrete-method check and first-order convergence |
| Subcycles / boundary epochs | roundoff | not fitted | Passed in serial production orchestration |
| S&O-style damped wave | 1.884574e-05 | 0.999614 | **Failed** second-order convergence requirement |
| Existing four application runs | — | — | Conditional: full HPX/Silo build and visualization dependencies absent |
| Ensman sphere and radiative shock | — | — | Conditional: unverified primary definition/reference; not implemented |

The damped-wave individual resolution checks pass, but its sweep fails the
retained minimum order 1.6. The mean absorption error dominates the 1e-6 wave
amplitude. This is consistent with the current backward-Euler absorption update;
no solver or tolerance change was made to conceal it. The suite returns failure.

These are observed orders over the stated range, not proofs of asymptotic order.
In particular, the diffusion results do not establish accuracy for arbitrary
unresolved mean free paths or stiff moving-material terms. Results for moving
gas conserve total gas energy; internal energy alone is not the invariant.

## Accuracy-policy correction for piecewise-constant isolation

The first pass incorrectly applied the PLM-oriented continuum envelope 2/N to
the first-order spatial method. All three resolutions exceeded that envelope.
It was replaced by a stronger, independently derived check of the actual
constant-reconstruction VL method. For a Fourier mode, with z=1-exp(-ik dx),
the per-step amplification is G=1-Cz+(Cz)^2/2. Multiplying G over the *recorded*
timesteps predicts the discrete final wave. The measured maximum residuals
were 1.110223e-16, 4.440892e-16, and 1.554312e-15 erg/cm^3. The acceptance bound
is 5e-12 erg/cm^3, and continuum errors and first-order convergence remain
reported. No observed continuum-error coefficient was fitted to obtain a pass.

Other criteria are explicit in each descriptor: relative budgets 2e-11,
reduced flux <=1+2e-12, nonnegative cell energies, phase error <=2pi/N, and
resolution-scaled continuum error envelopes. Discontinuous fronts require order
0.35; smooth streaming 1.6; thin Gaussian 1.3; static diffusion 0.7 (mixed space
and source discretization). Thermal and moving-source orders are reported;
the boundary test's roundoff errors are not assigned a convergence order.
The diffusion amplitude bound includes both 2/N discretization allowance and
2(k/chi)^2 for the finite-inertia departure from the parabolic limit. None of
these tolerances is a universal physical-accuracy guarantee.

## Boundary-counter diagnostic

For every gas step, `exchanges.csv` records hydro refresh, radiation state
exchange, and flux-correction events. It exposes hcycle, rcycle, physical time,
interior E and Fx at the send point, halo validity, and gas-array preservation.
Flux events record the substep start time and predictor state; the flux itself
is time-centered. They must identify the same interval as the preceding state exchange.
Checks require one hydro refresh, Nsub radiation exchanges plus final exchange,
exact monotonically increasing radiation epochs, flux epochs equal to the
preceding state epoch plus one, and unchanged hydro epoch during radiation
substeps. Every substep time is compared with the exact gas-interval partition;
the final exchange must occur at the final gas time. The fixture verifies that
halos contain the current state, not just that a call occurred. Nonzero damping
makes stale flux data observable. Gas ghost arrays are checked after each step.

The existing baseline also checks refined-parent schedules, inherited radiation
epochs on migration, independent gas/radiation minima, half-time physical inflow,
and AMR face restriction. New negative tests mutate epochs, time, halo validity,
and gas-preservation flags to ensure the checker rejects them.

Actual asynchronous HPX channels and shared hydro buffers remain conditional.
The communication endpoint fixture cannot detect a race inside a real remote
receiver. No evidence justified changing the shared scheduler in this step.

## Commands and products

From HOME:

```bash
cd ~
repo="$HOME/octotiger/src/octotiger"
"$repo/verification_results/run.sh" suite radiation 0 1 2 Release --threads 1
"$repo/verification_results/run.sh" run radiation.diagnostics.streaming_wave_1d 0 1 2 Release
"$repo/verification_results/run.sh" run radiation.skinner_ostriker.damped_wave 0 1 2 Release
"$repo/verification_results/run.sh" plan radiation.ensman
# Existing application commands, on the supported HPX/Silo/VisIt stack:
"$repo/radiation_results/run_live.sh" all 2 3 4 Release --threads 12
```

Each native level produces samples.csv, history.csv, comparison.csv, run.log,
run.json, comparison.png, movie.mp4, movie.log, and product metadata. Boundary
runs also populate exchanges.csv. Nine time samples are actual solver states;
movies show those snapshots at four frames/second, not interpolated simulation.
The live index updates after every completed level. The portable report embeds
previews and downloads of every resolution's logs, metadata, numerical data,
plots and movies. There is no public deployment or manual file-copy requirement.

Expected completed products are 30 profile/signed-error plots, 30 MP4 movies,
and nine convergence plots. Failed numerical sweeps still retain their plots
and raw data. Any plot/encoder failure causes a failed result and nonzero exit.
The test suite includes an intentionally failing encoder to verify this.

Source metadata records the commit, dirty flag, compiler and build flags,
thread count, exact descriptor/options, source hashes, generated fixture hash,
executable hash, and Python library versions. Runtime histories record actual
gas/radiation dt, subcycle counts and epochs. `SOURCE_VERSION` is expanded by
git archive so exported trees also know their commit. Large generated products
remain outside the source commit. Return codes distinguish passed (0), failed
(1), invalid invocation (2), and conditional without failures (3).

## Remaining gates

Step 05 is **not fully accepted**: the damped-wave convergence target fails,
published Ensman cases are conditional/unimplemented, and the four legacy
application runs and actual HPX/MPI delivery need the full runtime stack. The
existing legacy source-method baseline, Step 04 opacity checks, and the 18
harness/negative tests pass. No Step 06 work is included.
