# ULTRA follow-up to Step 07

This follow-up repairs demonstrated test/infrastructure defects. It does **not**
turn the Step 07 physics decision into a GO: full application hydro/gravity,
published Ensman problems, distributed boundary delivery, oblique transport and
Silo restart still need direct execution. No production physics files in `src/`
or `octotiger/` were changed. Hydro reconstruction, gravity, AMR algorithms,
timestep rules, radiation equations, opacity defaults and source solvers remain
unchanged.

The Step 07 commit `c31219be8f93c025438cc29d3726b8a7ce6e2146` and its archive
are preserved. Archive SHA-256:
`9b4be6272a69de9f099ff3289c476b132fdb9ee670b98b25a917e0648cca056e`.
This follow-up is an explicit patch against that commit, not a rewritten Step 07
history or a second Step 07 commit. The generated follow-up assessment records
the patch digest, working-tree state and per-run source/executable hashes.

All 90 native runs (ten cases, three levels, three modes) pass, with 90 complete
movies, 90 comparison/signed-error plots and 27 convergence plots. Independent
post-run audits verify all 90 movies by decoding every frame, retained raw
schemas/cadences, manifests, generated C++/executable digests, metadata and
embedded report payloads. All 56 harness tests pass without skips. Standalone
core and restored radiation CTest projects each pass seven tests in each mode;
real Boost compatibility-helper parser executables pass all three modes.

Matrix source snapshots predate final CTest archival durability, provenance and
cleanup-containment hardening, and smoke CLI compatibility fixes. Those changes
received subsequent harness tests; matrix records are not relabeled as exact
final-tree runs. The strict source audit intentionally fails on the two changed
adapter files in each mode. Artifact integrity separately passes. Native
radiation numerical code was unaffected; application hydro/gravity validation
remained conditional. Generated C++ and executable hashes agree with run records.

## Resolved or materially improved

| Issue | Diagnosis and change | Validation scope |
| --- | --- | --- |
| Corrupt raw files/movies labeled passed | Old adapter could accept files changed after evaluation; buffered close errors and movie completeness were unchecked. Added staged outputs, explicit close, strict CSV/cadence checks, fsync/readback, atomic publication, full movie decode and digest manifests with final audits. | Injected truncations now return failed/nonzero; original files remain untouched. Fresh native runs and independent audits are in the generated report. |
| Damped wave expected second order | Theta=1 absorption is backward Euler. Its first-order mean error dominates the tiny wave perturbation. Corrected the convergence policy and added a stricter independent discrete mean check. | Same continuum reference, same L1 bound, same numerical results; phase/symmetric-streaming checks also enforced. |
| Four broken Python radiation drivers | Removed names, fields and source interfaces had made tests unusable. Adapted fixtures to current production methods and split-source sequencing. | Physical-grid bounds 3e-12 and kernel-grid bounds 3e-11 unchanged; Debug/Release/sanitizer checks and unequal ghost strides pass. Four-profile driver remains metric-only. |
| Standalone radiation CMake/reference tests | Reference generator moved; names in reference C++ tests changed. Restored the generator and reference tests mechanically. | Independent DFT/ODE/binary/profile assertions retained. Current M1 successor has a distinct name; obsolete historical M1 is not claimed equivalent. |
| Hydro/gravity adapter omitted CTest diagnostics | Added configured-CTest inventory execution with complete selected fixtures, checks, variants and Silo comparisons. | Real CMake registration inventories and synthetic CTest execution validate adapter mechanics, not astrophysical results. |
| Legacy launch defects | Fixed six-argument Sod registration, Sod/blast cleanup directories, duplicate rotating-star cleanup and invalid Sod shell assignments. | Registration tests cover grid sizes 8/16 and separate CPU/CUDA/HIP/SYCL profiles. No physics inputs or thresholds retuned. |
| Boost parser could not compile | Acquired official matching Boost 1.83 headers in a task-local directory, linked installed runtime. | Actual compatibility-helper parser executable runs in Debug, Release and RelWithDebInfo. Full application parser remains unrun. |
| Missing Ensman definitions | Recovered published radiative-shock reproduction parameters and documented equation/boundary ambiguities. | Both descriptors remain conditional; neither reference is silently substituted for the current equations. |

## Why first order is correct for this damped-wave test

For constant absorption, velocity terms disabled, periodic transport and theta=1,

\[
\bar E_{n+1}=\frac{\bar E_n}{1+\hat c\chi_a\Delta t_n},\qquad
\bar E(t)=\bar E_0\exp(-\hat c\chi_a t).
\]

The accumulated backward-Euler error is first order in timestep. With a CFL step
proportional to cell width it is first order under spatial refinement too. This
does not imply that source-free smooth-wave transport is first order.

Measured orders at N=8/16/32 are 0.950345 and 0.999614. The invalid minimum 1.6
is changed explicitly to 0.8, a finite-resolution allowance below the theoretical
order 1. The continuum exponential reference and original normalized L1 <= 2/N
remain. New all-timestep mean residual <= 5e-12 and existing phase and F=cE
symmetry guards make this more than a relaxed convergence gate. Negative tests
reject the wrong source amplification or recorded timestep. Historical failures
are retained, not rewritten as passes.

## Artifact diagnosis: what is and is not established

Exact retained Step 07 binaries, rerun with the same inputs in new directories,
produce complete thermal histories. Nonempty truncated originals are exact byte
prefixes of those repeats. Re-encoding the original boundary movie frames
produces a valid nine-frame movie. The broken originals equal the corresponding
embedded report downloads, so corruption predates embedding. These observations
do not identify the original loss mechanism or prove a deterministic numerical
generator defect. In particular, the old boundary movie log ends at frame zero;
it is not a complete successful encoder log.

The new checks establish integrity at publication/final audit. They cannot
prevent arbitrary external mutations after exit; retained manifests allow later
read-only revalidation. Independent source-identity differences are reported
separately from raw/product integrity, never silently ignored.

## Coverage and unresolved gates

- The native matrix uses GCC 13.3.0, C++23, one thread, physical c, the original
  periodic domains/options and N=8/16/32. Flags are -O0 -g (Debug), -O2 -DNDEBUG
  (Release), and -O2 -g -DNDEBUG (RelWithDebInfo). Exact options, output cadence,
  timesteps/subcycles, references and per-case bounds are in every run.json.
- These fixtures execute production methods with serial communication. They do
  not validate HPX/MPI message delivery, races or multi-locality regridding.
  The four-substep versus four separate calls test and direct epoch/ghost checks
  remain applicable only to their stated source modes, not arbitrary coupled hydro.
- Full application configuration still stops at missing Vc. HPX, CPPuddle and
  Silo development dependencies and an application executable are unavailable.
  CMake/CTest itself and Boost/FFTW headers are no longer the relevant blockers.
- All six hydro/gravity descriptors retain conditional status without a
  configured application build. Eight generated registration inventories cover
  every enabled test in their respective separate backend profiles, not actual
  CPU/GPU simulation execution or pre-/post-migration numerical equivalence.
  The descriptor called AMR covers the historical Sod-big scenario, not the
  separate unregistered amr_test. Simultaneous CUDA+HIP retains a pre-existing
  duplicate blast registration; no legacy names were quietly changed.
- Old `--exe` smoke runs remain conditional and distinct from `--build` CTest
  runs. Split and equals option spellings work. Requested CTest parallelism is
  one; authoritative application thread arguments are not overwritten.
- The four original-profile regression driver has no inherited accuracy gates.
  Running it successfully only establishes execution and metrics. The front's
  coarse-grid nonmonotonic error is explained below; no threshold was invented
  to certify it. The Gaussian-source equilibrium bulb is not an Ensman
  equilibrium sphere.
- The old test_m1.cpp asserts removed minmod/primitive-reconstruction/monolithic
  coupling behavior and an old CFL value. It remains unchanged. Its explicit
  historical build target fails with a migration message; the current algebra
  suite is separately named radiation.unit.m1.current, not an equivalence pass.
- Published shock parameters and model mismatches are in
  [ensman-reference-followup.md](ensman-reference-followup.md). The original
  purported Ensman equilibrium-sphere identity/reference remains unverified.
- Codec restart tests do not substitute for application Silo restart. ASan/UBSan
  passes use detect_leaks=0 because LeakSanitizer fails under tracing; leak
  detection remains unvalidated. No full oblique transport regression was run.

## Isolated front diagnostic discrepancy

At t=0.2, the restored original front fixture has cell-average L1 values
0.06845846, 0.03382730, 0.03517463 and 0.01870599 at N=8/16/32/64. Its exact
finite-volume reference is correct. An independent scalar reduction of the
streaming update, including harmonic PLM, the VL predictor and the actual
six-face positivity limiter, reproduces full production profiles to roundoff.

Continuous L1 errors of the piecewise-constant numerical profiles at N=8/16/32
are 0.13476368, 0.08817348 and 0.05062928: monotonically decreasing. Taking the
absolute value only *after* cut-cell averaging cancels 0.06630522, 0.05434617
and 0.01545464 of error respectively. Between N=16 and N=32 the loss of that
cancellation exceeds the actual profile-error improvement by 0.001347329,
exactly explaining the apparent reversal. At aligned-front t=0.25, production
N=16/32 errors are 0.07868664/0.05009426 with zero cut-cell cancellation.
The extra near-zero terminal step changes the profile only at roundoff.

No reconstruction/reference/timestep fix or tolerance change is justified by
this discrepancy. This explains the particular coarse-grid reversal, not a
general accuracy certification of the original four-profile metric-only suite.

## Reproduction

Apply the supplied patch to the preserved Step 07 commit. Use fresh output paths:

```bash
python3 verification_results/runner.py suite all 0 1 2 Release \
  --threads 1 --output /absolute/results/Release
python3 verification_results/runner.py suite all 0 1 2 Debug \
  --threads 1 --output /absolute/results/Debug
python3 verification_results/runner.py suite all 0 1 2 RelWithDebInfo \
  --threads 1 --output /absolute/results/RelWithDebInfo
python3 -m verification_results.audit_artifacts /absolute/results --decode
```

Without the application the complete suite returns 3 (conditional), not zero.
To execute configured hydro/gravity checks use `suite hydro Release --build
/absolute/application-build --ctest /absolute/ctest --output /fresh/results`.
Existing outputs are refused and raw solver data are archived and hashed before
registered cleanup. The generated portable report contains numerical/reference,
signed-error and convergence plots, movies, raw data, logs and metadata.

Decision remains **NO-GO for full physics acceptance** until the listed core
application/reference/distributed/restart cases have direct passing evidence.
