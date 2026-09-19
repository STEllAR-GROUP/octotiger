# Step 07 physics validation — NO-GO

This is a validation handoff, not completion of physics acceptance. Core
hydro/gravity regressions, published Ensman tests, distributed boundary delivery,
and application restart have not been validated. Do not advance on a claimed pass.

## Provenance

Started from clean Step 06 `95f722b114ed01491d68c4282aa8f6e53c1bf9d1`.
The step-06 archive SHA-256 is
`bf47dbd03cc147fc5359ff153d7afd97268f78e1ce0ca28410a6d8b9c680b44b`.
Its 441 regular files match that commit (including SOURCE_VERSION export expansion).
All 22 descriptors are present: hydro 4, gravity 2, radiation 16.
Step 05 to Step 06 changed no `src/`, `octotiger/`, or `test_problems/` files.
Step 07 likewise changes no files in those directories. Numerical equations,
opacity behavior, defaults, hydro timesteps and numerical tolerances are unchanged.

The final handoff commit is recorded in the accompanying generated assessment.
Matrix run metadata honestly records the Step 06 parent plus dirty=true because
the fixes were validated before committing. Per-run source-file, generated-C++,
executable and descriptor hashes are preserved. The assessment checks that all
recorded runtime source hashes and native descriptors still match the handoff
tree; it does not relabel those runs as clean-commit runs.

## Executed matrix and interpretation

GCC 13.3.0, C++23 for production-method fixtures; C++20 for standalone unit tests.
Debug: `-O0 -g`; Release: `-O2 -DNDEBUG`; RelWithDebInfo: `-O2 -g -DNDEBUG`.
Each matrix uses one thread and fresh build directories. Native descriptor levels
0/1/2 mean N=8/16/32 per dimension, on a periodic cube of side 6e10 cm with
physical c=2.99792458e10 cm/s. Every test retains its individual final time,
source mode and parameters in its descriptor and run.json. CLI override levels
3..5 are not certified by this matrix. Hydro/gravity inputs retain their actual
fixed levels 1/3/4; radiation resolution arguments do not override them.

| Native radiation case | Status | Final observed order | Required minimum |
| --- | --- | ---: | ---: |
| Boundary/subcycle diagnostic | Pass, serial fixture only | roundoff | none |
| Diffusion, chi L=100 | Pass | 2.298956 | 0.7 |
| Moving pure scattering | Pass | 0.927055 | informational |
| Static diffusion, chi L=30 | Pass | 2.688585 | 0.7 |
| Streaming front, 1D | Pass | 0.681175 | 0.35 |
| Streaming wave, 1D | Pass | 1.924280 | 1.6 |
| Piecewise-constant wave | Pass | 0.885171 | 0.65 |
| Thermal relaxation | Pass | 1.001324 | informational |
| Thin Gaussian beam | Pass | 1.638262 | 1.3 |
| S&O-style damped wave | Historical convergence gate fails | 0.999614 | 1.6 |

Final values and cross-build comparisons are generated from completed results,
not inferred from this table. The four old application S&O descriptors and two
Ensman descriptors are conditional. All six hydro/gravity descriptors are
conditional. Each completed build therefore has 9 passes, 1 failed gate and
12 conditional cases; the unified command returns 1, never success.
These are the recorded execution statuses, not certification of retained raw
artifacts: the subsequent audit also found a truncated Release thermal/l2 history
and an empty Debug thermal/l1 sample file with a truncated history and empty log.
The original files are retained and the report lists these as artifact failures.
Separate clean Debug and Release thermal reruns at all three levels pass and
produce complete samples/histories. Their success does not erase the original
integrity failures; the cause of that persistence failure remains unresolved.
The RelWithDebInfo boundary_subcycles/l2 MP4 is also incomplete (48 bytes,
missing its moov atom) despite the earlier encoder success log. It is a failed
visual product, not an acceptable movie; the artifact audit records it explicitly.

Unchanged tolerances: normalized L1 <= 2/N (front <= 4/N), except the PC control
which checks its independent discrete Fourier update at absolute 5e-12.
Reduced flux <= 1+2e-12; symmetric streaming |F/c-E| <= 2e-12;
wave phase error <= 2 pi/N. Energy/momentum budgets use 2e-11 with the existing
per-case normalization. Moving/scattering energy checks normalize by total
energy, which can include a large gas reservoir; source-cell tests provide
additional conservation evidence but do not establish uniform relative accuracy
for every small radiation energy change. Diffusion amplitude tolerance is
2/N + 2(k/chi)^2 against the diffusion asymptote; the profile reference retains
finite light speed. No asymptotic-preserving unresolved-cell claim is made.

The damped-wave equation has an exact uniform-mean numerical solution
E_n = E_0 product_j [1+c chi_a dt_j]^-1 for theta=1. At N=8/16/32 its means are
0.9049102311526761, 0.9048750994474135, 0.9048562637770617; the continuum mean is
0.9048374180359595. Independent amplification matches the measured means to
roundoff. First-order damping error is expected for backward Euler and dominates
the 1e-6 perturbation. The order-1.6 gate was incorrectly specified in Step 05;
its failure is a test-policy mismatch, not evidence of unexpected solver order.
The threshold and solver were deliberately left unchanged in final validation.

28 readable Step 05 sample files match Release numerically exactly. Two older
scratch baseline CSVs (moving_scattering/l1 and streaming_wave_1d/l1) are truncated;
those two raw comparisons are unavailable. The old compact convergence results
agree. Release and RelWithDebInfo sample CSVs are bitwise identical. The generated
assessment excludes incomplete raw runs from its cross-build comparisons and
checks Debug separately. This does not establish any hydro/gravity
pre-migration numerical equivalence.

## Additional gates

- 27 unified harness tests pass, including wrong counters/times/ghost mutation,
  encoder failure, missing executables, partial checks, family status aggregation,
  isolated output directories, read-only plans and legacy radiation wrapper routing.
- Seven legacy-option source-contract tests pass. The actual Boost parser test
  cannot compile: boost/program_options.hpp is absent. Full application argument
  parsing/effective-value equivalence remains conditional, not a demonstrated pass.
- M1 algebra: 175509 checks per build mode. Grey opacity units, positivity,
  pure absorption/scattering, vacuum, thin/thick limits, equal-opacity behavior,
  legacy/explicit S&O equivalence, codec validation and old-checkpoint defaults pass.
  FPE guard tests pass in all three modes.
- Existing S&O source/driver suite passes before changes, in Release, and under
  AddressSanitizer/UBSan; stdout is identical. LeakSanitizer itself fails under
  this environment's tracing, so the successful sanitizer rerun explicitly uses
  ASAN_OPTIONS=detect_leaks=0. Leak checking is unvalidated.
- New physical-c subcycling comparison passes all three builds: one four-substep
  driver call versus four separate CFL-sized calls, for chi_s=0 and 0.7 cm^-1,
  c_hat/c=1 and 0.03. Radiation/gas states agree within 3e-12 scaled tolerance.
  One subcycled material step refreshes hydro once; independent steps refresh
  it four times. Radiation boundary counts are respectively 5 and 8. This
  comparison uses fixed-density, velocity-terms-off scattering/source-free data;
  it is not a proof for nonlinear thermal splitting or evolving hydro.
- Production conservation collector passes leaf ownership, interval draining,
  regrid ledger, duplicate-time rejection, final sampling and FPE scope checks.
- Direct epoch/state traces validate production compute_radiation/all_rad_bounds
  sequencing using serial communication endpoints. HPX actions, remote delivery,
  races and multi-locality regridding are not exercised. The inherited M1 tests
  check oblique beam algebra, but a full oblique transport regression is unrun.
- Opacity checkpoint codec/override tests pass; Silo file restart plus continued
  application evolution is unrun. These are different validation claims.

## Discovered implementation defects and remaining blockers

Fixed with targeted harness tests: mixed-family output collisions and lost return
codes; source-directory solver output; recorded-but-unused thread/build settings;
scenario regex failures mislabeled passed; empty-check smoke runs mislabeled
regression passes; incomplete plan output; descriptor schema/level metadata;
sphere regex typo (restored exact CTest pattern); an unsubstantiated rotating-star
stop-step override (removed to preserve its legacy config).

The scenario adapter is still not an equivalent migration of all legacy CTests.
It omits Silo comparisons, many field regexes, GPU/VC/Kokkos variants and the
rotating-star input generator. Successful smoke runs therefore remain conditional.
The descriptor named AMR is sod_big.ini, not the separate amr_test configuration.
These gaps invalidate Step 06's earlier completion claim. Original CTest files
are unchanged, but their runtime equivalence has not been demonstrated here.
The old test_sod.sh contains invalid shell assignments (`$OCTOTIGER=$1`) already
present in Step 06; preserving the file did not establish a usable entry point.

Four stale auxiliary radiation drivers fail independently of the new matrix:
tests/validateRadiationGrid.py has outdated fixture fields/interfaces;
tests/radiation/test_grid.py and test_regression_kernels.py extract old method
names; tests/radiation/test_comparator.py imports missing
test_problems/radiation/check.py. Raw error logs are retained. No dummy wrappers
or substituted tests were used to turn these failures into passes.

CMake/CTest, the built Octo-TIGER application, HPX/Silo development dependencies,
Boost headers, FFTW headers and the application visualization stack are absent.
The Ensman sphere/shock descriptors lack verified published definitions/reference
data and are unimplemented. An existing Gaussian-source diffusion bulb is not
an accepted substitute for an Ensman sphere.

The native artifacts include numerical/reference and signed-error PNGs, nine-frame
MP4s and convergence PNGs. The assessment checks cadence, sample counts, metadata,
movie decoding/frame counts and retained products. There are no hydro/gravity
plots or movies because no corresponding simulation was run. The common index
links all families; portable radiation reports embed per-resolution data, logs,
metadata and movies. No external website was deployed.

## Reproduce and follow up

From HOME, using the actual checkout path in OCTO_SOURCE:

```bash
cd "$HOME"
OCTO_SOURCE="$HOME/octotiger/src/octotiger"
python3 "$OCTO_SOURCE/verification_results/runner.py" suite all 0 1 2 Release \
  --threads 1 --output "$HOME/step07-release"
```

Repeat with Debug and RelWithDebInfo and distinct output directories. Exit 1 is
the recorded NO-GO result. Scenario-only commands accept --root/--build/--exe;
their result remains conditional until legacy CTest equivalence is established.
Run tests/validateFinalRadiation.py for the subcycle reference comparison.
The reusable final_assessment.py audits the three completed mode directories and
creates the portable report without rerunning physics.

Required follow-up: restore the full build/test toolchain and stale entry points;
finish CTest/Silo/kernel-variant migration and actual before/after runs; verify
published Ensman definitions and implement the missing tests; exercise distributed
boundaries and file restart; review the damped-wave first-order acceptance policy
as an explicit test-definition correction. No broad physics change is justified
by the present evidence, and no ULTRA escalation is needed for the resolved
damping discrepancy.
