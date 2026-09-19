# Unified verification harness design (Step 02)

`verification_results` is the permanent test-description, execution, result,
and reporting boundary for hydro, gravity, and radiation.  Step 02 introduces
that boundary and adapts the current radiation workflow; it intentionally does
not rename or move `radiation_results`, alter a solver, or reinterpret a test.

## Compatibility baseline

The existing `radiation_results` C++20 program remains authoritative for its
four cases.  `run.sh`, `run_live.sh`, and `results.sh` are unchanged, including
their aliases, positional levels/build, resume rules, CGS conversion, reference
generation, validation, plots, movies, VisIt sessions, live web pages, and
output formats.  In particular, existing commands continue to write beneath
`radiation_results/results` unless the caller chooses `--output`.

The current workflow was inventoried as follows:

| Concern | Existing implementation retained by the adapter |
|---|---|
| Entry points | `run.sh` selects `results.sh run`; `run_live.sh` selects `results.sh live`; `results.sh` builds and invokes `.build/radiation-results` |
| Test inputs | Four INI files in `radiation_results/configs`; the runner scales them to CGS and writes effective `run.ini` |
| Execution | C++ `run.cpp` selects cases/levels/build, rebuilds `octotiger`, runs HPX threads, validates completion, and supports resume |
| References | Exact profile cell averages for wave/front/sphere; FFTW telegraph generator and `reference.bin` for Gaussian |
| Diagnostics | `L1.dat`, `L2.dat`, `Linf.dat`, slice CSVs, conservation CSV, log validation, and optional Silo states |
| Plots | C++ readers plus gnuplot emit paired numerical/reference/error products, profiles, convergence, and conservation summaries |
| Movies | C++ Silo reader/VisIt session generation, VisIt frames, FFmpeg H.264 encoding, and render metadata |
| Website | Static batch index, per-problem pages, combined plot report, movie gallery, and live JSON refresh state |
| Metadata | `batch.json`, per-run `run.json`, `live.json`, plot JSON/CSV, movie/render JSON, source and executable SHA-256 values |
| Tests | C++ unit, integration, and terminal regressions plus the retained Python conservation/serial/SO validators |

This inventory is the behavioral contract for the mechanical migration in Step
03.  Historical generated `.build` files are removed from the source tree.  The
unchanged `results.sh` entry point rebuilds them when dependencies are present,
and ignore rules keep those products out of future commits and source archives.

## Descriptor contract

Every test is a versioned `test.json` validated against
`descriptor.schema.json`.  Its stable identity is `family.suite.name`.  The
descriptor records:

- physical family, test name, regime, dimensionality, resolution levels, build
  type, and required executable;
- input parameters and expected diagnostics;
- reference kind, owner, and provenance description;
- tolerance mode and its owner; and
- requested visualization products and the temporary compatibility adapter.

Descriptors say what a test means.  Adapters translate that declaration to an
existing executable; adapters may not implement physics.  Four
`radiation.skinner_ostriker.*` descriptors provide the first adapter coverage.
The Ensman location is reserved but deliberately has no invented descriptor.

## Runner interface

The common launcher works from any current directory:

```sh
verification_results/run.sh list
verification_results/run.sh plan radiation.skinner_ostriker.streaming_wave 2 3 --build Release --threads 8
verification_results/run.sh run radiation.skinner_ostriker.streaming_wave 2 3 Release --threads 8
verification_results/run.sh live radiation 2 3 4 Debug --threads 12 --no-open
```

`run` and `live` accept the legacy radiation runner's positional resolution
sweep/build syntax and pass its options through unchanged.  Selectors may be a
single stable ID, the `radiation` or `radiation.skinner_ostriker` family, or a
legacy case alias.  `plan` emits the stable manifest without executing.  A
normal common run defaults to a new ignored directory beneath
`verification_results/results`; `--output` and `--resume` remain available.
`--dry-run` is delegated to the existing runner, so its own configuration and
build checks remain authoritative.

With no explicit levels/build, common metadata records the exact legacy
defaults: `run` selects levels 2,3 and Release; `live` selects levels 2,3,4 and
Debug.  Descriptor levels are the recommended sweep, not an implicit override
of adapter behavior.

After a successful adapted run, `verification.json` supplements—never replaces
or rewrites—the legacy `batch.json` and `run.json`.  Direct invocations of all
old scripts behave exactly as before.

## Stable metadata

`verification.json` schema version 1 records the source commit and dirty flag,
descriptor path and checksum, exact requested parameters and arguments,
compiler/build directory and build type, levels, thread count, timestep/output
controls, executable requirement, reference/tolerance policy, adapter command,
and artifact locations.  Legacy metadata continues to record effective
configuration, executable/source hashes, cell counts, physical units, capture
cadence, norms, and conservation.  Together these are the provenance record for
every report product.  Step 03 will make plot/movie/web metadata embed or link
the common manifest ID when their code moves; no historical format changes in
Step 02.

Metadata is descriptive, not a cache key.  A dirty source tree is recorded and
visible rather than silently presented as the named commit.  Source commit,
descriptor checksum, effective `run.ini`, legacy executable hash, and generated
reference checksum provide the reproducibility chain.

## Artifact and source boundaries

| Content | Location and rule |
|---|---|
| Test definitions | `tests/<family>/<suite>/<test>/test.json`; small, reviewed, tracked |
| Reference inputs | `references/`; tracked only with provenance/checksum/review |
| Common code | `runner.py`, `adapters/`, later `cpp/`, `web/`, `plots/`, and `movies/` definitions |
| Numerical data and logs | `results/<run>/`; ignored and never consumed as a descriptor/reference |
| Rendered plots, movies, pages | Inside the same immutable run artifact root |
| Temporary builds/logs | `build/`, `tmp/`, and `logs/`; ignored |
| Legacy outputs | Remain under `radiation_results/results` and retain their existing rules |

The legacy `.build`, `results`, and `visit_sessions` directories are ignored as
generated products.  They are not included in source archives.

The runner rejects outputs beneath descriptor or reference directories.  Git
ignore rules prevent new result, temporary, log, build, and Python cache data
from silently appearing as changes.  Promotion of any generated result to a
regression reference is a separate reviewed change with origin run, source
commit, executable hash, descriptor checksum, selection rationale, and an
explicit new reference identifier.

## Reference ownership and tolerances

Four reference classes remain distinct:

| Kind | Owner | Acceptance rule |
|---|---|---|
| Analytic | Test/profile or reviewed generator source | Error norm and convergence policy owned by the descriptor/solver CTest; formula changes require review |
| Published | Vendored digitization plus citation/license/checksum | Declared comparison variables and combined publication/digitization tolerance |
| Regression | A reviewed, immutable baseline linked to full provenance | Explicit absolute/relative tolerances; never overwrite in place |
| Qualitative | Named visualization product and reviewer expectation | Informational unless a separate numeric diagnostic exists; no visual “pass” masquerades as a norm |

The adapter preserves radiation's current policy: reports expose all errors and
orders, while solver CTests retain pass/fail thresholds.  Reference generation
and result generation are different roles and different directories.

## Step 03 migration map

Step 03 can migrate mechanically in this order: copy common C++ utilities into
`cpp`; move test definitions/config ownership behind descriptors; relocate
reusable plot/movie/web definitions; make the common runner native; then turn
the old scripts into wrappers.  At each stage golden command-plan and artifact
layout tests must compare the old and new paths before ownership switches.

Hydro and gravity adapters follow the same seam: describe an existing test,
invoke its existing executable/configuration, normalize diagnostic metadata,
and leave numerical source untouched.  Algorithm changes, new opacity,
reconstruction, coupling, or subcycling are outside the harness migration.
