# White-dwarf equilibrium regression

This case evolves a cold, nonrotating, one-solar-mass white dwarf for three
timesteps with the corrected WD sound speed. The `STAR + WD` initializer uses
the tabulated hydrostatic model implemented in
`src/test_problems/wd_equilibrium.cpp`. Its code units normalize the stellar
mass and radius to one. The model supplies the corresponding physical units
and WD EOS constants.

The grid8 configuration uses `min_level=1`, `max_level=4`, and `xscale=2.56`.
The finest cell width is

```text
dx = 2 * xscale / (INX * 2^max_level) = 0.04
diameter / dx = 2 / 0.04 = 50
```

Refinement follows the existing stellar density criterion with
`refinement_floor=1e-5`. Uniform refinement and the rotating-star spatial
criterion are disabled; the atmosphere has density `1e-10`. Thus “50 cells
across the diameter” specifies the finest spatial resolution, rather than a
uniform grid over the full domain. The application uses gravity and the legacy
CPU hydro/FMM kernels, with two HPX worker threads and no GPU executors.
The cold thermal tracer starts at zero. A `tau_floor=1e-30` prevents roundoff
below zero during evolution; its thermal energy density is only `1e-50` in
code units and does not change the hydrostatic model.

`stop_step=2` is inclusive in this solver and therefore executes three
timesteps. The run writes the gravity-initialized snapshot `X.0.silo` and the
evolved `final.silo`, together with their `.data/0.silo` files. Stock analytic
comparisons and diagnostics are disabled. This short regression checks the
initial model and reproducible evolution; it is not a long-time stability or
resolution-convergence test.

## Running the checks

Configure a grid8 build with `OCTOTIGER_WITH_TESTS=ON` and the matching
`octotiger-testdata` checkout, or pass `OCTOTIGER_TESTDATA_DIR` explicitly.
Build `octotiger` and `wd_equilibrium_model_test`, then run:

```sh
ctest --test-dir BUILD_DIRECTORY -R wd_equilibrium --output-on-failure -j2
```

The model test compiles the actual production model source. The application
fixture starts from an empty `test_problems/wd_equilibrium/run_grid8` directory
inside the build directory. Its outputs and `stdout.log`/`stderr.log` remain
there until the next run for inspection. Selecting the `.diff` or `.physics`
test through CTest also runs its required simulation fixture.

The `.diff` check compares `final.silo.data/0.silo` with
`wd_equilibrium_1msun_grid8.silo` in the reference repository, using the existing
Silo browser tolerances and the shared failure pattern. Every physical field
and the mesh/object inventory remain in scope. Only `idle_rate`, `locality`,
`/_hdf5libinfo`, and `/_silolibinfo` are excluded as runtime/library metadata.

When Python 3 and NumPy are available, `.physics` also runs `validate.py` on
the initial and final snapshots and writes `metrics.json`. It checks the
physical mass, finest resolution, coexistence of coarse and fine leaves,
and short-time equilibrium diagnostics. The physical stellar radius comes
from `code_to_cm` in the master SILO metadata for this model's unit code
radius. These checks require a loadable shared Silo library.

Conserving gas plus potential energy can produce negative stored `egas` in
surface and atmosphere cells. The EOS uses the auxiliary thermal variable
when subtracting kinetic and cold energy is unreliable; WD pressure also
includes the cold degenerate contribution. The physical check therefore
verifies finite fields, positive density and EOS pressure, and a nonnegative
thermal tracer. Negative conserved gas energy is recorded as a diagnostic
and is not itself a failure condition.

## Reference data

The reference is the single-locality raw data file
`run_grid8/final.silo.data/0.silo`, copied to the independent
`octotiger-testdata` repository as `wd_equilibrium_1msun_grid8.silo`. Keep
provenance there alongside it: source commit/diff, compiler and dependencies,
build options, configuration, exact run command, reference hash, model mass
and radius, and the measured physical diagnostics. Generate it with the
corrected EOS and validate a separate repeat run before recording the new
submodule commit in Octo-Tiger.

The application regression is registered only for CPU-only grid8 builds. A
grid16 run at `max_level=3` would have the same finest cell width, but a
different block hierarchy and SILO inventory. It needs its own generated and
validated reference before being enabled. The model unit test remains
available for other block sizes and GPU builds.
