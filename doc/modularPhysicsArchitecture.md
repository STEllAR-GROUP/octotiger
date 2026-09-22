# Modular mesh and physics architecture

## Decision

The new implementation separates mesh topology, finite-volume mechanics, and
physics systems into ordinary C++ libraries. HPX remains responsible for
distribution and scheduling; it is not a dependency of the numerical kernels.
Each `node_server` owns one `octotiger::Subgrid`. A subgrid has one mesh layout
and optional hydro, radiation, and gravity field sets.

New C++ names follow the project rule used for this work: types use
`UpperCamelCase`; functions, variables, and data members use `lowerCamelCase`.
Private class data members add a trailing underscore to that lower-camel-case
base name (for example, `dimensionCount_` and `hydroData_`); parameters and
local variables do not. External compatibility keys such as `mesh.ndim` keep
their established spelling.

| Library | Responsibility |
| --- | --- |
| `OctoTiger::Mesh` | Dimensional indexing, patch time, AMR topology, shadow hierarchy |
| `OctoTiger::Physics` | Reconstruction, boundary policies, unsplit finite-volume update |
| `OctoTiger::Hydro` | Ideal-gas conserved/primitive states, HLLC, positivity |
| `OctoTiger::Radiation` | M1 closure, HLL transport, realizability |
| `OctoTiger::Gravity` | Gravity field ownership and the three-dimensional guard |
| `OctoTiger::Subgrid` | Optional field composition, runtime dimensional dispatch, leaf error |

`grid` and the Unitiger update remain the default legacy execution path. The
explicit `runtime.modular.enabled=on` path (also selected by the earlier
`hydro.modular.enabled=on` switch) creates actual HPX `node_server`
components, owns populated `Subgrid` fields, and advances them through
`node_client` actions. It does not create or update legacy `grid` storage.
Both hydro and source-free M1 radiation are connected to this distributed
path. Gravity, gas-radiation coupling, radiation sources/opacities, and legacy
checkpoints have NOT been migrated and are rejected. New transport code must
not add dependencies on `grid`.

## Distributed transport path and current limits

The frontend selects the modular driver before creating the legacy tree. A
dimension-aware `MeshHierarchy` supplies 2/4/8-child topology. Each physical
leaf and each independent shadow leaf is an ordinary `node_server` component
placed round-robin on the available HPX localities. Each component owns exactly
one `Subgrid`; there is no alternative hydro-specific component type.
The same component can enable hydro, radiation, or both field sets. Radiation
does not allocate a second Subgrid or a legacy `rad_grid`.

Each synchronous step has explicit completion barriers:

1. Gather immutable old-time snapshots from node actions.
2. Reduce all enabled hydro/radiation physical and shadow CFL limits to a shared timestep.
3. Send the appropriate hierarchy's snapshot directory to each advance action.
4. Gather physical-time-tagged face fluxes and apply coarse-side reflux actions.
5. Gather corrected end-time interiors and refresh all end-time halos.
6. Compare each physical leaf with its immediate shadow parent at the same time.

All halo cells, including transverse predictor corners, are communicated. The
solver produces only face fluxes; that does not remove its need for corner
states. Same-level neighboring patches see identical old states and timestep.
Fine-to-coarse halos use conservative volume overlap; coarse-to-fine halos use
piecewise-constant prolongation (currently first-order at AMR interfaces).
Coarse-face reflux replaces the coarse flux by area-averaged fine flux over
the matching physical interval. Invalid time tags or coverage are errors.
An inadmissible reflux state stops the run; timestep retry is not implemented.

Transport is a correctness-first centralized snapshot prototype: full
directories are sent to every patch, giving quadratic communication/storage
growth with patch count. It is not yet a scalable nearest-neighbor transport.
There is no load balancing or dynamic error-driven regridding. Equal minimum
and maximum levels create a uniform tiled mesh; if maximum is larger, only
the lower-corner branch is refined beyond the uniform minimum. Error values
are available separately for every physical leaf but do not yet change topology.

The independent shadow directory is initialized analytically and advanced
through the same actions, with its own halos and reflux, using
`oneLevelCoarser()`. If an immediate parent is internal in that hierarchy, its
comparison state is conservatively assembled from evolved **shadow** leaves,
including fresh shadow halos. Physical leaves are never donors to this operation.

The driver supports ideal-gas hydro (`sod`, `advection`, `kelvinHelmholtz`) and
source-free M1 radiation (`streamingGaussian`, `isotropicPulse`), periodic/outflow
boundaries, synchronous stepping, and dimension-correct Silo visualization
output. It requires `problem.name=NONE`; initial conditions are selected by
`hydro.modular.problem` and `radiation.modular.problem`. Rotation, gravity,
nonideal hydro EOS, reflecting/inflow boundaries and restart are rejected. Silo output is
visualization only, not a restart checkpoint. Legacy source/species physics,
legacy diagnostics, regrid controls, and performance-kernel selections are not
used. The numerical scheme is Van Leer MUSCL-Hancock with its own small
positivity floors; a nondefault legacy density-floor request is rejected.
The legacy `dt_max` means a fractional-state-change limit, not a timestep cap;
it is not repurposed here, and a nondefault request is rejected. This driver
uses enabled hydro/radiation CFLs and output/final synchronization times. Legacy diagnostics
are not produced; progress and the maximum per-leaf shadow error are printed.
Error tolerances currently use absolute `1e-8` and relative `1e-2` per field.

Run the independent 2D Kelvin--Helmholtz case after building the application:

```sh
./build/octotiger --runtime.config_file=test_problems/kelvinHelmholtz/kelvinHelmholtz.ini
```

`kelvinHelmholtzOutput/modular_000000.silo` is the initial frame; later files
are written at `output.interval` and at the final achieved time. Files carry
cycle/time metadata and dimensional multimeshes; existing filenames are never
overwritten. Use a fresh `output.directory` for repeat runs. The mesh has
`INX * 2^level` zones per active axis. The provided level-3 run is 64-square
when compiled with `INX=8`.

## Radiation runtime and shared Silo output

Run the source-free radiation cases through the normal executable (replace
`./build/octotiger` with the executable path from your configured build):

```sh
./build/octotiger --runtime.config_file=test_problems/modularRadiation/streaming1D.ini
./build/octotiger --runtime.config_file=test_problems/modularRadiation/streaming2D.ini
./build/octotiger --runtime.config_file=test_problems/modularRadiation/streaming3D.ini
```

These cases use cgs coordinates `[-3e10,3e10]` cm and evolve for `0.8` s.
The Gaussian is centered at the lower quarter of each active axis and moves
in the normalized positive-axis diagonal. `radiation.test.width`, `.background`,
and `.amplitude` configure its initial energy density. `isotropicPulse` uses
the same energy profile with initially zero flux. Radiation-only mode explicitly
sets `hydro.enabled=off`; turning hydro on independently advances its field set
on the same mesh and timestep, **without any gas-radiation coupling**.

Source-free mode must be requested explicitly:

- `radiation.modular.source_free=on`;
- `radiation.opacity.model=legacy`, `radiation.opacity.constant=0`;
- `radiation.implicit=off`, `radiation.velocity_terms=off`, `radiation.subcycling=off`;
- cgs unit conversions (`units.centimeters`, `units.seconds`, `units.grams`) equal to one.

Nonzero opacities, alternate opacity models, implicit exchange, velocity terms,
nondefault legacy energy-source modes, and source coupling are rejected, not
silently approximated. No source-term solver or radiation temporal subcycling
is connected to this runtime path. Only the Gaussian width/background/amplitude
legacy radiation test parameters are consumed; prescribed-medium extinction,
luminosity, and reference-file test drivers are not used.

Internal radiation state is `(E,Q=F/c)`, where physical
`c=2.99792458e10 cm/s`. The transport kernel uses
`cHat=radiation.reduced_light_speed_ratio*c`. The ratio changes the transport
speed, **not the conversion from Q to physical output flux**.

`OctoTiger::Output` exposes the single shared `writeModularSilo` writer used by
hydro-only, radiation-only, and combined snapshots. It has no HPX or options
dependency; each snapshot supplies its geometry, selected fields, and time.

| Enabled fields | Silo zone-centered variables |
| --- | --- |
| Hydro | `density`, `momentumX`, `momentumY`, `momentumZ`, `energy` |
| Radiation | `radiationEnergy`, `radiationFluxX`, `radiationFluxY`, `radiationFluxZ` |
| Both | Both sets on the same dimensional mesh |

The writer outputs radiation energy in erg/cm³ and physical flux `F=cQ` in
erg/(cm² s), including zero inactive components for these initial conditions.
There are no Q variables mislabeled as F. Every dimensional block has
cycle/time metadata and is referenced by root-level multimesh/multivars.
Shadow fields are not written. Runtime Silo execution/readback is still pending
on a host with Silo and HPX installed; the wiring is not a claim of validation.

## Dimensional mesh

`mesh.ndim` accepts 1, 2, or 3 and defaults to 3. The mesh allocates only active
dimensions:

| Dimension | Interior extents | Storage with ghost width `g` |
| --- | --- | --- |
| 1-D | `N × 1 × 1` | `(N + 2g) × 1 × 1` |
| 2-D | `N × N × 1` | `(N + 2g) × (N + 2g) × 1` |
| 3-D | `N × N × N` | `(N + 2g) × (N + 2g) × (N + 2g)` |

The x coordinate is contiguous. Inactive coordinates and inactive lower bounds
are exactly zero. Faces exist only for active normals. Refinement creates 2, 4,
or 8 children in 1-D, 2-D, or 3-D. Gravity is rejected unless `mesh.ndim=3`.
The legacy driver rejects `mesh.ndim<3`; it cannot silently run a 3D tree for
a lower-dimensional request.

## Hydro and radiation transport

The common spatial integrator is a dimensionally unsplit MUSCL-Hancock method:

1. Fill or receive the time-tagged ghost state.
2. Reconstruct direction-specific face states with PLM limiting.
3. Apply one half-step predictor containing flux divergence from every active
   direction.
4. Solve one Riemann problem per face.
5. Apply the conservative divergence of those face fluxes.

Only face-centered fluxes are produced. There are no corner fluxes or the
legacy multidimensional Unitiger reconstruction structures. Each completed
step returns its face arrays and physical `TimeInterval`, so the AMR driver can
reflux without recomputing fluxes.

Hydro uses primitive reconstruction and an HLLC Riemann solver with an HLL
fallback. A convex face-flux limiter preserves positive density and pressure.

Radiation transports `(E, Q)` with `Q=F/c`, uses the M1 closure and the
Skinner--Ostriker HLL wave speeds, and applies a convex realizability limiter.
Physical storage adapters retain `(E, F)` units.

Skinner and Ostriker's Athena radiation implementation used the VL integrator,
which is a MUSCL-Hancock variation. Athena hydro can also run with that VL
integrator (or its CTU integrator). Thus the shared MUSCL-Hancock scaffold is a
sound hydro choice, while HLLC supplies the gas-specific Riemann solve.

## Independently evolved shadow hierarchy

The refinement estimate never restricts current children into a parent and
calls that the coarse solution. Instead:

- Every node can retain persistent hydro, radiation, and/or gravity shadow fields.
- A shadow is copied once at a common synchronization time and cannot be
  overwritten by later fine data.
- Shadow patches are advanced by their own transport calls, or independently
  recomputed by the coarser gravity solve.
- `MeshHierarchy::oneLevelCoarser()` derives the complete shifted AMR topology.
- The synchronized shadow driver prepares every boundary from one time-level
  snapshot before advancing any shadow patch.
- A fine leaf is compared at the same physical time with a limited linear
  prolongation of its immediate parent's independently evolved shadow.

For each enabled field `f`, the normalized error is

\[
e_f = \frac{|U_f^{\mathrm{fine}}-P(U_f^{\mathrm{shadow\ parent}})|}
{a_f+r_f\max(|U_f^{\mathrm{fine}}|,|P(U_f^{\mathrm{shadow\ parent}})|)}.
\]

The result reports the maximum normalized error plus maximum and mean absolute
errors per field. The root has no parent error. Layout, bounds, cell width, and
physical time must match before a comparison is accepted.

## Future refinement in time

Time refinement is not implemented in this change, but the data and interfaces
do not assume a global cycle:

- Every `PatchData` owns a `TimeState`, including time, step size, temporal
  level, and substep.
- Boundary updates are requested at a physical time.
- Flux registers receive a physical `TimeInterval` with every face array.
- Shadow patches can choose their own stable step and advance to a requested
  synchronization time.

A later subcycling driver can therefore interpolate coarse boundary states and
accumulate reflux contributions over matching physical intervals without
changing the mesh or solver storage contracts.

## Verification

`modularPhysicsTests` covers exact 1-D/2-D/3-D allocations and child counts,
runtime dimension dispatch, face-flux sizes, AMR topology shifting, and the
non-overwrite shadow invariant. Its physics problems are:

- 1-D Sod shock tube for hydro;
- 2-D periodic Kelvin--Helmholtz instability for hydro;
- 1-D periodic free-streaming Gaussian for M1 radiation;
- 2-D periodic diagonal free-streaming Gaussian for M1 radiation.

The tests check conservation, positivity or realizability, expected solution
features, and a leaf error against an evolved (not restricted) parent shadow.

`hydroExchangeTests` runs the exact HPX-independent exchange/reflux helpers
called by the new node actions. It checks nonuniform tiled-vs-single-patch
equivalence, all-five-field coarse/fine periodic conservation, corner halos,
physical-time and coverage rejection, and independent shadow synthesis in
1D/2D/3D. These tests are NOT evidence of successful HPX serialization,
multi-locality execution, or Silo integration. The current environment lacks
CMake, HPX and Silo headers/libraries; those application-level checks remain
required on a configured build host before this path is production-ready.

The numerical libraries are defined once in `cmake/modularLibraries.cmake`,
shared by the full application and the dependency-free test build:

```sh
cmake -S tests -B buildTests
cmake --build buildTests --target modularPhysicsTests hydroExchangeTests radiationExchangeTests
ctest --test-dir buildTests -R '^(modularPhysics|hydroExchange|radiationExchange)$' --output-on-failure
```

For an already configured full application build, enable the opt-in runtime
tests with `-DOCTOTIGER_WITH_TESTS=ON
-DOCTOTIGER_WITH_MODULAR_RUNTIME_TESTS=ON`. After building, run:

```sh
ctest --test-dir build -R '^(modularSilo|modular(Radiation|Mixed)?Runtime[123]D)$' --output-on-failure
```

These tests exercise actual HPX components, a mixed-level advection mesh, and
successful initial/final Silo emission in unique output directories. They do
not inspect Silo readback or exercise multiple HPX localities. A configured
host must also verify dimensional multimesh/variable metadata and visualization,
then repeat the modular problem with its normal two-locality HPX launcher.
None of these application tests were run in the current environment.
In particular, Silo creation **and readback in each of 1D, 2D, and 3D** remain
pending, as do single- and multi-locality action/serialization runs.

The radiation exchange suite exercises the same shared transport-exchange
implementation as hydro, including 1D/2D/3D tiled equivalence, free-streaming
realizability, coarse/fine conservation, independent radiation shadows, and
reduced-speed versus physical-flux conversion. Application smoke tests and
shared-writer Silo roundtrip tests are supplied for a configured build host;
they are not substitutes for the standalone numerical tests, or vice versa.
`modularSilo` is the nine-case real-Silo roundtrip test (hydro-only,
radiation-only, and both, in each dimension). `modularRadiationRuntime1D`,
`modularRadiationRuntime2D`, `modularRadiationRuntime3D`, and
`modularMixedRuntime2D` run the actual HPX application and emit Silo files.
None of these HPX/Silo-dependent tests has been executed in this environment.
