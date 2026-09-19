# Step 04: parameterized grey opacity

## Baseline and scope

Starting commit: `97d84fc3fb84da3c34f395dd4b898d212a6be7ee`, clean worktree.
Step 03 archive SHA-256:
`a06e6a23b756344c6c46652e3065f86edbf223880cde4f67a3d56ff8ee8c7c1a`.
The serial validator had a stale repository-root calculation after Step 03.
Correcting `parents[1]` to `parents[2]` allowed the production-method baseline
to run before changing any physics. Its complete stdout was unchanged by the
new legacy selectors. Streaming-wave L1 errors remain 9.529321e-03,
4.447625e-03, 1.234238e-03 (last order 1.849414).

This adds selectable material opacities, not the Step 05 shock suite. Hydro,
gravity, M1 equations, closure, reconstruction, boundaries, source integration,
and subcycling are unchanged. The existing radiation contribution to the gas
acoustic bound now selects the same total opacity as flux damping; its formula
and the hydro timestep code are unchanged. This small coefficient-interface
change is tested separately. No tolerances were relaxed.

## Existing path and compatibility

`options::process_options` resolves defaults/config/CLI, optionally loads Silo
options, restores explicit radiation overrides, validates, then broadcasts
`options`. The active material path is `rad_grid.cpp`:

| Consumer | Before Step 04 | Selected role now |
|---|---|---|
| `rad_imp`, emission and matter/radiation thermal exchange | rho*rad_opacity if nonnegative, otherwise kappa_p | radiationAbsorption |
| `prepareSources`, absorption-only attenuation | same absorption coefficient | radiationAbsorption |
| `prepareSources`, implicit flux damping and momentum feedback | rho*rad_opacity if nonnegative, otherwise kappa_R | radiationTransport |
| Explicit velocity work | 2*chiAbsorption - chiTotal | unchanged |
| Radiation acoustic contribution | kappa_R or the constant override | radiationTransport |
| Hyperbolic M1 face flux | closure/HLL, no explicit opacity | unchanged |

Despite their names, existing `kappa_p` and `kappa_R` return inverse code length,
not area/mass. Their special problem branches and density/temperature/composition
formulae are untouched. Rosseland includes the existing free-free/bound-free
and Thomson-like terms; Planck has its separate absorption fit. Different means
cannot be subtracted to infer a scattering mean. The old `cpu_kernel.hpp` uses
these original functions but has no caller in the active M1 source path.

Default `model=legacy` retains all old choices and operation order, including
lazy evaluation of only the mean needed by each consumer. The explicit
`skinner_ostriker` mode requires nonnegative `radiation.opacity.constant` and
uses that exact legacy equal-mean calculation. The old `rad_opacity` alias and
its warning/conflict rules remain unchanged. No other opacity spelling is removed.

The prescribed Gaussian/sphere tests retain their independent `rad_test_chi`
(`radiation.test.extinction`), including their fixed-medium analytic source.
The streaming tests remain source-free. These four regression problem types
reject a non-legacy material model to prevent silently ignoring it. Their
existing configurations and commands still work. Coupled problem inputs can
select the new model independently of the verification adapter.

## New model, means, and units

| Canonical option | Default | Meaning |
|---|---|---|
| radiation.opacity.model | legacy | legacy, skinner_ostriker, grey |
| radiation.opacity.constant | -1 | Existing equal-mean override in **code area/mass**, with legacy alias rad_opacity |
| radiation.opacity.units | cm2/g | New grey inputs: cm2/g or 1/cm |
| radiation.opacity.absorption | 0 | Planck emission AND radiation-energy absorption mean |
| radiation.opacity.scattering | 0 | Coherent isotropic transport scattering |
| radiation.opacity.transport_absorption | -1 | Rosseland/flux absorption mean; -1 follows absorption |

For `grey`, thermal extinction chi_A uses absorption; transport extinction chi_T
uses transport_absorption + scattering. Transport absorption is not required to
exceed the Planck mean: these are differently weighted coefficients. The total
transport coefficient is derived, avoiding an inconsistent independent total.
The defaults of the new fields are inert until grey is selected. Non-default
separate coefficients/units with legacy modes fail clearly. Grey combined with
a nonnegative legacy constant fails and explains how to remove the conflict.
All coefficients must be finite and nonnegative except the exact -1 sentinel.

Let L=code_to_cm, M=code_to_g, and rho be code density:

* cm2/g inputs: chi_code = rho * (M/L^2) * kappa_cgs.
* 1/cm inputs: chi_code = L * chi_cgs, independent of density.
* Thus physical optical depth is always chi_code * dx_code.

The second form supports prescribed constant volume extinction, without
forcing users to divide by a changing density. Setting all three unit conversion
factors to 1 keeps cgs throughout. Legacy constant inputs deliberately keep
their historical units even when the new `units` default says cm2/g.

These are supplied, constant grey means. They are not frequency integrals,
opacity tables, a frequency-dependent model, or a temperature/density power-law
model. Planck and radiation-energy absorption are identified; Rosseland is used
as the flux/transport mean outside diffusion as a grey approximation. Scattering
is elastic and isotropic: no Compton heating or anisotropic phase function.
No frequency groups or new Ensman initial/boundary conditions are introduced.

## Source equations and conservation

Write Q=F/c, r=c_hat/c, and B=a_R*T^4. The inherited isotropic source model is

    dE/dt = c_hat chi_A (B-E) + r (2 chi_A-chi_T) v dot Q
    dQ/dt = -c_hat chi_T Q + r chi_T (4/3) E v.

The thermal part is implicit; velocity terms and damping retain the existing
split and theta fallback. Gas receives the opposite applied thermal/work energy
increment divided by r, and the opposite Q source increment divided by c_hat.
The conserved local combinations are gas total energy + E/r and gas momentum
+ Q/c_hat. For physical c these are the physical totals. Scattering alone has
zero thermal exchange, damps flux, transfers momentum, and permits mechanical
work in moving matter. Gas *internal* energy need not be invariant under a
finite momentum update; it is recomputed from conservative total energy.

The retained equal-mean formulation and approximation of the velocity terms
follow Skinner & Ostriker (2013), equations (5), (8), and (18):
https://arxiv.org/abs/1306.0010 . That paper neglects scattering. Here coherent
scattering is an explicit grey extension: zero comoving thermal source and a
transport force, with lab-frame work supplied by the retained chi_T term.

As in Step 03, `energy_mode=absorption` intentionally removes absorbed band
energy without gas thermal deposition; it is not a conservative thermal model.
`equilibrium` disables thermal exchange. Use `thermal` for coupled grey energy
conservation. The gas EOS, including the existing Marshak special emission law,
is not changed by choosing opacity.

## Limits and validation

* Zero extinction: no material thermal/force/work source.
* Pure absorption: thermal equilibration, nonnegative energies, total conservation.
* Pure scattering: zero thermal exchange; implicit thin/thick flux relaxation,
  momentum conservation and moving-medium mechanical work.
* Equal means: legacy, explicit S&O, and grey source/driver states are bitwise equal.
* Independent means: production absorption/transport caches and acoustic opacity
  use their specified roles; non-unit cgs conversions are checked.
* Diffusion: the isotropic pressure/flux source balance gives
  F = -c/(3 chi_T) grad(E), independently of the reduced light-speed ratio.
  The test checks that operator balance and inverse-opacity scaling, **not** an
  asymptotic-preserving discretization on unresolved mean free paths.

The existing transport discretization can have excessive numerical diffusion
in optically thick cells. Explicit velocity terms retain their beta*tau stiffness
restriction, and thermal splitting can reject a beam if cooling leaves |F|>cE.
There is no claim of accurate arbitrary dynamic diffusion, or of fully validated
Ensman shocks, from these source-cell tests. Resolve the relevant scales and
use Step 05's benchmark comparisons before trusting that regime.

## Restart and result metadata

New Silo fields use additive `rad_opacity_schema=1` and an explicit model/units
encoding. The tested production codec reads pre-Step-04 checkpoints as legacy;
rejects unknown schemas, invalid enums and incomplete extensions; and round-trips
all new fields. Explicit CLI/config fields override checkpoint values individually.
The existing rad_opacity Silo field is unchanged. Older executables do not know
the grey extension and must not restart new grey checkpoints. The HPX broadcast
appends the new structure after the old fields; mixed-version localities and
persisted raw HPX options archives are not supported disk checkpoint formats.

Startup reports model, units and coefficients; Silo carries the same settings.
New `run.json` records an `opacity` object separate from prescribed test chi.
Plot `plot-inputs.json` already embeds the full run metadata. Movie metadata now
also embeds the run, and the report exposes opacity settings and links to the
run/batch records. Existing old results remain marked as lacking recorded opacity
metadata; resuming them does not invent new provenance or change their config.

## Reproduce from HOME

```bash
cd ~
repo="$HOME/octotiger/src/octotiger"
g++ -std=c++20 -O2 -I"$repo" "$repo/tests/greyOpacity.cpp" -o /tmp/grey-opacity-tests
/tmp/grey-opacity-tests
python3 "$repo/verification_results/radiation/validate_so.py" --opacity-checks
python3 "$repo/verification_results/radiation/validate_so.py" --opacity-checks --release
python3 -m unittest discover -s "$repo/verification_results/tests"
python3 "$repo/tests/options_compatibility_test.py"
```

Example coupled material configuration fragment (not a complete shock test):

```ini
radiation.opacity.model = grey
radiation.opacity.units = 1/cm
radiation.opacity.absorption = 1e-10
radiation.opacity.scattering = 0
radiation.opacity.transport_absorption = -1
radiation.energy_mode = thermal
```

The illustrative coefficient is not asserted to be a published Ensman parameter.
For reproducible legacy equal means use `radiation.opacity.model=skinner_ostriker`
and the original nonnegative `radiation.opacity.constant` value.

The standalone M1 suite passes 175,509 checks; the verification harness passes
11 tests and the Step 01 compatibility suite passes 7 tests. Grey unit/codec and
production-source tests pass with assertions enabled and with NDEBUG, using GCC
13.3.0. AddressSanitizer and UndefinedBehaviorSanitizer pass (leak detection
disabled for this container). FpeGuard also passes. The
compiled Boost parser and full C++ harness tests include the new options/metadata
cases but require the absent Boost/HPX/Silo/CMake dependencies. Actual Silo I/O,
MPI/HPX delivery, full production binaries and the full Ensman suite have not
been run in this container.
