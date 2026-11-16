# Cold white dwarfs with FLD

The WD path separates the cold degeneracy energy from the material reservoir
that emits and absorbs radiation. For each cell,

\[
e_{\rm th}=E_{\rm gas}-|\boldsymbol{p}|^2/(2\rho)-e_{\rm cold}(\rho),
\qquad e_{\rm th}=C_V T.
\]

The existing dual-energy switch uses `tau^gamma` when subtraction cannot recover
the thermal component reliably. The implicit step changes the evolved total
gas energy by the accepted radiation exchange and updates the thermal tracer;
it does not reconstruct total gas energy from the tracer. Cold energy remains
in the WD pressure and hydrodynamic energy equations.

`fld_material_model=auto` selects fully ionized material for `eos=wd` and the
legacy constant `fld_cv` for the ideal EOS. The fully ionized model evaluates
`sum(X_s (Z_s+1)/A_s)`, `sum(X_s Z_s/A_s)`, and `sum(X_s Z_s^2/A_s)` from the
cell's normalized species tracers. An optional per-tracer
`atomic_number_squared` supplies the ion-number-weighted charge-squared moment
for a homogeneous mixture; its default is `atomic_number^2` for a single ion
species. Its fixed-density heat capacity is
`rho * (k_B/m_H) * sum(X_s (Z_s+1)/A_s) / (gamma-1)`. The cold WD EOS still uses
its existing electron molecular weight of two; arbitrary compositions do not
alter that cold EOS. The fully ionized thermal model includes classical ions
and electrons and does not include electron degeneracy corrections to heat
capacity, ionization energy, nuclear burning, or Compton exchange.

## Units and opacity

`fld_unit_system=physical` uses the finalized `code_to_g`, `code_to_cm`, and
`code_to_s`, including normalization performed by SCF and restored from a
checkpoint. Temperature is Kelvin. The speed of light and radiation constant
are derived from these units; `fld_c` and `fld_ar` remain parameters for explicit
code-unit fixtures. Physical opacity outputs are inverse code length.

Three providers are available:

| `fld_opacity_model` | Material exchange | Radiation transport |
| --- | --- | --- |
| `legacy` | `fld_absorption * rho^p * T^q` | Absorption plus `rho * fld_scattering` |
| `independent` | Same Planck absorption power law | Independent **total** `fld_rosseland * rho^p_R * T^q_R` |
| `free_free_thomson` | Planck mean of classical electron-ion free-free absorption | Rosseland harmonic mean of the frequency-dependent free-free absorption **plus Thomson scattering** |

For independent means, the exponents are `fld_density_exponent`,
`fld_temperature_exponent`, `fld_rosseland_density_exponent`, and
`fld_rosseland_temperature_exponent`. The Rosseland input is already total
extinction; scattering is not added to it a second time. Thomson scattering
produces no thermal source. `fld_gaunt=0` selects pure Thomson transport in the
physical provider; otherwise the Gaunt factor is a positive constant.

The thermal source is `c * alpha_P * (a_r*T^4 - E0)`. Using the Planck
absorption mean for absorption of `E0` is the gray approximation
`alpha_E = alpha_P`; an arbitrary radiation spectrum need not have that mean.

The physical provider evaluates the harmonic mean of the total monochromatic
coefficient through a growing interpolation cache backed by quadrature. It does
not add separately averaged free-free and scattering coefficients. Its free-free
normalization is

\[
\alpha_\nu=3.692\times10^8\,g_{\rm ff}\,n_e
\left(\sum_i Z_i^2n_i\right)T^{-1/2}\nu^{-3}
\left(1-e^{-h\nu/k_BT}\right)\ {\rm cm}^{-1}.
\]

This is a controlled fully ionized, classical opacity model, not an opacity
table for arbitrary WD material. Bound-bound, bound-free, degeneracy and
Compton corrections are absent.

`fld_opacity_temperature_floor` is a positive Kelvin cutoff required by the
physical provider. It regularizes **only opacity evaluation** at
`max(T,T_cut)`; the source remains `a_r*T^4` at the actual material temperature.
It never supplies gas heat or initial radiation. `opacity_regularized_cells`
counts leaf interior cells affected by the cutoff. A cutoff sensitivity study
is required before interpreting a production result dominated by such cells.

## Building a binary initial state

`test_problems/fld/dwd_example.cfg` is a short, coarse setup example. It uses
helium species and a `1e4 K` opacity cutoff explicitly as example choices. It
does not select the production masses, fill factor, composition, spatial
resolution, or atmosphere treatment. Create and review `scf.init` in the run
directory before launching it. That separate file accepts:

```text
M1=<accretor mass in solar masses>
M2=<0.7 times the accretor mass, in solar masses>
a=<initial SCF separation parameter>
fill2=<donor SCF fill parameter>
fill1=0.99
equal_struct_eos=1
contact_fill=0
```

SCF determines the cold WD structure and normalizes its physical units. With
`equal_struct_eos=1`, the applied donor surface constant is
`C2=(1-fill2)*phi_center2+fill2*phi_L1`, so `fill2=1` places its surface at L1;
`fill1` is ignored. The initial `fill2` estimate also enters as its fifth power
in the donor radius estimate. Neither quantity is the final volume filling
fraction reported by SCF. `scf_max_iterations` is a positive setup-only iteration
budget, defaulting to 256. The example explicitly uses 16 iterations at
level three for a bounded initialization check. Its `stop_step=1` limits the
short evolution after SCF, not the SCF work itself; `stop_step=0` would return
before any evolution. Saving that state also requires `output=<checkpoint_stem>`;
`stop_step=0` by itself does not write the final model. Check the SCF masses,
filling fractions, density maxima and virial residual before using its result.
An illustrative low-resolution run is not proof of SCF convergence.

`scf_tolerance=0` retains the historical fixed-count behavior. A positive value
requires the volume-integrated density fixed-point defect, normalized by target
total mass and with the relaxation factor removed, the actual post-update
component mass errors, and successive changes in central density, separation,
angular frequency, surface constants and EOS scales to meet that tolerance.
The default minimum is 25 iterations, with five consecutive stable iterations.
Exhausting the budget throws before evolution. `scf_convergence.dat` reports
the updated masses, mass ratio, density maxima, volume fills and virial after
the updated gravity solve. Virial and volume fills are reported independently
because their discretization error does not vanish merely by iterating SCF.
Line-of-centers samples retain their actual cell positions: their centrifugal
potential uses the same cylindrical radius and rotation center as the density
update. Treating an off-axis cell as a point on the line produces a nonzero
density defect even at an otherwise stationary SCF state.

The stellar SCF thermal parameter is zero. Its entropy tracer retains the
existing small numerical floor. The density-floor atmosphere has a separate
gravitational-support thermal prescription, so it is not a globally cold
vacuum. `cold_energy`, `thermal_energy`, `radiation_energy`,
`atmosphere_thermal_energy`, and `atmosphere_radiation_energy` expose the distinct
reservoirs. The standard five-tracer layout is retained: accretor core,
accretor envelope, donor core, donor envelope, vacuum. Setting
`core_frac1=core_frac2=1` places each uniform star entirely in its core slot
and initializes both envelope slots to zero. The unused slots remain available
for later models. The atmosphere classification uses the vacuum tracer and is
a diagnostic convention, not an extra opacity switch.

Use `fld=on` with `radiation=off`: the latter controls the separate M1 solver.
`light_curve_opacity_model=fld` evaluates the same total Rosseland provider as
evolution for the optical-depth diagnostic. This consistency does not make a
gray Rosseland photosphere a frequency-dependent spectrum. Unresolved optical
depth surfaces remain reported by the existing light-curve diagnostics.

The example selects `fld_boundary=vacuum`, a zero-incident-radiation modified
Marshak (Sánchez–Pomraning) boundary. The default remains `reflecting`, with zero
normal diffusive flux, so existing configurations retain their behavior.
For the vacuum condition, the outward face flux is \(F_{\rm out}=\beta cE_b\),
where \(\beta=\log(\cosh R)/(R\tanh R)\) ranges from \(1/2\) to \(1\).
The positive half-cell Robin discretization uses
\(F_{\rm out}=cE_c/[1/\beta+\chi_R(\Delta x/2)/\lambda]\), with the interior
closure estimate refreshed each nonlinear iteration. See the
[boundary equations and implementation limits](comoving_fld.md#outer-radiation-boundary).

`escaped_diffusion_energy` is the accepted per-step radiation energy leaving
the computational box through diffusion, and `boundary_diffusion_luminosity`
is that energy divided by the timestep. The six `escape_{x,y,z}_{minus,plus}`
fields separate its box-face contributions; `boundary_vacuum` identifies the
selected condition. These diagnostics use code energy and power units and
exclude explicit advective transport. The photosphere luminosity remains a
separate diagnostic proxy, rather than the escaping box-boundary flux or a
frequency-dependent observable. Domain-size and outer-atmosphere sensitivity
still require assessment for a production run.

New checkpoints preserve `fld_boundary`. Older checkpoints without this
optional field restore `reflecting`, overriding the command-line boundary
selection to retain the saved model's prior behavior.

## Staff et al. (2018) q = 0.7 candidate

The separate [seed configuration](../test_problems/fld/staff2018_q07/seed.cfg)
and [SCF input](../test_problems/fld/staff2018_q07/scf.init) follow the
q = 0.7, total-mass 0.9 solar-mass AMR initial-model family in
[Staff et al. (2018), Sections 2 and 2.2.1](https://arxiv.org/pdf/1806.00337).
That model has a homogeneous equal-mass carbon/oxygen accretor and a helium
donor. Its donor fills the Roche lobe; the paper describes initial numerical
readjustment initiating overflow. The configuration constructs a synchronous
cold-WD counterpart of that family, with zero imposed angular-momentum or
entropy driving. FLD is an added physical model, and this coarse candidate
does not reproduce the paper's resolution or establish its merger outcome.

| Parameter | Value |
| --- | --- |
| Accretor target mass | `0.5294117647058824` solar masses = `0.9/1.7` |
| Donor target mass | `0.37058823529411766` solar masses = `0.9*0.7/1.7` |
| Donor surface | `fill2=1`, with `equal_struct_eos=1` and `contact_fill=0` |
| Rotation | `async1=async2=0`, synchronous |
| SCF seed scale | `a=1`, `xscale=1`; physical units follow the cold WD normalization |
| Mesh candidate | Fixed level three; repeat at level four before assessing resolution sensitivity |
| SCF stopping | Tolerance `1e-6`, budget `1024`, minimum `25`, consecutive stable iterations `5` |
| FLD solve | Nonlinear tolerance `1e-8`, outer-iteration budget `150`, multigrid-cycle budget `200` |
| Radiation | Initially zero; fully ionized free-free/Thomson model, vacuum diffusion boundary |
| Numerical atmosphere | Density `1e-12` in code units, helium composition; support heat follows the SCF prescription |
| Opacity regularization | `1e4 K`, applied only to opacity; not an initial temperature |

The current level-three candidate exhausted its 1,024-iteration SCF budget
without meeting the unchanged `1e-6` criterion. Its final density residual was
`2.9369332187e-5`, component-mass error `2.7505207324e-6`, and scalar change
`6.1145009684e-11`, with zero consecutive stable iterations. Initialization
was rejected before evolution; no equilibrium seed or physical stability
pilot has been accepted. The bounded normal-CFL solver evidence below uses a
separate, deliberately incomplete SCF initialization.

Only the accretor-core, donor-core, and vacuum slots are populated initially;
both envelope slots are zero, so neither star has a core/envelope subdivision.
For the accretor mixture, `sum(X/A)=7/96`, `sum(X*Z/A)=1/2`, and
`sum(X*Z^2/A)=7/2`. Its tracer therefore uses `A=96/7`, `Z=48/7`, and
`atomic_number_squared=48`. These preserve the fully ionized heat-capacity,
electron-density, and free-free charge moments of the 50/50 mixture exactly
within the model; replacing the last value with `Z^2` would not. The donor and
vacuum tracers each use `A=4`, `Z=2`, and `atomic_number_squared=4`.
The unused accretor-envelope slot has the same composition as its accretor-core
slot; the unused donor-envelope slot has the same composition as its donor-core
slot. The cold
EOS has electron molecular weight two for both stars.

Create an isolated run directory and copy `scf.init` into it; that file is read
from the process working directory, independently of `--config_file`. The
following commands assume a built executable and one allocated HPX worker:

```sh
octotiger_source=/absolute/path/to/octotiger
octotiger_exe=/absolute/path/to/build/octotiger
run_dir=/absolute/path/to/staff2018_q07_l3
mkdir -p "$run_dir/seed"
cp "$octotiger_source/test_problems/fld/staff2018_q07/scf.init" "$run_dir/seed/scf.init"
cd "$run_dir/seed"
"$octotiger_exe" --hpx:threads=1 \
  --config_file="$octotiger_source/test_problems/fld/staff2018_q07/seed.cfg" \
  > application.log 2>&1
```

The seed configuration sets `stop_step=0` and `output=staff2018_q07_seed`.
The explicit-output path runs after SCF and the bulk-velocity correction, then
writes `staff2018_q07_seed.silo` and its companion
`staff2018_q07_seed.silo.data/` directory. Retain both. Check the final
`SCF converged` message, the last `scf_convergence.dat` row, and checkpoint
readability before using the model. A configuration file alone is not evidence
that this convergence target has been achieved.
The named `SCF_STATE` line records the finalized mass, length and time units,
post-update component masses, separation, angular frequency, stellar volumes,
and volume-equivalent radii. Multiply the radii and separation by `code_to_cm`
for centimeters; the physical period is `2*pi*code_to_s/omega` seconds.

Both model configurations set `fld_max_iterations=150`: the normal-CFL probe
required 104 outer iterations at the unchanged `1e-8` tolerance. FLD solver
options are stored in checkpoints and restored over run-time options, so the
seed must save this budget; changing only the evolution configuration would
not override a checkpoint's lower limit.

Resume into a separate directory using `evolve.cfg` for a short undriven CFL
pilot. This file omits the seed-only `output` option, which would otherwise
write a checkpoint and exit without evolving. An empty `--output=` is rejected
by the HPX command-line parser. The saved seed has hydro enabled; `hard_dt=-1` allows the
CFL-selected timestep rather than the smoke test's `1e-8` cap.

```sh
mkdir -p "$run_dir/pilot"
ln -s "$run_dir/seed/staff2018_q07_seed.silo.data" \
  "$run_dir/pilot/staff2018_q07_seed.silo.data"
cd "$run_dir/pilot"
"$octotiger_exe" --hpx:threads=1 \
  --config_file="$octotiger_source/test_problems/fld/staff2018_q07/evolve.cfg" \
  --restart_filename="$run_dir/seed/staff2018_q07_seed.silo" \
  --stop_step=19 --hydro=on --hard_dt=-1 \
  --driving_rate=0 --entropy_driving_rate=0 \
  > application.log 2>&1
```

This is a 20-step pilot with the current inclusive `stop_step` convention.
The shard-directory link lets Silo resolve the checkpoint's relative shard
paths from the new working directory.
Use its measured timestep and wall time to budget longer undriven evolution.
Track component mass and central-density drift, separation and orbital
frequency, angular-momentum and total-energy conservation, thermal/radiation
growth, and escaping radiation. Column five of `step.dat` is the accumulated
orbital phase `rotational_time`; divide it by `2*pi` for elapsed orbits. At exact contact,
an undriven model may begin physical mass transfer during numerical
readjustment; a detached stable plateau is not a promised outcome. Compare
level-three and level-four seeds and startup histories before interpreting
transfer onset or selecting any additional forcing. To form the level-four
candidate, rerun the seed in another directory with
`--min_level=4 --max_level=4` and the same physical inputs.

## Bounded validation

The `staff_five_cfl_l2_retry150` application probe passed two normal-CFL steps
at level two (32,768 cells), with `cfl=0.4` and `hard_dt=-1`. The timesteps were
`0.01551547` and `0.02029541` in code units. Coupled residuals were
`9.17034e-9` and `9.79584e-9` after 104 and 99 outer iterations. Reported
radiation energies and material temperatures remained finite and nonnegative;
the relative per-implicit-step thermal-plus-radiation-plus-escape energy
defects were below `2.5e-16`. These ledgers test the implicit exchange and
boundary leakage, not complete hydrodynamic energy conservation. The setup
used only two SCF iterations and is evidence for solver robustness, not an
equilibrium binary or physical stability.

After the global inner-target correction described below, the separate
`fld_global_target_amr` pure-scattering regression passed three steps on
147,456 cells with residuals below `7.7e-10`. This checks transport on a mixed
AMR hierarchy; it does not extend the DWD pilot's physical coverage. The saved
application logs and `metrics.json` files retain the detailed measurements.

`check_dwd_fld.py` launches one process at a time, defaults to one HPX worker,
and applies a 600-second total wall-time budget across all phases of a case.
Use a distinct empty work directory for each invocation. These WD runner cases
retain the default reflecting diffusion boundary; the standalone DWD example
selects vacuum. Available cases are:

| Case | Check |
| --- | --- |
| `cold` | Existing cold `STAR+WD` initializer; frozen-gas cutoffs at `1e3`, `1e4`, `1e5 K`, followed by two tiny coupled hydro/gravity steps with density and cold/thermal/radiation drift recorded |
| `heated` | Uniform WD thermal reservoir, physical units/composition and physical free-free source against an independent scalar backward-Euler root |
| `independent` | Same physical material with unequal Planck and Rosseland means, repeated with length/time units scaled by `0.5` and `2` while preserving the physical state, domain and elapsed time |
| `amr` | Mixed AMR hierarchy, density-dependent independent total extinction, pure transport conservation and pulse spreading |
| `restart` | Independent means with physical material; continuous versus Silo restart, refinement and coarsening |
| `shock` | WD pressure discontinuity evolved to code time `0.1` with hydro on and measurable motion; radiation off/on and timesteps `0.005`, `0.0025` |
| `scf` | Optional synthetic `M1=1`, `M2=.7` solar-mass pair, `xscale=1`, level-three SCF for 16 iterations followed by one physical FLD step; records residuals without claiming equilibrium |

For example:

```sh
python3 test_problems/fld/check_dwd_fld.py --exe /absolute/path/to/octotiger \
  --case heated --work-dir /tmp/wd_fld_heated --threads 1
```

The SOD-based heated fixture supplies `cold_energy + C_V*T` through the existing
SOD pressure input, disables its ideal-gas analytic comparison, and selects
`dual_energy_sw1=0`. SOD's initial entropy tracer includes that supplied total
internal energy; the first FLD step replaces it with the recovered thermal
tracer. This fixture tests conservative WD thermal subtraction but does not
test a separately initialized, consistent thermal tracer before that step.
The cold `STAR` fixture covers the genuine cold tracer initialization.
Its short dynamic check records density and energy drift without imposing a
resolution-dependent equilibrium threshold on a coarse grid. The frozen
cutoff comparison tests that opacity regularization does not create a material
or radiation reservoir. It is not a cutoff sensitivity result for a heated,
partially ionized production atmosphere.

For the SOD fixtures, `radiation=off` and `gravity=off` select the explicit-unit
branch of `these_units`, which preserves the supplied `code_to_s`. The unit
comparison keeps density and velocity units fixed, scales mass by the cube of
the length factor, and scales the inverse-code-length opacities and code
timestep in opposite directions. This preserves `A=B=1` as well as the
physical temperature and exchange problem.

Each case saves the commands, application logs, and checked diagnostics in
`metrics.json`. Test presence is not evidence of a passing run. The shock's two
timesteps and coarse spatial profile do not establish temporal order or a
resolved shock structure, and the SCF smoke does not establish
equilibrium or a resolved donor stream. These checks do not select or validate
a full DWD production campaign.

The synthetic SCF fixture requires enough spatial resolution to retain both
stellar centers. At level two with `xscale=1`, its existing SCF partition
collapsed after the first update, and the resulting orbital frequency produced
an invalid enthalpy target; the same failure occurred with FLD disabled.
Level three retained both stellar components through a 30-iteration diagnostic
probe. This observation motivates the fixture resolution but does not establish
a converged initial binary or a general minimum resolution.

The legacy `log.txt` iteration diagnostics are written after mass normalization
and before each density update; `scf_convergence.dat` now supplies the separate
post-update summary. The runner labels legacy values `last_iteration_*` and separately reads
`final_*` diagnostics from `binary.dat` after evolution. In the bounded
16-iteration fixture, the last normalized masses were `1` and `.7` solar
masses, while the final component masses were `.9976108` and `.6994727`, giving
`q=.70114788` (a relative target difference of about `0.164%`). The final ratio
is reported rather than asserted equal to the target; this is not a converged
production initial state.


## Frequency quadrature and nonlinear iteration

With \(x=h\nu/(k_BT)\), the physical provider evaluates

\[
\chi_R^{-1}=\frac{15}{4\pi^4}\int_0^\infty
\frac{x^4e^{-x}}{(1-e^{-x})^2}
\frac{dx}{A_{\rm ff}(1-e^{-x})/x^3+\sigma_T n_e}.
\]

The Planck mean is analytic, \(\alpha_P=15A_{\rm ff}/\pi^4\).
Composite Gauss-Legendre integration uses intervals [0,.125], [.125,.5], [.5,2],
[2,8], [8,24], [24,80]. Estimates with 16 and 32 points per interval are compared
at a relative tolerance of 2e-11. Additional 64- and 128-point estimates are
used only if needed; failure to converge raises an error. Spectral factors are
precomputed. The initial comparison costs 288 point evaluations per newly
sampled value; the maximum is 1440.

The universal function `log(chi_R/chi_s)` is cached against
`log(A_ff/chi_s)`, using four-point cubic interpolation on a natural-log grid
with spacing `1/128`. This adapts CosmicTiger's Einstein--Boltzmann background
interpolation to a table that grows on demand. A query outside the available
range samples the missing range and builds new polynomial coefficients,
extending to an aligned one-e-fold boundary. Existing coefficients stay fixed;
an in-range query performs no quadrature. Immutable snapshots permit concurrent
reads and extensions without holding a lock during integration. The cache is
recreated as needed after restart; it is not part of the physical state.

The reusable `octotiger::InterpolatedCache` in
`octotiger/interpolated_cache.hpp` binds a sampling function at
construction and accepts a grid spacing and extension block size. It supports
signed finite coordinates and values. Callers supply any log transformations
and choose spacing appropriate to their function; the interpolator does not
estimate approximation error. `LogRosselandCache` is the thin physical wrapper.
Tests also cover an unrelated positive function in log-log coordinates and a
signed nonmonotone cubic.

Pure Thomson and pure free-free limits are analytic. Ratio calculations and
reconstruction avoid forming potentially overflowing amplitude ratios. The
measured maximum interpolation error over the numerical test's dense log-range
scan is `2.25e-12`, below the `1e-10` test limit. Independent quadrature,
extension order, interval boundaries and concurrent extensions are checked.
There is no arbitrary opacity coefficient cap.

Temperature-dependent opacities and the flux limiter are refreshed in the outer
nonlinear iteration. The inner Jacobian holds them fixed and includes the
4 ar T^3 emission derivative. Final convergence is checked with refreshed
coefficients; a failed nonlinear solve never commits a radiation/material update.

The inner solve checks both a volume-weighted residual and a local residual
scaled to each cell's radiation and thermal reservoirs. Its global target is
`max(0.01 * initial, 0.05 * fld_tolerance * sqrt(scaleSquared))`, where `initial`
is the initial linear L2 residual and `scaleSquared` is the volume-integrated
squared nonlinear energy scale. There is no additional `0.1 * initial` cap:
forcing another relative reduction after reaching the absolute target can
demand accuracy below roundoff. The independent local guard is unchanged and
continues to protect cold cells whose errors the global norm can hide. The
final nonlinear tolerance is also unchanged. After global convergence,
leaf-only relaxation removes tiny cold-cell errors without reinjecting coarse
corrections. Trial radiation values are projected onto nonnegative values, and
the material candidate is evaluated as a sum of nonnegative terms. Damping
interpolates the absolute states, avoiding cancellation when the new radiation
is many orders of magnitude smaller. Every trial still requires the refreshed
nonlinear residual check, and accepted gas updates use the conservative exchange.

New FLD WD checkpoints also preserve the exact code-unit cold EOS A and B.
Reconstructing them from older CGS conversion constants could otherwise change
the cold reservoir by several parts per million. Existing checkpoints without
new FLD model metadata retain their historical restart behavior.
