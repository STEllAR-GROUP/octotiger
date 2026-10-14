# Variable-step SSP multistep integration

`--time_integrator=SSP_MULTISTEP` selects the two-step, two-stage, third-order
method from the supplied manuscript, *General Linear Mulistep Methods for Codes
Requiring 3rd Order Runge-Kutta with SSP*. The existing three-stage method remains
the default, `--time_integrator=RK3`. Configuration files accept the same names,
case-insensitively, through `time_integrator`.

The selection covers hydrodynamics and its coupled self-gravity calculations.
Gravity-only runs solve an elliptic problem and do not enter either time
integrator.

With `radiation=on`, multistep integration is disabled and hydrodynamics uses
`RK3`, even when `SSP_MULTISTEP` is requested. Radiation continues with its
configured integration method. This implementation does not adapt the
multistep scheme to radiation. Exact SSP checkpoint payloads that declare
radiation are rejected rather than discarding history or changing the saved
method. A valid hydro-only exact SSP checkpoint restores its saved physics and
integrator, including `radiation=off`, even if the command line requests
radiation. Ordinary radiation restarts resolve a requested multistep selection
to `RK3` after restoring the saved radiation setting.

## Update and coefficients

Let the current step be \(h=t_{n+1}-t_n\), the previous step be
\(h_{n-1}=t_n-t_{n-1}\), and \(\alpha=h_{n-1}/h\). With the previous state and
its derivative cached, the update is

\[
\begin{aligned}
k_1 &= F(y_{n-1}), & k_2 &= F(y_n),\\
y_3 &= c_1(y_n+vhk_2)+c_2y_{n-1}, & k_3 &= F(y_3),\\
y_{n+1} &= c_3y_n+c_4(y_3+vhk_3)
           +c_5(y_{n-1}+v_5hk_1).
\end{aligned}
\]

Only \(k_2\) and \(k_3\) need new evaluations. The cached derivative is always
the derivative at the previous step's initial state, not its final stage.

The implementation evaluates an exact member of the manuscript's coefficient
family, using its optimal nonnegative boundary \(c_3=0\):

\[
\begin{aligned}
v &= \frac{\alpha}{\sqrt{\alpha^2+2}+\alpha-2},\\
c_2 &= \left(\frac{v}{\alpha+v}\right)^2,
& c_1 &= 1-c_2,\\
c_3 &= 0,
& c_4 &= \frac{(\alpha+1)^2(\alpha+v)^2}
                    {2\alpha^2v(\alpha+2v)^2},\\
c_5 &= 1-c_4,\\
v_5 &= -\frac{(\alpha+1)
 [2\alpha^2v-2\alpha^2+5\alpha v^2-6\alpha v-3v^2]}
 {2\alpha^2(\alpha+2v)^2c_5}.
\end{aligned}
\]

Zero is a valid weight in a convex combination. Within this family, nonnegative
\(c_3\) requires \(v\) to be at least the value above; both \(v\) and \(v_5\)
increase with \(v\) on the admissible branch. The boundary therefore minimizes
\(\max(v,v_5)\). Neither coefficient can be ignored: \(v_5\) can exceed \(v\)
when the timestep decreases.

This replaces the manuscript's empirical fit for \(c_4\), which can produce a
negative weight, and avoids cancellation in its cubic-root expression. The
algebraic helper `from_c4` retains the free-parameter family for checking the
manuscript's verification table. Several table rows have negative \(c_3\);
they verify the algebra but are not production SSP coefficients.

The second stage has time
\(t_n+(c_1v-c_2\alpha)h\). The corresponding rotational time is used when
evaluating time-dependent driving sources. Bootstrap steps in this option use
the RK3 stage times \(0,1,1/2\) relative to the step. The default RK3 execution
path is preserved.

## Timestep selection and stability assumptions

Timestep growth is capped at 1.9 times the previous accepted step:

\[
0<h\leq1.9h_{n-1},\qquad \alpha\geq1/1.9.
\]

This leaves a margin from the singular SSP boundary at \(\alpha=1/2\).
There is no artificial limit on timestep decreases. The implementation evaluates
coefficients using the bounded variable \(z=v/(\alpha+v)\), avoiding powers of
large \(\alpha\) and cancellation in \(c_5=1-c_4\). The historical Euler duration
\(hv_5\) is formed in extended precision once per stage before updating fields.
A very small endpoint timestep therefore does not require RK3 merely because
its ratio to the previous step exceeds four.

Each leaf estimates its forward-Euler (FE) limit as `cfl * dx / a`, where `a`
is the maximum wave speed returned by its hydro flux calculation. This limit
is reduced over the tree independently of the requested step, which also
respects `hard_dt` and the existing stop-time/batch cap. The previous global FE
limit is retained separately from the previous accepted step size.

The selector finds the largest supported step no greater than the requested
step and the 1.9 growth cap, subject to both bounds:

\[
hv\leq\Delta t_{\mathrm{FE},n},\qquad
hv_5\leq\Delta t_{\mathrm{FE},n-1}.
\]

These are separate bounds because the cached derivative and current derivative
can have different wave speeds. When no positive representable step satisfies
them, or no history exists, the whole tree uses RK3 for that step. In particular,
\(hv_5\to2h_{n-1}/5\) as \(h\to0\), so reducing the new step cannot
repair an arbitrarily small historical FE bound. The root broadcasts this
choice with the accepted timestep so leaves and internal nodes execute the
same number of hydro exchanges and gravity passes.

The SSP interpretation also requires the FE limit used for the current state
to be valid for the intermediate state \(y_3\). The implementation uses the
existing CFL assumption for intermediate stages; it does not reject and retry
a step if a later stage's wave speed would require a tighter bound. Consequently
the coefficient and timestep checks do not establish an unconditional SSP
guarantee for arbitrary state-dependent wave-speed changes. Existing density,
species, and entropy floors and the final dual-energy update still apply.

## Self-gravity and conservation

Gravity remains an elliptic solve coupled to each hydro derivative evaluation.
Each stage follows the existing sequence:

1. Compute fluxes, exchange AMR flux corrections, and form the hydro/gravity
   source terms and density derivative.
2. Solve the `DRHODT` gravity problem to obtain the potential derivative.
3. Complete the energy derivative, then apply the selected stage combination.
4. Solve the `RHO` gravity problem for the updated density, preserving total
   gas-plus-gravitational energy during the potential replacement.

The historical derivative is saved after the potential-derivative contribution
has been added to gas energy. Its momentum and angular-momentum components
already include gravitational forces. The historical state includes the
`pot_i` field, \(\rho\phi\), and the same weights combine this field and gas
energy. The subsequent gravity solve preserves
\(E_{\mathrm{gas}}+\rho\phi/2\) while replacing the combined potential field
with the potential of the new density. This retains the existing conservative
energy accounting in the multistep update.

Cumulative boundary outflows are also evolved with the multistep formula,
including their rotating-frame source terms. Their previous values and complete
initial derivatives are saved alongside the cell histories. Thus conservation
diagnostics use boundary contributions integrated with the same method as the
interior state.

## History lifetime and cost

A fresh run takes one RK3 startup step. Completed multistep history survives
scheduled batch boundaries and mesh changes. Refinement conservatively
interpolates the previous state and complete derivative using historical
interior values, with second-order one-sided reconstruction at block edges.
The historical state and derivative share each selected slope stencil. One
factor per coarse cell limits all eight children and all fields together,
preserving parent means while protecting admissible density, species, entropy,
and ideal-gas thermal states and historical Euler endpoints. It cannot repair
an already inadmissible coarse endpoint, and it is not a gravity positivity
proof.

Coarsening volume-averages both field histories before deleting the children
and sums integrated outflow histories. Refinement assigns the outflow ledgers
once, and new children inherit the previous timestep and FE limit. Historical
potential and gas energy remain paired; angular-momentum redistribution is
applied to both the historical state and derivative. Historical restriction
traffic is added only at refinement checks, not at ordinary stage exchanges.

A [fixed-mesh temporal experiment](ssp-history-remap-accuracy.md) found a
first-order time defect from direct transfer at fixed old/new mesh resolutions.
The [joint space/time experiment](ssp-history-joint-convergence.md) retained
the same convergence order and comparable errors as RK3 restart. The more
expensive [virtual-history correction](ssp-history-reconstruction.md) is a
separate research prototype and is not used by production code. Application
AMR validation uses the dedicated moving `AMR_ADVECTION` problem and
`test_problems/ssp_multistep/amr_accuracy.py`, with a separate `MOVING_STAR`
hydro/gravity harness. The [application validation report](ssp-history-amr-validation.md)
records approximately second-order smooth-flow L1 convergence with errors
within 0.35% of the matched rebootstrap reference, completed gravity/restart
coverage, and the remaining accuracy limitations.

Migration serializes the current state, histories, timestep/FE metadata, and
gravity caches. An exact checkpoint resumes at its saved absolute step and
batch phase, without inserting an RK3 step or changing the regrid schedule.
`--stop_step` still counts steps requested for the current invocation.

SSP Silo output includes a versioned lossless restart payload alongside the
usual visualization fields. The latter are insufficient for exact restart:
they use unit conversions and reconstructed density, and omit integration
history. `restart_version=2` and `restart_groups`, aligned with `node_list`,
identify the payload in each data group's `restart/<octal node id>/state`.
Every node, including internal gravity nodes, is saved. Each payload contains
the raw current hydro fields, gravity fields, outflows, previous state and
complete derivative, historical outflows and derivatives, gravity centers,
cached masses and multipoles,
clocks, absolute step, previous timestep, and previous FE limit. Global
metadata preserves physical constants, units, and rotation without arithmetic
conversion. Scratch stage buffers are recomputed before use.

An exact restart reconnects the saved tree without extra gravity solves,
density reconstruction, normalization, or startup regridding. It restores the
saved integrator when no explicit selection is supplied. Incompatible solver
settings, layouts, or malformed claimed-exact payloads produce an error; they
do not silently discard history. Format-1 exact checkpoints inside a batch
remain readable. Format-1 checkpoints at a completed batch/regrid boundary are
rejected explicitly because newly refined leaves may lack the history needed
by the new continuation; use an inside-batch checkpoint or the original
executable. Older Silo files without any exact payload retain the legacy loader
and RK3 bootstrap and cannot promise exact continuation.

Bitwise continuation is tested with the same executable, CPU backend,
single-locality configuration, solver settings, and stopping-time policy.
`--stop_time` may be changed to extend a run; this intentionally changes the
timestep cap near the end and therefore the trajectory being compared. Changing hardware,
backends, decomposition, or floating-point reduction order is not covered by
that result. Extra checkpoint data can exceed twice the old visualization
file size: it includes raw current state and both historical field arrays,
plus gravity caches on internal nodes. File size depends on mesh and fields.

A mature multistep step uses two complete RHS evaluations, including the
first step after regridding. Over \(N\) accepted steps with valid histories,
the count is \(2N+1\), versus \(3N\) for RK3. History transfer adds interpolation,
restriction, and communication work, but no new hydro RHS or gravity probes.
With gravity enabled, a steady multistep step uses four gravity passes (two
`DRHODT`, two `RHO`), compared with six for RK3. Necessary startup/fallbacks
remain when usable history or an admissible timestep is unavailable. These
operation counts do not establish a measured wall-clock speedup.

The selected option allocates three additional interior field arrays per leaf:
the previous state, previous complete RHS, and current initial RHS awaiting
promotion to history. With `safe_real=double`, their main storage is
`3 * n_fields * INX^3 * sizeof(double)` bytes per leaf, plus three outflow
vectors and allocation overhead. The existing `U0` stores the current initial
state. These extra arrays remain empty when using the default RK3 integrator.

## Tests

The coefficient and nonlinear-order test is independent of HPX and Silo. From
the repository root it can be built and run with a C++17 compiler:

```sh
mkdir -p build-ssp-check
c++ -std=c++17 -O2 -I. test_problems/ssp_multistep/coefficients.cpp -o build-ssp-check/ssp_multistep_coefficients
./build-ssp-check/ssp_multistep_coefficients
```

It checks the manuscript's coefficient table, nonnegative production
coefficients, nonlinear third-order conditions and convergence at constant and
variable step sizes, evaluation counts, the 1.9 growth cap, decreases beyond a factor of four,
extreme timestep scales, and separate current and historical FE bounds.

For a normally configured Octo-Tiger build with its required dependencies,
enable the test suite and run the dedicated tests sequentially:

```sh
cmake -S . -B build -DOCTOTIGER_WITH_TESTS=ON
cmake --build build --target octotiger ssp_multistep_coefficients ssp_history_amr -j1
ctest --test-dir build -R ssp_multistep --output-on-failure -j1
```

Retain the dependency and backend settings required by the existing build when
configuring it. Python 3 enables the CPU runtime tests; the coupled-gravity test
also requires interaction lists (`OCTOTIGER_DISABLE_ILIST=OFF`). The runtime
harness compares omitted/default selection against explicit RK3 and exercises
one initial SSP startup, two-stage updates through multiple regrid/batch boundaries on small
Sod problems with and without self-gravity. It checks finite positive states,
mass/species conservation including outflows, hydro energy conservation, and a
self-gravity energy-drift bound relative to RK3. Failed runs retain their logs
and diagnostics for inspection.

The separate hydro and gravity restart tests split runs inside a batch (3+7
steps), at a boundary (5+5), and after a timestep change (11+2). They compare
every saved node archive byte-for-byte, check raw global metadata, and require
the restarted stage/ratio sequence to equal the uninterrupted suffix. A
scaled-unit rotating gravity case also exercises raw clocks and conversions.
Additional restart cases intentionally change the stopping time to force a
100-fold timestep decrease and 1.9-fold growth inside an active batch; both
must retain two-stage updates. These cases test timestep transitions separately
from the identical-stopping-policy exact-continuation comparisons.
These fixed-mesh runtime tests use Python's standard library and the configured
Silo shared library. Set `SSP_TEST_HPX_THREADS=1` to run them with one HPX worker
when needed for a shared local resource budget. The independent `history_amr`
C++ test checks conservative paired reconstruction, block-edge accuracy,
admissibility helpers, and extreme thermal-state scales.

When NumPy is available, CMake also registers moving-mesh `amr_hydro` and
`amr_gravity` tests; the latter requires interaction lists. Their harnesses
use one HPX worker and require actual refinement/coarsening, continued
multistep updates, conservation checks, and exact restart through later mesh
changes. The hydro regression uses `AMR_ADVECTION`, levels 1:3, final time 0.35,
and contact steepening on, with accepted steps required after both directions
of topology change. These are topology and continuation regressions, not
resolution studies.

All eight SSP CTests and separate real format-1 compatibility checks passed
in the recorded validation. Controlled smooth-flow AMR retained approximately
second-order density L1 convergence, with errors within 0.35% of the matched
RK3-rebootstrap reference. Moving hydro and coupled gravity passed actual
refinement/coarsening, continued two-stage evolution, and exact restart.
See the [application validation report](ssp-history-amr-validation.md) for the
measured errors and limitations, including slower maximum-cell convergence.
These CPU results do not establish multilocality or GPU behavior.

## AMR RHS efficiency benchmark

The dedicated [smooth-pulse benchmark](ssp-rhs-efficiency.md) compares this
method with ordinary RK3 at the same CFL parameter and final time. Density
above 1.005 tags the moving pulse to level 3; native nesting determines the
surrounding coarser mesh, with level 1 permitted. It requires refinement and
coarsening, checks exact-solution errors, and measures both RHS calls and
interior-cell-weighted RHS work per simulated interval. Run it explicitly with
the `ssp_rhs_efficiency` build target. Use `--mesh uniform` on its Python runner
for a fixed-grid control.

The full paired benchmark is separate from the correctness suite. A ninth
CTest, `rhs_accounting`, checks work accounting across mesh changes and final
partial batches; the eight-test application validation above predates that
addition.

The [moving self-gravitating polytrope benchmark](ssp-polytrope-efficiency.md)
compares the same methods with density-based AMR and coupled gravity. Run it
explicitly with the `ssp_polytrope_efficiency` build target.
