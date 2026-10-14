# Drifting self-gravitating polytrope: SSP versus RK3

This benchmark compares `SSP_MULTISTEP` with ordinary `RK3` on the same moving,
self-gravitating $n=1.5$ Lane–Emden polytrope. It measures RHS evaluations per
simulated time, AMR work, and departures from the initial stellar equilibrium.
Both methods use the same executable, initial mesh, initial fields and physical
options; only the time integrator changes.

## Problem

All quantities below are in code units. The domain is $[-1,1]^3$, with ideal
gas $\gamma=5/3$, $G=1$, central density $\rho_c=1$, and nominal stellar
radius $R=1/3$. The Lane–Emden scale is

\[
\alpha=0.09123047097730642,\qquad
P=K\rho^{5/3},\qquad K=0.0418359551921.
\]

The initializer integrates the Lane–Emden equation with radial step
`star_dr = R/1024`. It floors the exterior density at $10^{-6}$ and thermal
energy density at $10^{-8}$. The atmosphere is initially stationary. Species
1 tracks stellar material; species 2 tracks the atmosphere.

The star starts at $(-0.3,0,0)$, moves with prescribed initial bulk velocity

\[
\boldsymbol v=(0.2,0,0),
\]

and evolves to $t=3$, so the reference center travels to $(0.3,0,0)$.
The nominal sphere remains inside the domain throughout the run. An independent
Lane–Emden calculation gives ideal zero-atmosphere mass

\[
M_\star=0.0258969297284,\qquad
t_{\rm dyn}=\sqrt{\frac{R^3}{GM_\star}}=1.19589755715.
\]

Thus the run covers about $2.51\,t_{\rm dyn}$. The dynamical-time convention is
stated explicitly because other conventions differ by numerical factors.

The default CFL factor is 0.4. There is **no hard timestep cap**; each integrator
uses its own timestep selection. Contact steepening is disabled for the smooth
stellar profile. The gravity opening parameter is 0.35, and the ideal-gas
dual-energy switch is 0.001.

## Density-based AMR

The allowed levels are 1 through 3, with uniform refinement disabled. The
existing `MOVING_STAR` density criterion requests level 3 wherever total density
exceeds $10^{-3}$. Its threshold decreases by a factor of eight for each
coarser requested level. Native refinement buffers and face, edge and corner
nesting determine the surrounding mesh. Regridding is performed every five
accepted steps at the default CFL.

The initial mesh has 52 level-2 and 96 level-3 leaf blocks. Although level 1 is
allowed, the default star's position and extent, together with native nesting,
require its background to be level 2 initially. This is a mixed AMR mesh, not a
uniform level-3 grid. Level 0 is the single root; once refined, it is no longer
a leaf.

The runner audits every saved mesh for the density criterion, the presence of
finest-level stellar cells, some coarser background, and a maximum one-level
jump between touching blocks. The full benchmark also requires actual refinement
and coarsening, each followed by an accepted step. Short exploratory runs can
use `--allow-no-topology-change`, which relaxes only that topology-event check.

## Work and physical diagnostics

The SSP stage log must contain exactly one initial RK3 step and two new RHS
evaluations for every subsequent step, including steps after mesh changes.
Ordinary RK3 uses three RHS evaluations per accepted step.

The report provides two work measures, each divided by the same elapsed
simulated time:

- **Global RHS calls:** the sum of actual time-integration stage counts.
- **Interior-cell RHS evaluations:** the sum of each stage count multiplied by
  the number of interior leaf cells present during that step. Mesh changes are
  applied to subsequent steps, not retroactively to the preceding batch.

The second measure accounts for differences in the two methods' adaptive meshes.
It excludes ghost-cell work, communications and regrid/history-transfer work.
It is neither a gravity-operation count nor a complete wall-clock cost model.

There are two integration gravity passes per RHS stage, `DRHODT` and `RHO`.
Their reported count includes the initial RK3 time step and excludes separate
initialization and regrid gravity solves. Process timings include initialization,
diagnostics, AMR and checkpoint output; the methods can write different numbers
of checkpoints. One sequential pair does not establish a repeatable timing gain.

`results.json` records topology, stage counts, work, admissibility, and mass and
species conservation, including boundary outflow. Physical pressure follows the
solver's ideal-gas dual-energy choice. Energy drift is reported as a diagnostic.
Stored visualization fields and times are converted back to code units using
each Silo snapshot's conversion metadata.

`equilibrium-analysis.json` adds total peak density, stellar tracer centroid,
stellar RMS radius and its change from the initial value, tracer-weighted bulk
velocity and RMS peculiar velocity, density error, and virial/binding measures.
The main RMS radius integrates the piecewise-constant cell representation,
including each cell's internal spatial variance. A cell-center-only radius is
also supplied to expose this sampling choice.

The density error is

\[
\frac{1}{M_\star}\sum_{\rm leaf\ cells}\Delta V\,
\left|\rho_{\rm numerical}-\rho_{\rm LE}
  (\boldsymbol x_{\rm center}-\boldsymbol x_0-\boldsymbol v t)\right|.
\]

Here the translated continuum reference is evaluated at cell centers, with the
initializer's density floor and radius cutoff. These are **point samples**, not
exact cell averages. Initial and final errors are retained separately. This
reference measures combined initialization, atmosphere, spatial, AMR, gravity
and time-integration effects; it is not an exact discrete equilibrium or an
isolated temporal-error measurement. No convergence order is inferred from this
single-resolution comparison.

The virial diagnostic is

\[
\frac{2T_{\rm pec}+3\int P\,dV+W}{|W|},\qquad
W=\frac12\int\rho\Phi\,dV.
\]

It includes all fluid in the box and evaluates peculiar kinetic energy relative
to the stellar tracer's bulk velocity. It omits boundary pressure and flux
terms, so it is not an exact finite-domain virial identity. The report also
records total energy in that bulk frame divided by $|W|$.

For a full pair, the runner requires fewer global RHS calls and fewer
interior-cell RHS evaluations per simulated time, plus final density error no
more than 1.05 times ordinary RK3 by default. `--max-error-ratio` exposes this
declared comparison tolerance. Passing these checks establishes this benchmark's
work/error comparison, not general convergence or long-term equilibrium quality.

## Reproduce or analyze

Run from the source checkout containing the SSP history-transfer implementation.
Use an otherwise idle local CPU allocation. The harness runs the two applications
sequentially, each with one HPX worker and CPU kernels, and limits NumPy/BLAS
threads to one.

```bash
python3 test_problems/ssp_multistep/polytrope_efficiency.py \
  --executable /tmp/octotiger-ssp-history-build/octotiger \
  --output-dir /tmp/ssp-polytrope-efficiency
```

The configured build also provides the dedicated target:

```bash
cmake --build /tmp/octotiger-ssp-history-build \
  --target ssp_polytrope_efficiency --parallel 1
```

The output directory must not already contain either method's run directory.
The harness preserves exact application commands, executable hashes, logs,
execution status, checkpoints and analysis JSON. New runs also preserve before/after
hashes of the four dynamically loaded project libraries, resolved with `ldd`.
Older launcher records are accepted for analysis with an explicit indication
that per-run library manifests are unavailable. Reanalyze the same pair with
the same executable and options:

```bash
python3 test_problems/ssp_multistep/polytrope_efficiency.py \
  --executable /tmp/octotiger-ssp-history-build/octotiger \
  --output-dir /tmp/ssp-polytrope-efficiency \
  --analyze-only
```

To recompute only equilibrium diagnostics from an existing result:

```bash
python3 test_problems/ssp_multistep/polytrope_diagnostics.py \
  --results /tmp/ssp-polytrope-efficiency/results.json
```

## Results

Completed locally on 2026-10-14, on `codex/ssp-history-amr` at HEAD
`767a1585939ea2fb7218ac3612356244d79b5d12` plus the existing uncommitted SSP work.
Both methods reached code time 3 with identical initial coordinates and all 14
stored hydro/gravity fields, identical initial meshes, and matching physical
options. Their final leaf inventories also match.

| Quantity | SSP multistep | Ordinary RK3 |
| --- | ---: | ---: |
| Accepted steps | 155 | 115 |
| RHS evaluations | 311 | 345 |
| RHS per unit code time | 103.666667 | 115.000000 |
| Interior-cell RHS evaluations | 22,849,536 | 25,282,560 |
| Integration gravity passes | 622 | 690 |
| Refined / coarsened parents | 4 / 4 | 4 / 4 |
| Final density L1 / ideal stellar mass | 0.093157915 | 0.093142598 |
| Final RMS-radius change | +2.410762% | +2.411953% |
| Final peak-density change | -4.130217% | -4.126471% |
| Final RMS peculiar velocity | 0.015506989 | 0.015505723 |
| Final virial residual | -0.008912286 | -0.008900241 |
| Final centroid x | 0.301061211 | 0.301061916 |
| Process time, seconds (uncontrolled load) | 1174.284 | 768.908 |

The declared comparison passed: **9.855072% fewer RHS calls** and
**9.623329% fewer interior-cell RHS evaluations** per unit time.
The final density-error ratio SSP/RK3 is **1.000164439**
(0.016444% greater for SSP), below the predeclared 1.05 limit.
The absolute normalized density L1 error is about 9.3% in both methods; this
coarse benchmark demonstrates comparable error and reduced work, not a
high-accuracy stellar equilibrium or a convergence order.

SSP used exactly **one initial RK3 step plus 154 two-stage steps**. It coarsened
four parents at step 70 and refined four at step 95, with accepted multistep
steps after both changes. The mesh changed between 120 and 148 leaves on
levels 2 and 3; level 1 was allowed but native nesting did not leave any level-1
leaves. Every saved mesh passed density tagging and face/edge/corner 2:1 checks.

Both runs retained positive density, tau and physical dual-energy pressure.
Conservative thermal energy becomes negative in some cold cells; the standard
dual-energy pressure branch is active there. This is not a claim that every
cell retains positive thermal energy computed directly from total energy.
Mesh-plus-outflow mass drift was at most 2.28e-15 relative, and the corresponding
energy drift at most 1.35e-14. The common initial density-reference error was
1.052524e-7. The final reference center was x=0.3.

The timing numbers **do not establish a speedup**. Another local CPU campaign
overlapped part of SSP, while the later RK3 run experienced different load and
brief movie-rendering activity. The methods also wrote 32 versus 24 distinct
states. The reported advantage is the measured RHS work reduction, not elapsed
time. Repeated timing under controlled load remains a separate measurement.

The executable and four project libraries match the pre-existing frozen-build
manifest. The original SSP launcher recorded only the executable hash; its
per-run before/after library manifest is unavailable. RK3's before/after library
manifests match. These scopes are explicit in the saved JSON; no missing
historical record was invented.

The first analysis attempts encountered a seven-significant-digit log rounding
issue. The reader now uses bounds derived from each printed decimal place,
including cumulative time checks. The simulations were not rerun. All 20 RHS
accounting tests pass, including eight new timestamp cases, and four previously
saved pulse logs plus the full star logs pass the corrected analysis. CMake
registers the explicit `ssp_polytrope_efficiency` target.

The VisIt movie contains all 32 distinct SSP states as a central density slice
with fixed logarithmic colors, native cells and leaf-block outlines. Visible
time is code time; the renderer converts Silo's CGS seconds with `code_to_s`.
The 15-second 1440x1080 H.264 movie passed per-state time/domain-coverage checks
and a complete FFmpeg decode. The final encoded poster was visually inspected.
VisIt returned 250 at shutdown after the full render; its completion sentinel,
manifest, session, image checks and absence of render errors confirmed success.

Artifacts: `output/ssp-polytrope-amr-20261014/results.json`,
`equilibrium-analysis.json`, `analysis.log`, and
`movie/ssp-polytrope-amr.mp4`. Exact commands, logs, checkpoints, rendering
scripts, source frames, playback mapping and hashes remain alongside them.
All work was local. **QB login attempts: 0 / 0 authorized.**
