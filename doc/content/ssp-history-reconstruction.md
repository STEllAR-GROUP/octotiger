# Reconstructing virtual SSP history after regridding

The standalone prototype in
[`history_reconstruction.py`](../../test_problems/ssp_multistep/history_reconstruction.py)
recovers third-order temporal convergence in smooth scalar tests after a mesh
change. It reconstructs a history compatible with the **new** spatial operator
and limits its admissible Euler pair conservatively. This virtual-history
correction remains a standalone experiment, separate from the direct history
transfer implemented in Octo-Tiger. See the separate
[application validation report](ssp-history-amr-validation.md) for application
evidence and its limits, and the [implementation description](ssp-multistep.md)
for the current algorithm.

This is a proof of concept, not an efficiency improvement. Its transition step
normally requires **five RHS evaluations**, versus three for the RK3 restart
comparison. Although reconstruction runs accept only their initial step as RK3,
they perform a discarded forward RK3 probe at each regrid. Thus this approach
does not satisfy the computational intent of “RK3 only once.”

The [direct-transfer experiment](ssp-history-remap-accuracy.md) explains the
old/new operator mismatch. Its fixed-mesh temporal-accuracy criterion is
stronger than demonstrating the expected combined space/time convergence for
a PDE with CFL-scaled timesteps. The present correction addresses that stricter
criterion; it does not establish that direct transfer necessarily loses the
PDE's nominal joint convergence order.

## Reproduce

Python, NumPy, and SciPy are required. From the repository root:

```sh
python3 test_problems/ssp_multistep/history_reconstruction.py --json /tmp/ssp-history-reconstruction.json
```

The default suite ran in 5.11 seconds on one NumPy/BLAS thread. It includes
linear advection and nonlinear Burgers, independent refinement/coarsening,
an unchanged-mesh control, repeated mesh changes every eight accepted steps,
and halving the timestep when refining. A separate sharp near-vacuum profile
exercises the history limiter.

For the asymptotic repeated-regrid check:

```sh
python3 test_problems/ssp_multistep/history_reconstruction.py --equations burgers --events frequent --steps 256 512 1024 2048 --skip-front --json /tmp/ssp-history-reconstruction-frequent.json
```

That run took 2.36 seconds. The script reports errors, orders, minimum limiter
weights, total RHS calls, and numbers of accepted RK3 steps. Optional JSON
contains all diagnostics, including probe calls and admissibility checks.
It is not registered as a production regression test.

## Construction

Let \(U\) be the remapped current state, \(H\) the preceding timestep, and
\(h\) the next timestep. The multistep formula needs a historical state \(p\)
and its historical Euler point

\[
E=p+hv_5K_p.
\]

For smooth third-order time accuracy with bounded timestep ratios, sufficient
history approximations are

\[
p=\Phi_{\mathrm{new}}(-H,U)+O(H^4),\qquad
K_p=F_{\mathrm{new}}(\Phi_{\mathrm{new}}(-H,U))+O(H^3).
\]

Direct backward integration can evaluate negative trial densities. Instead,
the prototype takes a forward SSPRK3 **probe** over \(\delta\leq H/2\), obtaining
\(Q\), and evaluates \(F(Q)\). Cubic Hermite interpolation from
\((U,F(U),Q,F(Q))\) is then extrapolated to \(-H\) to obtain \(p_*\) and its
derivative \(K_*\). For \(x=-H/\delta\),

\[
\begin{aligned}
p_*={}&(2x^3-3x^2+1)U+(x^3-2x^2+x)\delta F(U)\\
 &+(-2x^3+3x^2)Q+(x^3-x^2)\delta F(Q).
\end{aligned}
\]

Differentiating this polynomial supplies \(K_*\). No RHS evaluation uses a
synthetic past state. Every probe RHS input is checked for finite, nonnegative
values **before** evaluation; the Euler duration must also satisfy the current
state's scalar monotonicity bound. The initial probe duration is
\(\min(H/2,0.8\Delta t_{\mathrm{FE}}(U))\). An excessively large extrapolation
factor, \(H/\delta>16\), explicitly declines reconstruction and retains the
admissible baseline history. Such declines can lose accuracy and are counted;
none occurred in the reported runs.

Forward probes do not make extrapolated history automatically admissible.
Build a baseline pair \((p_0,E_0)\) from the **actual** old historical state and
Euler point, using piecewise-constant prolongation or averaging restriction.
Both transfers are positive, conservative linear maps. Choose one global
\(\theta\in[0,1]\), as large as possible subject to nonnegativity of

\[
p_\theta=p_0+\theta(p_*-p_0),\qquad
E_\theta=E_0+\theta(p_*+hv_5K_*-E_0).
\]

The same global scalar preserves the mean of each conservative history array.
An arbitrary cellwise limiter would not have this property. The next update
uses \(E_\theta\) directly, avoiding division by \(hv_5\) followed by immediate
multiplication. Both arrays are checked for finiteness and nonnegativity after
limiting. The usual current and intermediate Euler bounds are also checked
before the forward multistep RHS evaluations.

This establishes the checked scalar admissibility properties in these tests.
It is not a general SSP proof for nonlinear remapping, backward extrapolation,
or the full hydrodynamic/gravity system.

## Evidence

All problems are periodic on \([0,1]\). The default smooth runs start with 16
cells and regrid at \(t=1/8\), ending at \(t=1/4\). Refinement doubles the cell
count; coarsening halves it. Current-state prolongation is conservative minmod.
The smooth advection profile is \(2+\sin(2\pi x)+0.2\cos(6\pi x)\); the smooth
Burgers profile is \(1+0.2\sin(2\pi x)+0.05\cos(6\pi x)\). Initial data are
exact cell averages.

The spatial operator is first-order upwind flux differencing. Its forward
Euler monotonicity limit is \(\Delta x\) for unit-speed advection and
\(\Delta x/\max u\) for positive Burgers flow. Advection uses an exact Fourier
semidiscrete reference. Burgers uses DOP853 with relative and absolute
tolerances \(3\times10^{-14}\). Each reference applies the **same current-state
spatial remaps at the same prescribed times** as its numerical run, excluding
ordinary remapping error from the time-integration comparison.

At 512 initial-interval steps, the independent smooth cases gave:

| Equation | Event | Reconstruction L1 | Last time order | Minimum theta |
|---|---|---:|---:|---:|
| Advection | Refine | 4.03313e-10 | 3.001 | 1 |
| Advection | Coarsen | 1.30752e-10 | 3.000 | 1 |
| Advection | Refine and halve timestep | 2.41991e-10 | 3.002 | 1 |
| Burgers | Refine | 1.15468e-10 | 3.002 | 1 |
| Burgers | Coarsen | 2.68385e-11 | 3.001 | 1 |
| Burgers | Refine and halve timestep | 6.28273e-11 | 3.002 | 1 |

Repeated regridding tests alternate refinement and coarsening every eight
accepted steps. The number of mesh changes therefore increases as the
timestep decreases. Each resolution has its corresponding remapped reference;
this checks accumulated transition error rather than a single fixed regrid.
For nonlinear Burgers:

| Initial-interval steps | Reconstruction L1 | Last order | RK3-restart L1 |
|---:|---:|---:|---:|
| 256 | 3.45839e-9 | — | 3.33918e-9 |
| 512 | 4.78575e-10 | 2.853 | 4.61775e-10 |
| 1024 | 6.31523e-11 | 2.922 | 6.09149e-11 |
| 2048 | 8.09444e-12 | 2.964 | 7.80927e-12 |

A cheaper second-order backward reconstruction, investigated separately,
passed isolated-event third-order convergence but dropped to second order
under repeated regridding. It is not used in this prototype.

The sharp-front test starts on 32 cells with \(u=1\) on half the domain and
\(u=10^{-12}\) on the other half. With regridding every eight steps:

| Initial-interval steps | Minimum theta | Reconstruction L1 | RK3-restart L1 |
|---:|---:|---:|---:|
| 32 | 0.0592 | 2.88248e-4 | 6.30868e-6 |
| 64 | 0.2085 | 7.91948e-5 | 1.21070e-6 |
| 128 | 0.4745 | 2.20822e-5 | 2.78820e-7 |

The limiter preserves nonnegativity, but the large error increase is material:
third-order accuracy is **not** claimed where it activates. Across the default
suite, the maximum mean drift was \(6.42\times10^{-14}\), the minimum RHS input
was \(10^{-12}\), and the limited historical state remained nonnegative
(minimum \(4.00\times10^{-26}\)). The largest probe Euler ratio was 0.125.

## Cost and remaining scope

Four probe RHS evaluations are required: three for SSPRK3 and one at its
endpoint. Reusing the probe's \(F(U)\) leaves one additional RHS for the accepted
multistep transition, totaling five. RK3 restart requires three. For \(N\)
accepted steps and \(R\) regrids, the usual counts are
\(2N+1+3R\) for reconstruction and \(2N+1+R\) for restart. For the 512-step
Burgers repeated-regrid case, those totals were 2430 and 2176, respectively.

These are uniform one-dimensional scalar meshes. The experiment does not
cover local moving AMR boundaries, nonuniform-volume conservation, Euler
internal-energy/species constraints, gravity potential and energy bookkeeping,
outflows, distributed synchronization, or exact restart. The present prototype
provides prior evidence for a separate smooth-accuracy correction and its costs;
it is not the implemented production history-transfer method. Integrating this
virtual-history correction would require those checks and a justified treatment
of active limiting. The [application validation report](ssp-history-amr-validation.md)
tracks evidence for the implemented direct-transfer approach.
