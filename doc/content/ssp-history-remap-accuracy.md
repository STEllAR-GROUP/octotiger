# Accuracy of SSP history transfer across regridding

At fixed old and new spatial resolutions, conservatively transferring the
previous state and its complete derivative preserves mass but does **not
generally** retain third-order temporal accuracy across a change of spatial
operator. This controlled advection experiment gives first-order time
convergence after either refinement or coarsening.

That criterion is not sufficient by itself to reject practical AMR history
transfer. The [joint space/time convergence experiment](ssp-history-joint-convergence.md)
shows that the original transfer prototype and RK3 restart have matching
observed orders and closely matching errors when the mesh and timestep are
refined together, including errors immediately around mesh changes.

Direct history transfer is implemented in Octo-Tiger. The scalar results below
are prior evidence for this work; the [application validation report](ssp-history-amr-validation.md)
records the separate application evidence and its limits. See the
[implementation description](ssp-multistep.md) for the current algorithm. The [virtual-history reconstruction experiment](ssp-history-reconstruction.md)
examines a separate way to recover the stronger fixed-mesh temporal property.

## Reproduce

From the repository root, with Python and NumPy available:

```sh
python3 test_problems/ssp_multistep/history_remap_accuracy.py --json /tmp/ssp-history-remap.json
```

The script limits NumPy/BLAS to one thread. It reports independent unchanged-mesh,
refinement, and coarsening cases, each with both history transfer and RK3 restart.
Optional checks are:

```sh
python3 test_problems/ssp_multistep/history_remap_accuracy.py --prolongation linear
python3 test_problems/ssp_multistep/history_remap_accuracy.py --events refine --halve-on-refinement
python3 test_problems/ssp_multistep/history_remap_accuracy.py --operator upwind3
```

This is an accuracy experiment, not a registered runtime regression test. It
prints the observed errors/orders without treating failure of the stronger
fixed-space temporal criterion as a production regression failure.

## Setup and results

The criterion in this report is third-order temporal convergence with the
old/new mesh resolutions and physical regrid time held fixed. Its reference includes the
same spatial transfer, so ordinary current-state remapping error is excluded.
This gate does not assert loss of conservation or necessarily reduced combined
space/time convergence.

Solve periodic scalar advection, \(u_t+u_x=0\), on \([0,1]\), starting with exact
cell averages of \(2+\sin(2\pi x)+0.2\cos(6\pi x)\). Initially there are 32 cells.
At \(t=1/8\), an independent case either refines to 64 cells, coarsens to 16, or
keeps the mesh unchanged. Each case ends at \(t=1/4\).

The primary operator is first-order upwind finite volume, whose forward-Euler
update is monotone for \(h/\Delta x\leq1\). All default runs satisfy the Euler
bounds in the multistep decomposition. This low-order spatial operator is
intentional: an exact Fourier exponential propagates its semidiscrete ODE on
each fixed-mesh interval, removing spatial truncation error from the temporal
comparison. The reference applies exactly the same current-state remap at the
same prescribed physical time.

Coarsening averages adjacent cells. Refinement uses conservative piecewise
linear reconstruction: the children are \(U_i\mp s_i/4\), with minmod slope
\(s_i\). This is the one-dimensional reduction of Octo-Tiger's current
prolongation. The historical derivative uses the historical state's frozen
limiter stencil, so both are transferred by a common linear map \(P\).
Independent limiting of the derivative is not responsible for the failure.
Every run takes an initial SSPRK3 step; the restart comparison also takes one
SSPRK3 step immediately after its mesh change. Both otherwise use the same
production \(c_3=0\), two-step, two-stage coefficient family.

Measured mean absolute errors for the primary minmod/upwind experiment:

| Event | History handling | 256 steps/interval | 512 | 1024 | Last order |
|---|---|---:|---:|---:|---:|
| Refine | Transfer state and RHS | 2.03487e-5 | 1.01756e-5 | 5.08795e-6 | 1.000 |
| Refine | RK3 restart | 1.02331e-8 | 1.27883e-9 | 1.59837e-10 | 3.000 |
| Coarsen | Transfer state and RHS | 3.14657e-5 | 1.57322e-5 | 7.86604e-6 | 1.000 |
| Coarsen | RK3 restart | 3.16574e-9 | 3.95643e-10 | 4.94489e-11 | 3.000 |

Common centered linear prolongation gives the same first-order failure.
A third-order upwind spatial operator and halving the timestep on refinement
also reproduce it. Conservation remains at roundoff. The unchanged-mesh
control retains third-order temporal convergence.

## Leading error at a mesh change

Let \(F_o\) and \(F_f\) denote the old and new semidiscrete operators, and
\(P\) a fixed linear conservative remap (as in the centered-linear test). For equal consecutive timesteps \(h\), transfer
an otherwise exact history:

\[
U=P U_o^n,\qquad
U_-=P U_o^{n-1}=U-hP F_o(U_o^n)+O(h^2),\qquad
K_-=P F_o(U_o^{n-1}).
\]

Expanding the first new-mesh multistep update about \(U\), and subtracting exact
new-mesh evolution from that same state, gives

\[
U_{\mathrm{MS}}^{n+1}-U_{\mathrm{exact}}^{n+1}
=hB\left[P F_o(U_o^n)-F_f(P U_o^n)\right]+O(h^2),
\qquad B=-c_4c_2+c_5(v_5-1).
\]

For the selected coefficient family at \(\alpha=1\),
\(B=-0.33974596215561387\). Conservation of \(P\) does not make the bracket
vanish: a transferred coarse-grid derivative generally differs from the
fine-grid derivative of the transferred state. The same issue occurs under
restriction. Direct small-step calculations with linear transfer agree with
this coefficient. Recomputing only the historical derivative does not repair
the historical state's incompatibility with the new evolution operator.

## Scope

This evidence concerns temporal accuracy at fixed old/new mesh resolutions.
The [joint convergence report](ssp-history-joint-convergence.md) separately
includes current-state projection error and refines space and time together.
It finds matching transfer/restart orders with independent refinement and
coarsening, frequent regridding, maximum errors over time, shifted meshes, and
timestep halving. Thus the temporal criterion here was overly strong as the
sole reason to reject the inexpensive transfer approach. Passing that joint
comparison does not establish third-order time accuracy through a change of
fixed semidiscrete operators; the two results address different limits.

These are standalone scalar experiments. They establish a counterexample to
a generic fixed-mesh temporal-order claim and provide prior evidence about
the direct-transfer approach now implemented in Octo-Tiger. Neither scalar
report is an Octo-Tiger hydro/gravity run, a local moving refinement-boundary
test, a checkpoint test, or a positivity proof. Application validation is
tracked in the [application validation report](ssp-history-amr-validation.md). A compatible history
reconstruction or operator correction is necessary if the stronger fixed-mesh
temporal property itself is required, but is not established as a prerequisite
for retaining joint space/time convergence.
