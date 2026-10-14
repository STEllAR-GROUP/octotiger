# Joint space and time convergence of transferred SSP history

Conservatively transferring the old state and its complete RHS retains the
observed joint space/time convergence in the scalar controls below. Its errors
and orders closely match an SSPRK3 restart at each mesh change. These results
provided prior scalar evidence for implementing the inexpensive transfer approach.

The [fixed-mesh temporal experiment](ssp-history-remap-accuracy.md) answers a
different question: at fixed spatial resolutions, the transferred history is
inconsistent with exact evolution under the new semidiscrete operator, and its
first outgoing step has a first-order temporal defect. That result alone was
an overly strong reason to reject the practical AMR proposal. It does not
establish a loss of the fully discrete method's joint convergence when the
mesh and timestep are refined together.

Direct history transfer is implemented in Octo-Tiger. See the separate
[application validation report](ssp-history-amr-validation.md) for application
evidence and its limits, and the [implementation description](ssp-multistep.md)
for the current algorithm. The [virtual-history reconstruction
experiment](ssp-history-reconstruction.md) concerns a separate attempt to
recover the stronger fixed-mesh temporal property.

## Reproduce

From the repository root, with Python and NumPy available:

```sh
python3 test_problems/ssp_multistep/history_joint_convergence.py --json /tmp/ssp-history-joint.json
```

The default uses coarse grids of 32, 64, and 128 cells, both centered and minmod
prolongation, and all seven event schedules. NumPy/BLAS is limited to one thread.
The script imports the common coefficient, remapping, and stepping routines
from `history_remap_accuracy.py` in the same directory.

The recorded results below were reproduced with the packaged script:

```sh
python3 test_problems/ssp_multistep/history_joint_convergence.py \
  --grids 16 32 64 128 256 --json /tmp/ssp-history-joint-full.json
```

Use `--cases`, `--prolongations`, and `--operator` to select controls, and
`--all-rows` to print every resolution. For example:

```sh
python3 test_problems/ssp_multistep/history_joint_convergence.py \
  --operator upwind1 --cases none refine coarsen --grids 32 64 128 256
python3 test_problems/ssp_multistep/history_joint_convergence.py \
  --cases shifted-every8 refine-halved --prolongations minmod --all-rows
```

The JSON contains every resolution, both error norms, errors after remapping
and after the first outgoing step, conservation diagnostics, RHS counts, and
the difference between transfer and restart. These are convergence experiments,
not registered Octo-Tiger runtime regression tests or hard-coded pass/fail
thresholds on observed orders.

## Reference and schedules

The problem is periodic scalar advection, \(u_t+u_x=0\), on \([0,1]\), with
exact initial cell averages of
\(2+\sin(2\pi x)+0.2\cos(6\pi x)\). Every run ends at \(t=0.5\).
The reference is the analytic continuum solution averaged over the current
cells. **Current-state projection errors are included.**

For coarse resolution \(N\), the fine mesh has \(2N\) cells. Ordinary cases
use \(h=0.2/(2N)\) throughout, holding the finest-grid CFL fixed as both space
and time are refined. Their coarse-grid CFL is 0.1. The separate
`refine-halved` case uses CFL 0.2 on each mesh and actually halves the timestep
at refinement, giving \(\alpha=2\) for the first outgoing multistep step.

The schedules are:

- `none`: unchanged coarse mesh.
- `refine` and `coarsen`: independent mesh changes at \(t=0.25\).
- `fixed-cadence`: alternating refinement/coarsening at \(t=0.1,0.2,0.3,0.4\).
- `every8`: alternating changes every eight steps; the number of changes grows
  as the mesh is refined.
- `shifted-every8`: the same schedule, but each coarsening shifts the coarse
  grid by one fine cell. The reference follows the translated grid. This
  checks that returning to the same aligned projection grid is not necessary
  for the observed convergence.
- `refine-halved`: one refinement at \(t=0.25\), with timestep halving.

Restriction averages adjacent cells. Centered prolongation and minmod
prolongation are conservative. The old RHS uses the old state's frozen
minmod stencil, so this experiment does not independently limit the state
and derivative. The current state has its own normal remap. Every run takes
one initial SSPRK3 step; only the restart comparison repeats it at mesh changes.

## Recorded results

The following errors use third-order upwind spatial differencing. This
operator is a smooth spatial-accuracy control; it has **no forward-Euler
positivity guarantee**, so these results are not an SSP proof.

The table reports the maximum \(L_1\) error over the complete run, including
immediately after current-state remapping and after every numerical step.
Errors are at \(N=256\); orders use \(N=128\to256\).

| Prolongation | Schedule | Transfer error | Order | RK3 restart error | Order |
|---|---|---:|---:|---:|---:|
| Centered | One refinement | 2.253328e-5 | 2.9993 | 2.253904e-5 | 2.9997 |
| Centered | One coarsening | 2.249240e-5 | 2.9978 | 2.248663e-5 | 2.9974 |
| Centered | Fixed physical cadence | 2.606670e-5 | 3.0020 | 2.606668e-5 | 3.0020 |
| Centered | Every 8 steps | 2.568107e-5 | 2.9967 | 2.568179e-5 | 2.9967 |
| Centered | Shifted every 8 steps | 2.648212e-5 | 2.9930 | 2.647989e-5 | 2.9930 |
| Centered | Refinement with timestep halving | 2.264413e-5 | 2.9993 | 2.264917e-5 | 2.9996 |
| Minmod | One refinement | 9.328163e-5 | 2.0354 | 9.328163e-5 | 2.0354 |
| Minmod | Fixed physical cadence | 9.381164e-5 | 2.0273 | 9.381164e-5 | 2.0273 |
| Minmod | Every 8 steps | 1.897563e-4 | 1.9571 | 1.898364e-4 | 1.9571 |
| Minmod | Shifted every 8 steps | 2.205988e-4 | 1.8990 | 2.205265e-4 | 1.8992 |
| Minmod | Refinement with timestep halving | 9.328688e-5 | 2.0357 | 9.328688e-5 | 2.0357 |

The minmod/coarsening-only case matches the centered/coarsening-only case,
because that schedule uses restriction without prolongation. The unchanged
mesh control has error 3.994761e-5 and order 2.9966 for both history strategies.

Final-time errors alone can hide remap transients. With minmod and one
refinement, the final \(L_1\) errors are 2.285785e-5 and 2.286359e-5, both
appearing third order. The maximum errors over time in the table are instead
second order and identical for transfer and restart: the current-state remap
causes a transient that subsequently damps. Immediately after the first
outgoing step, their errors are 6.536111e-5 and 7.184124e-5, with orders 2.0881
and 2.0682. Thus the lower order is present in both strategies, rather than
being introduced by retaining history.

Centered transfer retains approximately third-order errors both at events and
over the entire run. Minmod's lower order is also shared by both methods in
\(L_\infty\), which can converge more slowly than \(L_1\) under repeated
limiting. No third-order claim is made for the limited remap.

Across the packaged full upwind3 run, the maximum mean drift is 5.284662e-14
and the minimum evolved value is 0.8867436. Separate first-order upwind controls
also give matching transfer/restart errors as they approach their spatial
first-order limit; the high-frequency harmonic is not fully asymptotic on the
smaller grids. Positivity of these particular smooth scalar runs does not
establish a general positivity or SSP property of the transfer.

## What this establishes

The original history-transfer prototype passes this joint convergence
comparison against RK3 restart, including independent refinement/coarsening,
errors at mesh changes, frequent regridding, translated projection grids, and
a timestep decrease at refinement. The operator mismatch from the temporal
experiment can shrink with mesh size, so its fixed-space temporal defect
need not lower the relevant joint accuracy.

This is prior scalar evidence for the implemented direct-transfer approach,
not a proof for arbitrary operators, remaps, or regrid schedules. All meshes
here are globally uniform between events. These experiments do not cover
local moving coarse/fine interfaces, Euler hydrodynamics, shocks, the equation
of state, coupled self-gravity, boundary outflows, checkpoints, or distributed
execution. They also do not exercise the production block-boundary history
reconstruction and coupled state/RHS admissibility limiter. See the
[application validation report](ssp-history-amr-validation.md) for application evidence. The
fixed-mesh temporal limitation and these joint convergence results address
different limits and both remain relevant.
