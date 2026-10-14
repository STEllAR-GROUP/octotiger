# QB4 radiation matrix

## Current one-node M1 RK1 / FLD campaign

The October 14 study uses `one_node_campaign.py` in an existing single-node
Slurm allocation: one task, 64 HPX workers, CPU, physical light speed, and the
eight-problem level-2/3/4 plan with 121 states. The two completed local level-2
streaming-wave runs are preserved, leaving 46 QueenBee runs.

The verified site environment and submissions are recorded under
`output/qb4-m1-fld-20261014/`. Builds and simulations run on compute nodes.
Each `--run-limit 1` allocation exits after one validated completed result;
the workstation then imports the exact output inventory, renders the movie,
and verifies webpage publication before submitting the next single-node job.
The collector verifies the preceding job's terminal successful accounting
before allocating another node and saves idempotent submission receipts.

```sh
python3 radiation_results/qb4/check_progress.py \
  --directory output/qb4-m1-fld-20261014 --advance
```

The 30-minute heartbeat invokes the frozen collector. It reuses only the
recorded SSH master and disables fallback connections. The shared login ledger
counts every new attempt against the user's 12-attempt task ceiling. A lost
connection requires the access owner to reserve a replacement; scheduled
checks cannot authenticate another connection themselves.

Imported outputs require pinned source, executable, library, configuration,
job/node and file hashes. Local solver `STOP` remains present. Publication-only
execution uses `POSTPROCESS_STOP` as its separate cancellation marker, preserves
all movie/profile samples and endpoints, and removes intermediate local 3D
states only after verified extraction. QueenBee raw output remains intact.

The single partition permits seven-day allocations. A pre-timeout signal
preserves incomplete output and marks `needs_checkpoint_continuation`; it does
not silently restart a long calculation from zero. Automatic multi-segment
checkpoint continuation is not implemented by this worker. Numerical failures
and publication failures require inspection before another attempt.

## Historical matrix workflow

This workflow prepares, builds, tests, runs, and renders radiation problems at
levels 2–5 (finest equivalent 32³, 64³, 128³, 256³ with INX=8). It supports
the full seven-case matrix and explicitly recorded subsets. Preparation is offline and never
calls SSH, Slurm, or the solver. Submission is a separate, explicit action. These
scripts do not open remote connections or upload/publish a website.

## Site configuration and environment

Copy `site.example.json` to a site configuration and replace its paths and
`EDIT_ME` fields. All source, build, output, environment, and plan paths must be
on shared storage visible to every allocated node. Prepare the plan in its final
location on that shared filesystem; regenerate it if paths change.

For the current phased gradient-AMR study, start with `site.amr.example.json`.
It selects all seven L2 cases, the L3 blast, and all four transport cases at
L3–5: 20 full-duration runs. The eight remaining coupled case/level pairs are
deferred pending new AMR timing evidence. It retains factors α=1 for the blast,
α=0.1 for the supercritical shock and α=0.03 for the subcritical shock; these
are study settings, not demonstrated factor-converged solutions. All analytic
transport cases use physical c. This example requests nine snapshots and a
two-day run walltime, which does not guarantee a case will finish.

`amr_plan` names `radiation_results/amr/local-thresholds.json`, relative to
`source_root` or as an absolute path. Preparation embeds its entire contents
and a canonical JSON SHA-256 in `plan.json`. Workers use that frozen copy;
later changes to the original file do not silently alter submitted jobs. Each
run uses allowed leaf levels L−1 and L, the selected gradient field, mode,
threshold, floor, buffer and coarsening factor. The local transport profiles
use half-width 1 while the CGS results runner uses half-width 3×10¹⁰ cm:
both absolute and relative gradient thresholds are divided by 3×10¹⁰ exactly
once. Coupled profiles are already CGS. Field floors are unchanged.

The saved tuning evidence consists of short local runs and unrun proposals,
not full-duration mesh or accuracy validation. No quota removes flagged cells
to achieve the desired fine/coarse cell ratio. All-fine coarse exceptions,
geometry-imposed ratio misses and later mesh growth remain visible.

Use the single dependency environment template
[`tools/qb4/environment.sh.example`](../../tools/qb4/environment.sh.example) and
[build instructions](../../tools/qb4/README.md). The `environment_file` setting
points to your edited copy. Build and run jobs source that same file. It must
select a consistent compiler/MPI stack, an HPX **2.x** installation built with
networking and the MPI parcelport, application dependencies, and the results-tool
dependencies (C++20 compiler, Silo, FFTW3, OpenSSL, nlohmann_json, and gnuplot).
A local TCP-only HPX installation cannot launch these distributed jobs.

Specify the account, partitions, walltimes, and the installed Slurm MPI plugin
explicitly. `mpi_type` becomes `srun --mpi=...`; the script does not guess a PMIx
version or silently fall back to another MPI installation. `workq` and `single`
in the example are starting choices, not a live availability or quota check.
Historical local QB4 jobs and the public hardware description indicate 64 CPU
cores and 256 GB per CPU node. The example uses one 64-worker HPX locality per
simulation node, with the following initial allocation schedule:

| Level | Cells | Nodes | HPX localities | Workers per locality |
|---:|---:|---:|---:|---:|
| 2 | 32³ | 1 | 1 | 64 |
| 3 | 64³ | 1 | 1 | 64 |
| 4 | 128³ | 2 | 2 | 64 |
| 5 | 256³ | 4 | 4 | 64 |

These are configurable starting allocations, not measured scaling optima or an
account limit. Build and final plotting jobs request 8 and 2 CPUs respectively,
using separately configurable `build_partition` and `plot_partition` settings.
The example uses `single` for those smaller jobs. Helper libraries use one thread.
Every worker checks that it is executing on the allocated compute host, not in
an interactive allocation's login shell.
The test gate requests one node and four CPUs, running CTest serially because
the registered AMR application check itself uses four HPX workers. The selected
CMake Python interpreter must be at least 3.8; set `Python3_EXECUTABLE` in the
site environment (the current site environment supplies Python 3.11).

## Prepare, inspect, and submit

From the checkout, preparation alone is:

```sh
python3 radiation_results/qb4/workflow.py prepare \
  --config /absolute/shared/path/site.json \
  --directory /absolute/shared/path/radiation-plan
```

The uniform full-matrix default writes `plan.json` and 31 job scripts; the phased
AMR example writes 26 (build, tests, four timings, 20 runs). Preparation prints each case's node allocation,
physical duration, and transport timestep estimate. The directory must be empty.
The four radiation-only problems use 4 CGS seconds by default. The radiating
blast, supercritical shock, and subcritical shock use their full template times
of 22000, 7500, and 38300 seconds. Physical-light-speed evolution is the default.

`reduced_c_factors` optionally selects an independent RSLA factor for each
coupled case (`radiating_blast`, `supercritical_shock`, `subcritical_shock`).
Every factor must satisfy 0 < α ≤ 1; omitted entries and the uniform example values are
1. The four analytic transport cases always use α=1. A smaller value uses the
Skinner-Ostriker reduced-speed radiation equations while preserving physical c
in gas source terms and the F/c closure. It is an approximation requiring
factor-convergence studies, not an accuracy-preserving runtime switch. Plan
output, run configuration, metadata, and webpages record the selected factor
and both physical and transport speeds. Prepare a new plan after changing it.

After reviewing the plan, the explicit submission command is:

```sh
python3 radiation_results/qb4/workflow.py submit \
  --plan /absolute/shared/path/radiation-plan/plan.json --submit
```

This submits the build, test gate, optional timing jobs, selected simulations,
and (only for a complete 28-case matrix) collection. Jobs are
chained with `afterok`, so only one job in this workflow can run at a time. The
simulation order is level-major: at each level the four transport cases precede
blast, supercritical shock and subcritical shock. `run_selection`, when present,
is a nonempty list of unique `{"case": "...", "level": N}` objects; omitted
means all 28. A subset explicitly records `matrix_scope: selected-subset` and
never schedules final full-matrix collection. Remove the selection and prepare
a new plan to request the entire matrix; do not modify an already prepared plan.

A submission record is reserved before the first `sbatch` call and updated after
every submission. A failed or ambiguous submission stops immediately. A second
submission against the same plan is refused; inspect `submission.json` and the
scheduler manually before deciding how to recover. There are no polling,
automatic resubmission, or requeue loops. Do not invoke the generated job scripts
directly on a login node.

Raw scheduler stdout/stderr is saved before parsing. The parser accepts the
observed QB4 `sbatch:` accounting/Lua announcements followed by exactly one
final job ID (optionally `;cluster`). Announced IDs must agree with that ID;
unknown, conflicting, or malformed output is retained as `submission-ambiguous`
and stops submission without retrying an accepted or uncertain job.

## Build, output, and recovery

The build job calls the checkout's `build.sh` and produces:

- `build-qb4-radiation/octotiger`
- `build-qb4-radiation/radiation-results-tools/radiation-results`
- `build-qb4-radiation/radiation-results-tools/gen_radiation_reference`

`build_dir` can override the prefix. The application build fixes INX=8. Each
simulation invokes the existing C++ results runner with `--launcher mpi-launch.sh`.
The launcher wraps only Octo-TIGER in `srun`; reference generation and plotting
run once in the job's batch process. Metadata hashes the actual solver binary,
the launch script, and the relevant source files. MPI bootstrap is explicit,
there is one locality per node, and worker count matches `--cpus-per-task`.
The separate test gate runs solver tests labeled `radiation` or `gradient`
(including the six gradient suites and relevant application/restart checks),
then results-tool tests named `radiation-results-*`. It saves logs under
`output_root/workflow-status/`; unrelated repository test suites are excluded.
Every worker writes its command, allocation, frozen-plan digest and status in
that directory. A process killed by the scheduler may leave `running` there;
that is incomplete evidence, never a completion marker. No code infers success
from the mere presence of a checkpoint or an output directory.

Each case/level writes to `output_root/batches/CASE-lLEVEL/`. It produces its own
webpages and plots after successful completion. Per-locality stdout/stderr is
under `CASE/lLEVEL/mpi-logs/`; unprefixed locality-zero stdout is also retained in
`run.log` for the normalization and comparison readers.

The final collection requires all 28 planned results to be complete at their
requested times and resolutions, with compatible units and provenance, and
conservation output. It then builds `output_root/site/index.html` with all seven
problems and all four resolutions. The standalone site uses real files and
relative links; it does not include Silo snapshots. A failed or timed-out job
prevents dependent jobs and final collection from declaring success. Individual
completed batch pages remain available for inspection.
Collection refuses an existing `output_root/site`; explicitly preserve or rename
that directory before generating another site. Original batch manifests and run
metadata are copied unchanged into the site for provenance.
For adaptive runs, collection also requires the exact frozen gradient settings,
two adjacent allowed levels and `gradient-amr.csv`. Every recorded census must
cover the domain exactly, with 512 cells per leaf block. Native leaf CSVs and
the census are preserved; the C++ readers verify dyadic slice tiling and render
leaf averages onto the finest display grid. Observed ratios are reported but
are not treated as physical accuracy acceptance thresholds.

Three Silo snapshots are requested per simulation by default. Increase
`snapshots` only with a storage budget: 256³ grids contain many gas/radiation
fields, and the 28-run matrix can generate substantial data. `no_silo: true`
skips Silo while retaining final numerical slices and still PNG/PDF/CSV plots.
Movies are disabled, so VisIt and FFmpeg are not required. Logs and conservation
histories can also be large during multi-million-step shock runs.
An optional `checkpoint_odt` object maps keys such as `subcritical_shock-l2`
to checkpoint spacing in physical seconds. This overrides the runner's snapshot
cadence for that case. Neither a requested snapshot count nor a physical-time
spacing guarantees a checkpoint before a walltime limit; use measured advance
rates when choosing the spacing. Checkpoints and partial files remain in raw
batches and are excluded from the standalone website.

At level 2, the light-speed CFL ceilings imply at least approximately 12664 blast
steps, 1233432 supercritical steps, and 6298726 subcritical steps. Each refinement
level doubles those minimum step counts and multiplies cell count by eight.
These are physical-c timestep estimates, not runtime predictions; the chosen
walltime may not suffice. Selecting α<1 multiplies the transport step estimate
by α, but gas CFL constraints and other timestep limits can still dominate.
For adaptive plans, the reported lower bound permits every leaf to become
coarse; `finest_transport_steps_estimate` separately records the estimate when
at least one finest-level leaf remains. AMR does not automatically remove the
finest-level CFL limit, and cell-count savings are not a measured speedup.
RSLA budgets check the modified equations, whose closed-system invariants are
Egas + Er/α and gas momentum + F/(αc²), rather than total physical energy.

The existing batch `--resume` feature reuses complete runs and starts untouched
runs. It **does not continue a partially completed simulation from Silo** and
refuses existing incomplete run directories. MPI checkpoint continuation of
these coupled cases has not been validated. Preserve failed-run files and plan
recovery explicitly; this workflow does not silently restart or overwrite them.

## Short timing samples

`timing_samples: true` adds four allocated timing jobs after the test gate and
before the full-duration cases. Each runs all three coupled cases for 20 steps
at one level using the frozen AMR plan and the same configured RSLA factors.
Results are written under `output_root/timings/lN/`; each includes observed
mesh histories and occupancy misses. `timing_walltime` defaults to the run
walltime if omitted. The phase does not automatically add deferred long runs
or decide that a selected full-duration run will fit its allocation.

The equivalent command inside the matching allocation is:

```sh
python3 radiation_results/qb4/benchmark.py --plan /absolute/shared/path/radiation-plan/plan.json --level 2
```

Inside a compute allocation with the matching dependency environment sourced
and `QB_MPI_TYPE` set, the legacy uniform timing mode remains available:

```sh
python3 radiation_results/qb4/benchmark.py \
  --source /absolute/shared/path/octotiger \
  --build /absolute/shared/path/octotiger/build-qb4-radiation \
  --output /absolute/shared/path/new-timing-l2 --level 2
```

This executes 20 timesteps (`stop_step=19`) for each coupled problem, with no
Silo, analytic comparison, or legacy diagnostics. Final numerical CSV files
remain available. It uses the existing MPI launcher and the job's allocated
node/task/worker counts. The output directory must not exist. Benchmark defaults
are α=0.03 for subcritical, α=0.1 for supercritical, and α=1 for the blast;
`--subcritical-factor`, `--supercritical-factor`, and `--blast-factor` override
them. These timing defaults do not change the full-matrix defaults of α=1.

`benchmark.json` records actual physical time, exactly verified executed steps,
startup-inclusive wall time, resource allocation, configuration, source and
executable hashes. Its full-duration cost estimate extrapolates only the initial
advance rate and can be inaccurate as the flow develops. A timing sample does
not validate physical accuracy or RSLA factor convergence, and does not claim a
completed full-duration benchmark. Failed samples retain their logs and metadata
and prevent subsequent cases from starting. This script neither submits nor
queries Slurm.

## Offline validation

```sh
python3 radiation_results/qb4/test_workflow.py
python3 radiation_results/qb4/test_collect.py
python3 radiation_results/qb4/test_benchmark.py
bash -n radiation_results/qb4/mpi-launch.sh
```

The workflow tests mock every `sbatch` call and put a local fake `srun` first on
PATH. They verify the matrix, serial dependencies, explicit submission,
resource mapping, dry preparation, login-host rejection, and clean solver logs.
They do not establish QB4 MPI runtime correctness or full-duration physical
accuracy. The collector tests use synthetic files and mocked rendering.
