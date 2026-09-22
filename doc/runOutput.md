# Run output

Octo-Tiger separates interactive progress from durable diagnostics.

The terminal prints one row for every timestep, including step 0. Rows are grouped into
refinement intervals without hiding intermediate steps. Each interval ends with a compact
mesh summary. The legacy `DWD` problem identifier is displayed as
`Binary evolution [legacy problem ID: DWD]`; the identifier itself is unchanged in input,
checkpoints, and internal code.

Two optional outputs are independent of the terminal layout:

- `--output.detailed_log=PATH` writes root-level actions, every step, interval limiter data,
  mesh counts, and analytic norms to a text log.
- `--output.results_file=PATH` writes a versioned JSON run summary for tests and automation.
  It is rewritten during the run and receives `"status": "completed"` only after normal
  cleanup. Consumers should reject other statuses.

The JSON interface uses named fields such as `lastStep.dt`, `limiter.a`,
`mesh.amrBoundaries`, and `analytic.rho.l2`. Tests must consume these fields rather than
matching human-readable terminal text. Console wording and alignment are not an API.
