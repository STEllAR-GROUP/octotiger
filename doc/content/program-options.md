# Program Options

## Optional gradient refinement

`gradient_refine_field=none` is the default. Existing problem-specific density,
hydrodynamic, and gravity refinement criteria remain active without modification.
To replace that criterion with gradient tagging, set `unigrid=off` and select
`density` (total mass density) or `radiation_energy` (requires `radiation=on`).
Selecting a field together with `unigrid=on` is rejected instead of silently
ignoring the criterion. No shadow-grid evolution is used.

```ini
unigrid=off
min_level=4
max_level=5
gradient_refine_field=radiation_energy
gradient_refine_mode=relative
gradient_refine_threshold=10
gradient_refine_floor=1e-30
gradient_refine_buffer=0
gradient_refine_coarsen_factor=0.8
```

The example threshold is illustrative; physical field and coordinate units and
the problem determine its useful value. For every inspected cell and each of
its six face neighbors, the indicator is the maximum of:

- `absolute`: `abs(q_neighbor - q_cell) / dx`, in field units per code length;
- `relative`: `abs(log(max(q_neighbor,floor)) - log(max(q_cell,floor))) / dx`,
  in inverse code length.

These are one-sided differences, so an isolated peak is detected even when
opposite neighbors have equal values. The gradient is not multiplied by the
cell width, and it is not a Euclidean norm. `gradient_refine_floor` has the
selected field's units, only affects the logarithmic mode, and does not change
the evolved field. Threshold and floor must be finite and positive.

Any tagged cell refines its entire block. `gradient_refine_buffer` additionally
inspects zero (default), one, or two face-halo cell layers. A refined parent
remains refined while its indicator exceeds `gradient_refine_threshold` times
`gradient_refine_coarsen_factor` (default 0.8; allowed range `(0,1]`). Decisions
are evaluated at the parent's resolution. `min_level` is always enforced and
the new criterion never refines past `max_level`. Set `min_level=max_level-1`
to allow two adjacent active leaf levels; the retained internal tree nodes do
not count as leaves. Neither both-level occupancy nor a prescribed cell-count
ratio is guaranteed by a fixed physical threshold.

`gradient-amr.csv` records physical simulation code time, step, level, leaf
blocks, and leaf cells initially and after each evolution regrid, even with
other diagnostics disabled. Each block contains `INX^3` cells. For a desired
fine/coarse **leaf-cell count** ratio `R`, a fraction `R/(8+R)` of the coarse
blocks must be refined in three dimensions; block quantization can make a
requested ratio impossible. Uniform finest meshes are valid outcomes.

Enabled gradient options and level bounds are saved in checkpoint metadata.
A restart of such a checkpoint restores those settings. Older checkpoints
without these fields retain the supplied configuration (default: disabled).

`radiation_reduced_c_factor` is an opt-in ratio of the radiation transport
light speed to physical `c`. The default is `1` and retains the full-speed
model. Values must be finite and satisfy `0 < factor <= 1`. For example:

```sh
octotiger --config_file=run.ini --radiation_reduced_c_factor=0.1
```

The same name is accepted in configuration files. It does not change the
physical light-speed constant or the physical units of stored radiation flux.
Use a reduced factor only when its approximation is appropriate for the
problem and check sensitivity to that choice.

The factor is distributed with the other options and saved in radiation
checkpoint metadata version 2. A restart restores the saved factor even if
the command line or configuration requests a different value. Version 1
and older checkpoints use factor `1`. A version 2 checkpoint with a missing
or invalid factor is rejected.
