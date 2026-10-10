# Local gradient AMR study

This folder prepares initial-profile threshold proposals and checks short,
actual application runs. It does not start remote work or run the benchmarks to
their published final times. Solver gradient refinement remains opt-in.

```sh
python3 radiation_results/amr/study.py prepare --output /tmp/radiation-amr-plan
python3 radiation_results/amr/test_study.py
python3 radiation_results/amr/study.py run \
  --plan radiation_results/amr/local-thresholds.json \
  --executable cmake-build-hpx2rc-radiation/octotiger \
  --generator cmake-build-hpx2rc-radiation/tests/radiation/gen_radiation_reference \
  --cases equilibrium_sphere --levels 4 \
  --output /tmp/radiation-amr-smokes --threads 4 --timeout 180
```

`initial-thresholds.json` retains the original offline 28-row proposals.
`local-thresholds.json` is the practical plan with the stronger L4 blast
threshold and per-row links to the completed local smoke evidence. Untested rows
remain explicitly labelled offline proposals. The `prepare` command produces a
new offline plan; it does not automatically replace either saved artifact or
repeat the evolved threshold tuning.

The example exercises a genuinely mixed level-3/4 mesh. Use `--levels 2` for a
smaller all-fine coarse exception. Omitting the case/level filters requests all
28 pairs, including cases whose estimated storage exceeds the local memory cap;
the saved practical plan labels those limits. Runs are sequential; the default
is one HPX worker (the example selects four), six solver steps,
physical light speed, and adjacent active levels `L-1,L`. Output directories must
not already exist. Failures retain their logs and are never labelled successful.
The preparation process pins itself and its compiler/sampler to one CPU.
The local application monitor stops a process above a sampled 16 GiB RSS budget,
or half the memory available at launch, whichever is smaller. It records sampled
peak RSS on success. `--memory-limit-gib` sets a smaller or explicitly reviewed
larger cap; it does not override the half-available-memory cap. This bound can
prevent expensive higher-level cases from completing; they remain failed, never
reported as successful validation.

`sample_initial.cpp` calls the shared gradient kernel and existing initial-profile
headers. Radiation energy is the default field; `--blast-density` selects density
for the blast. Absolute thresholds have field/code-length units; relative
thresholds have inverse-code-length units and use the specified field floor.
Fixed-medium tests use `code_to_cm=physical_c`, giving physical `c=1` in code units;
the coupled tests keep CGS code units. The Gaussian sampler uses the generator's
exact periodized initial cell integrals; it does not reproduce FFT roundoff.
Quadrature symmetry caching changes cost only, not the sampling resolution.
The website runner instead stretches the four transport domains from `xscale=1`
to `xscale=3e10` in CGS while retaining their numerical energy amplitudes. Divide
these four cases' absolute or relative thresholds by `3e10` for that runner.
Coupled shock/blast thresholds already use its CGS coordinates. This conversion
applies to gradients per length; a field floor is not divided by domain size.

The offline counts are proposals, not evolved AMR results or accuracy tests.
Initial parent restriction, hysteresis, interpolated AMR ghosts, evolution, and
regridding can change application counts. `gradient-amr.csv` is the actual
mesh evidence. The checker requires two adjacent active levels, whole eight-child
families, exact domain coverage, an evolved regrid sample, and a closing radiation
energy ledger. It reports every observed mesh ratio, including failures to reach
the target. It does not claim solution accuracy from these checks.

The requested ratio is **fine leaf cells / coarse leaf cells**, not fine volume.
For `B` base blocks and `m` refined base blocks, the ratio is `8m/(B-m)` and the
fine volume fraction is `m/B`. The interval `[0.1,1]` corresponds to fine volume
fractions of approximately 1.23%–11.11%. Refinement selects entire blocks. At
finest level 2 there are only eight base blocks: one refined block already gives
`8/7`, so no nonempty mixed mesh can meet the interval. Symmetry or a planar front
can impose stronger constraints. Threshold proposals never break tied blocks to
meet a quota. The user-permitted all-fine exception is labelled separately and
defaults to finest levels 2 and 3 (`--allow-all-fine-through`).

Review a proposed threshold before using it as a scientific setting. A threshold
that attains an occupancy target can still omit important weak features. The
same threshold must subsequently be checked against evolved profiles and uniform
reference runs; these short checks do not provide that validation. If no threshold
meets the target, the initial plan uses all nonzero gradients at coarse levels,
or the complete strongest-gradient group at higher levels, and explicitly records
the unmet target. No solver-side cell quota is imposed.
