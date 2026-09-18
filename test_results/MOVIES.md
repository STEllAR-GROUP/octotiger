# Radiation movies

Movie mode runs all selected tests at level 2, then level 3, then level 4.
After each completed simulation it makes an MP4, so you can watch the smaller
runs while the larger runs are still pending. The usual error plots are retained.
All four supplied test configurations use `omega=0`.

## Install the update and run

From your Octo-TIGER checkout, with the existing Python environment active:

```bash
cd ~/workspace/octotiger
unzip -o ~/Downloads/radiation-movies-update.zip -d .
command -v visit ffmpeg
bash test_results/run.sh all 2 3 4 release --movies
```

Line by line:

1. `cd` enters the checkout, where `test_results/` and `release/` live.
2. `unzip` installs the updated scripts. `-o` permits replacing these script
   files; `-d .` puts them in the current directory. The update archive contains
   no simulation results, virtual environment, or C++ source files.
3. `command -v` prints the paths of VisIt and FFmpeg if your shell can find them.
4. `bash` runs the existing launcher. `all` selects the four tests; `2 3 4`
   selects refinement levels; `release` selects your build; `--movies` enables
   snapshot capture and MP4 generation.

VisIt must support its command-line interface, and FFmpeg must have the
`libx264` encoder. These are separate applications, not new pip requirements.
The runner checks for them before building or starting simulations. Supply
`--visit /path/to/visit --ffmpeg /path/to/ffmpeg` if needed. Run VisIt on a
machine where its headless rendering works; `-nowin` is supplied automatically.
Your existing CMake configuration, including the CUDA language setting, is used.
This movie update requires no additional C++ patch.

## Capture and playback are separate

| Setting | Default | Meaning |
|---|---:|---|
| `--movie-snapshots` | 61 | Target count of distinct Silo states, including initial and final |
| `--movie-seconds` | 20 | Total playback duration in seconds |
| `--movie-fps` | 30 | Encoded frames per playback second |
| `--movie-hold` | 1 | Pause at each endpoint, included in total duration |
| `--time` | 0.2 | Final simulation time in code units |
| `--movie-field` | `er` | Radiation energy; also `fx`, `fy`, `fz`, or `fluxmag` |
| `--movie-view` | `slice` | Central z=0 slice; also a fixed `3d` view |

For denser sampling and a longer movie:

```bash
bash test_results/run.sh all 2 3 4 release --movies \
  --movie-snapshots 121 --movie-seconds 30
```

The simulation's physical duration is unchanged by `--movie-seconds`.
The default front only travels 0.2 code-length units. To watch one complete
periodic crossing of the length-2 box at c=1:

```bash
bash test_results/run.sh front 2 3 4 release --movies \
  --time 2 --movie-snapshots 121 --movie-seconds 30
```

The sphere is initialized from a steady reference. Its movie may show small
relaxation or numerical drift; it is not an initially dark sphere lighting up.
The `3d` option changes the view only; it does not set the physical rotation rate.

## Why hard_dt is used

In the supplied solver, output is checked once per `refinement_freq()` steps:

```text
steps_per_check = int(2 / cfl + 0.5)
odt = final_time / (requested_snapshots - 1)
hard_dt <= odt / steps_per_check
```

With `cfl=0.4`, there are five timesteps per output check. At the defaults,
`odt=0.0033333333333333335` and `hard_dt<=0.0006666666666666668`.
The solver still takes a smaller timestep whenever its stability bound requires
one. An existing tighter `hard_dt`, or one supplied with `--hard-dt`, is retained.
An explicit `--odt` overrides the interval inferred from the snapshot count.

Floating-point comparisons at output boundaries can change the exact snapshot
count by one. The renderer checks the actual count and times. If a movie-mode
run contains substantially fewer states than requested, it stops and reports
that fact; `--allow-sparse` on the standalone renderer accepts such a run.

**A smaller hard_dt also changes temporal error.** The cap is recorded in
`run.ini` and `run.json`, included in the convergence comparison settings, and
shown on convergence plots. For the paper, document this cap; movie-mode results
must not silently be treated as runs using only the usual CFL timestep.

## Output and viewing

The terminal prints the new batch directory and the movie gallery path:

```text
test_results/results/<timestamp>/movies.html
```

Open `movies.html` in your browser. Each case and level has its own
`movies/er-slice-z/movie.mp4`, saved PNG frames, a VisIt session, logs, and JSON
metadata. Keep the batch directory together when copying it to another machine.
The static comparison/error report is still at `plots/index.html`.

The camera and color limits are fixed throughout each movie. The default limits
are measured over all displayed times; they are not independently rescaled for
each frame. Different movies can have different ranges. To enforce a common
range, pass `--minimum` and `--maximum` to `movies.py`.

The renderer reads `X.*.silo` in numeric order, followed by `final.silo`.
`analytic.silo` is excluded. If the last regular output duplicates the final
time, the final state is used once. Keep every `.silo.data` directory beside
its corresponding `.silo` root file.

Timestamps come from Silo metadata, which Octo-TIGER writes in physical seconds;
field and coordinate values follow its Silo unit conversions. This differs from
the static CSV plots, which use code units.

## Make movies later, or change their speed

Capture the Silo data on the simulation machine without requiring VisIt there:

```bash
bash test_results/run.sh all 2 3 4 release --movies --movie-data-only
```

Then render the completed batch on a machine with VisIt and FFmpeg. Replace
`BATCH` below with the actual directory printed by the runner:

```bash
python3 test_results/movies.py BATCH --seconds 30
```

`python3` runs the file named immediately after it. Here, `movies.py` reads
existing results; it does not rerun the simulations. You can also give a single
case/level directory instead of the entire batch.

Once PNGs have been rendered, change playback without running VisIt again:

```bash
python3 test_results/movies.py BATCH --reuse-frames --seconds 40
```

The same recorded states are held longer. Relative physical time gaps are
preserved, with the configured extra holds at the endpoints. No intermediate
simulation state is synthesized. With 61 states spread over a 20-second movie,
the rate of new scientific images is roughly three per second even though the
MP4 is encoded at 30 fps. More stored states improve temporal detail; slower
playback alone cannot recover missing data.

For existing runs with only a few Silo outputs, the standalone renderer can
still make a long movie and reports that its motion will be coarse. For a run
originally requested in movie mode, explicitly pass `--allow-sparse` if its
actual snapshot count falls short. An incomplete simulation is not rendered.

Change the plotted quantity or slice with the standalone script:

```bash
python3 test_results/movies.py BATCH --field fluxmag --axis y --position 0
python3 test_results/movies.py BATCH --field er --minimum 0 --maximum 1
```

`--reuse-frames` requires the same input paths and rendering options as the
previous render. After moving a batch, rerender without this flag. Generated
HTML and MP4 files can be viewed after copying without rerendering.

## Validation

The tests exercise output-check timing, numeric Silo ordering, exclusion of the
reference, actual time spacing, endpoint holds, and MP4 encoding. A VisIt API
fixture also checks fixed color limits and duplicate final-time handling.
The encoding test uses FFmpeg and checks the result with ffprobe: 20 seconds,
600 frames at 30 fps, H.264/YUV420 output with even image dimensions.

```bash
python3 -m unittest discover -s test_results/tests -v
```

VisIt itself and a full HPX Octo-TIGER run were not available in the development
environment. Actual Silo rendering must therefore be checked on your machine.
Rendering failure retains the completed simulation and its data, so you can
rerun `movies.py` after resolving it.

The rendering interface follows VisIt's
[Python scripting recipes](https://visit-sphinx-github-user-manual.readthedocs.io/en/develop/python_scripting/quickrecipes.html)
and [API reference](https://visit-sphinx-github-user-manual.readthedocs.io/en/develop/python_scripting/functions.html).
