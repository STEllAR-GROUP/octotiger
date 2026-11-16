# Bolometric light curves along the rotation axis

`light_curve=on` measures two instantaneous, bolometric blackbody photosphere
light curves, viewed from the positive and negative z axes. It works with
hydrodynamics alone, the IPR equation of state, or FLD. It does not require a
radiation energy field or change the evolution.

Each view covers `[-xscale,xscale]` in both x and y. A rectangular beam starts
at the observer-facing domain boundary and splits at the transverse boundaries
of actual AMR leaf cells. Each child beam retains its incoming optical depth
and accumulates optical depth through the next cells until

\[
\tau_{\rm LOS}=\int \chi\,ds=1,\qquad
\chi=\alpha_*\rho^pT^q+\rho\kappa_s.
\]

Density and temperature are constant within each crossed cell. The first cell
that reaches optical depth one supplies the temperature; traversal stops there.
Covered AMR parent cells are excluded. The two directions are traced
independently. There is no separate image pixel resolution: projected patches
follow the simulation mesh. The camera is along the rotation axis, so rotation
about z does not change its direction.

Optical depth starts at zero on each domain boundary. The domain must extend
far enough that omitted material outside it has negligible optical depth; this
diagnostic does not enlarge the simulation domain automatically.

The absorption law gives an inverse length directly, with no additional density
factor. Scattering is a mass opacity and is multiplied by density. These are
independent observer settings with `light_curve_opacity_model=override` (the
default). With `light_curve_opacity_model=fld`, FLD must be enabled and the
observer uses evolution's total Rosseland extinction, composition and opacity
cutoff. The manifest identifies the selected provider and its parameters.
Enter override coefficients in
the current runtime code units, including any automatic unit conversion applied
to the simulation; temperature is in kelvin. For a mass absorption
law proportional to \(\rho^mT^q\), set the density exponent to \(p=m+1\).

## Temperature and luminosity

The observer uses the gas temperature: IPR's stored temperature, FLD's
\(T=U/(\rho c_v)\), or the ideal-gas relation with the species-dependent mean
molecular weight. The ideal-gas and FLD paths use the existing dual-energy
recovery of thermal energy. No radiation temperature is substituted.
The fully ionized FLD material adapter returns Kelvin directly, subtracting
kinetic and cold WD energy and using the same thermal tau fallback as evolution.
For constant-Cv fixtures, the configured `fld_cv` must make
\(U/(\rho c_v)\) a temperature in kelvin for physical luminosities. A synthetic dimensionless FLD heat
capacity does not by itself define a physical luminosity scale.

For each resolved projected patch, the assumed intensity and reported directional
isotropic-equivalent luminosity are

\[
I_{\rm bol}=\frac{\sigma_{\rm SB}}{\pi}T^4,\qquad
L_{\rm iso}=4\sigma_{\rm SB}\sum_{\rm resolved\ patches}T^4\Delta A_\perp.
\]

Here \(\Delta A_\perp\) is the projected patch area. At distance \(D\), the
corresponding observer flux is \(L_{\rm iso}/(4\pi D^2)\). These two directional
values are not an angular integral of the object's total luminosity.

This is a blackbody photosphere approximation. A ray whose total optical depth
is below one contributes zero to this estimate and is counted as unresolved;
that does not imply zero optically thin emission. In scattering-dominated gas,
the gas temperature at total optical depth one need not be the thermalization
temperature. Spectra, light-travel delays, and velocity-dependent transfer are
outside this diagnostic.

## Options and output

| Option | Meaning |
| --- | --- |
| `light_curve` | Enable the diagnostic; default `off`. |
| `light_curve_interval` | Number of completed evolution steps between samples; default `1`. |
| `light_curve_output` | Append-only output filename; default `light_curve.dat` in `datadir`. |
| `light_curve_absorption` | Absorption coefficient \(\alpha_*\) in the law above; default `1`. |
| `light_curve_scattering` | Constant mass scattering opacity \(\kappa_s\); default `0`. |
| `light_curve_density_exponent` | Absorption exponent \(p\); default `0`. |
| `light_curve_temperature_exponent` | Absorption exponent \(q\); default `0`. |

The initial state is sampled, including static runs with `hydro=off` and
`stop_step=0`. Evolution then samples every `light_curve_interval` completed
steps, after the implicit thermal update when FLD is enabled. The file columns
are:

```text
# time_s step liso_plus_erg_s liso_minus_erg_s area_plus_cm2 area_minus_cm2 unresolved_area_plus_cm2 unresolved_area_minus_cm2 sample
```

`plus` denotes the observer at positive z looking inward, and `minus` the
observer at negative z. Times, luminosities, and areas are converted to seconds,
erg/s, and cm² using the configured physical units. `area_*` measures patches
that reach optical depth one. For each view, resolved plus unresolved area is
the full camera area. A matching `LIGHT_CURVE` line appears on stdout for each
sample. Reusing a filename appends a new run's samples; use a new file to keep
runs separate.

Each sample also saves its actual photosphere cells under
`<light_curve_output>.mesh/sample_<sample>/`. `manifest.json` records the sample's
time, step, totals, camera half-width, opacity settings, physical units, and
locality filenames. Distributed runs require this output directory to be on a
filesystem shared by all localities. Each `locality_<id>.json` contains the photosphere cells
owned by that locality. A cell entry has:

- `view`: `plus_z` or `minus_z`;
- `x`, `y`, `z`, `width`: the actual source cell's center and width, in cm;
- `temperature`: the gas temperature, in K;
- `visible_area`: its visible projected area, in cm²;
- `patches`: visible rectangles `[xmin, xmax, ymin, ymax]`, in cm.

A source cell appears once per view, with all of its visible patches grouped
together. A cell can be only partly visible, so its `visible_area` need not be
its full face area. The files retain the simulation's cell geometry; they are
not a separate uniform image mesh. Summing `visible_area` recovers the reported
photosphere area, and summing \(4\sigma_{\rm SB}T^4\,\mathrm{visible\_area}\)
recovers its luminosity. Fresh sample directories prevent a later run from
overwriting earlier saved meshes.

For example, add these options to an existing run with consistent code units:

```text
--light_curve=on --light_curve_interval=10
--light_curve_absorption=1 --light_curve_scattering=0
--light_curve_density_exponent=0 --light_curve_temperature_exponent=0
```

AMR resolution controls the transverse patches and the cells where optical
depth one is reached. Refine the simulation mesh to check convergence of this
piecewise-constant photosphere estimate.

## Validation

`ctest -L light_curve` runs the standalone column tests and bounded application
fixtures. The application fixtures check opposite-view ordering and uniform
normalization, rotation-axis invariance, hydro-only sampling within an evolution
batch, physical unit conversion, absorption
and mass scattering units, an opacity power law, thick and thin rays on a mixed
AMR hierarchy, rays crossing a refined layer before reaching a coarse
photosphere cell, FLD temperature changes and sampling cadence, and a consistent
IPR state whose stored temperature differs substantially from an ideal-gas
conversion of its total thermal energy. They also check the saved cell meshes:
visible patches lie inside their cells, do not overlap within a view, and
reproduce the reported area and luminosity. Each fixture keeps its command,
log, and parsed metrics in its test directory.
