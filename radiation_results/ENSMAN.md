# Ensman radiation hydrodynamics problems

These three problems are **Cartesian adaptations**, not exact reproductions of
Ensman's spherical Lagrangian calculations. The source is Lisa Ensman (1994),
[*Test Problems for Radiation and Radiation-Hydrodynamics Codes*, ApJ 424,
275–291](https://doi.org/10.1086/173889), §§5.1–5.3
([original PDF](https://articles.adsabs.harvard.edu/pdf/1994ApJ...424..275E)).
The setup constants and finite-volume initialization are implemented in
[`hydro_benchmarks.hpp`](../test_problems/radiation/hydro_benchmarks.hpp).

## Subcritical and supercritical shocks

The original paper uses a spherical shell from 8×10¹¹ to 8.7×10¹¹ cm, initially
at density 7.78×10⁻¹⁰ g/cm³. Its 300 equal-mass zones start with a temperature of
10 K at the outer edge, increasing by 0.25 K per zone inward. The moving inner
piston has speed 6×10⁵ cm/s for the subcritical case or 2×10⁶ cm/s for the
supercritical case. The initial absorption coefficient is 3.115×10⁻¹⁰ cm⁻¹.
The fully ionized pure-hydrogen ideal gas has γ=5/3 and mean molecular weight
μ=0.5. These are the original paper's parameters, rather than the γ=7/5, μ=1
parameters used by some later planar variants.

The local versions use:

| Quantity | Local setting |
|---|---|
| Domain | Cube spanning −3.5×10¹⁰ to +3.5×10¹⁰ cm |
| Flow | Planar x-dependent inflow toward the left wall |
| Velocity | −6×10⁵ or −2×10⁶ cm/s |
| Density | 7.78×10⁻¹⁰ g/cm³ initially |
| Temperature | Linear from 85 K at the left wall to 10 K at the right inflow |
| Boundaries | Reflecting left wall; fixed right inflow; zero-gradient transverse faces |
| Absorption | χ=κρ, κ=(3.115×10⁻¹⁰)/(7.78×10⁻¹⁰) cm²/g; no scattering |
| Gas | γ=5/3, μ=0.5 |
| Initial radiation | E=aᵣT⁴, Fₓ=(4/3)vₓE; first-order lab-frame LTE flux |
| Default final time | 38300 s subcritical; 7500 s supercritical |

Here aᵣ=4σ/c. Conserved fields are initialized using positive four-point
Gauss–Legendre quadrature in every dimension, including a volume average of T⁴.
The reflecting planar wall replaces the spherical moving piston; a stationary
Cartesian mesh and nonlinear M1 closure replace the original moving mesh and
radiative-transfer calculation. Curvature and the original equal-mass radial
spacing are not reproduced.

The original Figure 9 at 38300 s is a subcritical comparison target: gas just
upstream remains cooler than the downstream equilibrium gas, with a radiative
precursor and gas temperature overshoot. Figure 13 at 7500 s is a supercritical
target: upstream gas approaches downstream equilibrium temperature, while the
gas spike and asymmetric flux remain. The original solution includes negative
radiation flux behind the supercritical shock. These are physical diagnostics,
not exact analytic pass/fail curves for the adapted geometry.

### Approximate analytic shock scales

Ensman's equation (15) gives the gas-pressure-dominated strong-shock estimate
`T2 ≈ 2 (gamma-1) u1^2 / [R (gamma+1)^2]`, where
`R = k_B/(mu m_H) = 1.6502217176e8 erg/(g K)` and **u1 is the upstream speed
relative to the shock**, not the imposed wall-frame speed U. With negligible
upstream pressure and no net radiative energy loss, the adiabatic wall-shock
baseline has compression `(gamma+1)/(gamma-1)=4`, `u1=4U/3`, outward wall-frame
shock speed `s=U/3`, and `T2≈U^2/(3R)`.

| Imposed U | Adiabatic shock speed s | Adiabatic downstream T2 |
|---|---|---|
| 6 km/s | 2 km/s | 727.2 K |
| 20 km/s | 6.667 km/s | 8079.7 K |

These are approximate jump scales, not analytic transient profiles or acceptance
tolerances. Radiative losses can increase compression above four and change the
shock speed and cooled plateau. In particular, **8079.7 K is not a prediction for
the cooled supercritical plateau**: Ensman's Figure 15 shows a radiating plateau
near 4000 K versus about 8000 K in its adiabatic comparison.

In the approximate steady, gas-pressure-dominated supercritical diffusion
picture, `T_minus≈T2` and `T_plus≈(3-gamma)T2=4T2/3` (Ensman, p. 284).
The upstream temperature and spike are compared to the **cooled downstream
plateau T2**, rather than the adiabatic baseline. These relations do not provide
exact M1 profiles or universal pass/fail thresholds.

## Radiating blast

The original blast uses a hollow spherical shell from 10¹² to 10¹³ cm with
ρ(r)=10⁻⁶(r/10¹² cm)⁻⁷ g/cm³, initially in radiative equilibrium at luminosity
10³⁴ erg/s. It deposits 10⁴⁷ erg in the innermost Lagrangian zone, then sets inner
luminosity and velocity to zero. The electron-scattering coefficient is treated
as pure absorption. Figures 17–22 show the original numerical evolution.

The local blast has spherically symmetric initial density, temperature, radiation,
and energy-deposition profiles sampled on a Cartesian mesh. The computational
cube spans ±10¹³ cm. Its density is

```
R0 = 10^12 cm
rho(r) = 10^-6 [max(r,R0)/R0]^-7 g/cm^3
```

The central hole is filled with a constant-density core. A Gaussian gas-energy
addition of width R0 replaces deposition into one moving radial zone:

```
delta e_gas(r) = Edep exp(-r^2/R0^2) /
                 [pi^(3/2) R0^3 erf(Rout/R0)^3]
Edep = 10^47 erg, Rout = 10^13 cm
```

The exact integral over every cell is evaluated with products of error-function
differences. The box integral therefore equals Edep at every uniform resolution,
including coarse implementation checks. The deposition is initially gas energy;
subsequent emission transfers energy to radiation. The Gaussian is not a point
explosion and is not the original innermost-zone deposition.

An approximate LTE diffusion envelope is initialized from L=10³⁴ erg/s,
χ₀=κ×10⁻⁶ g/cm³, and Eedge=L/(4πcRout²). For R0≤r≤Rout,

```
E(r) = Eedge + [3 chi0 L/(32 pi c R0)] [(R0/r)^8-(R0/Rout)^8]
F(r) = L/(4 pi r^2) r_hat
```

For r<R0, the core adds
`3 chi0 L/(8 pi c R0) [1-r^2/R0^2]` to E(R0) and uses
`F = L r_vector/(4 pi R0^3)`. At radii greater than Rout, which occur near cube
corners, the diffusion E is clamped to Eedge while the spherical flux declines as r⁻².
The final initialization imposes `E=max(E_diffusion, (1+10^-12)|F|/c)` wherever
the approximate diffusion atmosphere would otherwise violate the M1 bound. This
streaming floor preserves the radial luminosity while keeping |F|<cE.
Before the gas-energy addition, Tgas=(E/aᵣ)^(1/4). The source luminosity is not
maintained during evolution. The default final time is 22000 s.

This LTE diffusion initializer is not the original atmosphere obtained by
relaxing the full radiation-transfer calculation. The filled core, Gaussian
energy addition, M1 closure, streaming floor, Cartesian outer boundary, and static grid also
change the initial-boundary-value problem. Original precursor breakout near
5800 s and luminosity maximum near 9600 s are useful context, not quantitative
acceptance times for this adaptation. A nonradiating Sedov–Taylor curve is not
an exact solution for this problem either.

For a point explosion with constant energy E0 in a cold, uniform density rho0,
the separate spherical adiabatic Sedov–Taylor limit is
`R_shock = xi(gamma) (E0 t^2/rho0)^(1/5)` and
`v_shock = 2 R_shock/(5t)`. The present finite Gaussian deposition, filled core,
truncated r^-7 atmosphere, initial radiation field, and radiative losses violate
those assumptions. Inserting the envelope exponent into a generalized Sedov
power law is not a valid reference for this problem. There is no supplied exact
analytic radiating-blast solution here; quantitative validation needs an
independent solution with matching physics and boundary conditions.

## Optional reduced speed of light

The default is `radiation_reduced_c_factor=1`, retaining physical light speed.
A constant factor `epsilon` in `(0,1]` sets `chat=epsilon*c`. The results launcher
accepts `--radiation-reduced-c-factor F` for these three coupled cases; the four
radiation-only reference tests always use factor 1. The QB4 workflow accepts a
`reduced_c_factors` mapping by coupled case, with omitted entries defaulting to 1.
Run metadata records the requested factor, physical `c`, and transport `chat`.

This uses the time-derivative convention of
[Skinner & Ostriker (2013), §§2.1 and 3.2](https://arxiv.org/abs/1306.0010).
With `Q=F/c`, and `S_E,S_Q` denoting the physical-speed matter sources, the
stationary-grid radiation equations are

```
dE/dt + chat div(Q) = epsilon S_E
dQ/dt + chat div(P) = epsilon S_Q
dE_gas/dt|source = -S_E
dm_gas/dt|source = -S_Q/c
```

Physical `c` remains in `Q=F/c`, `a_r=4 sigma/c`, the `v/c` source terms,
initialization, luminosity, the LTE relation, CGS flux units, and `|F|<=cE`.
The source update pairs changes so that **`E_gas+E/epsilon` and
`m_gas+Q/chat`** are conserved locally. These are the physical total energy
and momentum only at factor 1. The exported radiation ledger still accounts
for its actual numerical transport and source increments; a small ledger
residual does not establish conservation of physical gas-plus-radiation energy
for a reduced-speed calculation.
The LTE relation is unchanged, but a closed gas/radiation reservoir can settle
at a different temperature because its conserved weighted energy is different.

The implicit flux damping uses `chat chi_t dt`, with equilibrium flux
`Q_eq=(E I+P)v/c` and gas increment `(Q_old-Q_new)/chat`. For thermal exchange,
write `a_r T^4=alpha e^4`, `eta=chat chi_a dt`, and
`q=c chi_a dt/(1+eta)`. The same monotone quartic becomes
`e_new+q alpha e_new^4=e_star+q E_star`, and
`E_new=(E_star+eta alpha e_new^4)/(1+eta)`. Gas total energy receives
`(E_old-E_new)/epsilon`; the explicit radiation work increment is also reduced
by `epsilon`, while its gas counterpart and kinetic-energy accounting retain
their physical strength. The full equations and split update are documented in
[`radiation-paper.tex`](../doc/radiation-paper.tex).

For static diffusion, the timescale requirement is
`chat >> v_max max(1,tau_max)`, where `v_max=max(|v|+c_eff)` includes the acoustic
signal speed and `tau_max` spans the relevant flow or precursor, not one cell.
This preserves the ordering of radiation diffusion and gas evolution. It does
not guarantee accurate rapidly varying radiation transients. Keep the factor
constant within each run and compare different factors at the same resolution.

The two shocks initially have column optical depth `chi L=21.805`. As a rough
planning estimate, `v_max≈2U` and `tau≈30` give the following first **trial**
factors; neither estimate is a measured bound throughout a simulation.

| Case | First trial factor | Approximate `chat/(2U*30)` |
|---|---:|---:|
| Subcritical shock | 0.03 | 25 |
| Supercritical shock | 0.1 | 25 |
| Radiating blast | 1 (physical-speed baseline) | Static-diffusion argument does not apply |

For each shock, compare precursor extent, shock location, gas/radiation
temperatures, and signed flux against a larger factor or physical `c`, separately
from resolution convergence. The blast has radial optical depth about
`4.67e5` through the filled core and envelope. Its resolved central Gaussian
deposition gives an initial gas sound-speed scale about `1.41e8 cm/s`;
their product exceeds physical `c`. The static-diffusion condition therefore
cannot justify a reduction for this trapped, evolving radiation field.
No reduced-speed full-duration validation is claimed, and all supplied defaults
remain 1.

Short CPU implementation checks have passed at 16³ cells for factors 0.03
(subcritical shock), 0.1 (supercritical shock), and 0.1 (blast), each advancing
four steps. Their final times were 1.62149213, 0.48644764, and 138.98503967 s,
respectively. They check every initial/final checkpoint cell, verify the factor
in both checkpoints, and check matter/radiation evolution, compression for the
shocks, physical flux realizability, radiation ledger balance, and exported
slice units. The largest normalized radiation-ledger residual was 6.43×10⁻¹⁷
using the smoke check's maximum-budget normalization. **The blast factor 0.1
was only an implementation smoke check; it is not a validated physical choice.**
These short runs and the 14 passing local core CTests do not establish
full-duration accuracy or convergence in the reduced-speed factor.

## Full-duration local runs

From the repository root, preview the plan and then launch the three cases:

```sh
radiation_results/run_ensman.sh 2 --dry-run
radiation_results/run_ensman.sh 2 --output radiation_results/results/ensman-full-l2
```

The second command starts the actual long simulations. The output directory must
be new or empty. It runs **blast, supercritical shock, then subcritical shock**
sequentially at the template durations 22000, 7500, and 38300 seconds. The default
level is 2 (32³ cells for INX=8). Two HPX workers and two build jobs are used;
concurrent work still shares the 12-core local limit. The launcher rejects
larger worker/build counts. It uses the existing configured
`cmake-build-hpx2rc-radiation` application build and rebuilds its `octotiger`
target before running; `--no-build` reuses the existing executable.

The standard C++ results workflow checks output, generates PNG/PDF/CSV plots,
and updates the batch's `index.html` and problem pages after each completed
case. For the explicit command above, open
`radiation_results/results/ensman-full-l2/index.html`. Its live status updates
while running. It is a new batch; the earlier smoke-preview website and its
recorded data are not overwritten or promoted into full-duration results.
Pages for the other four problems retain their setup descriptions, but this
batch runs only the three selected coupled cases. Nothing is uploaded.

By default, 21 Silo snapshots are requested and kept, with no movie rendering
or automatic browser launch. Final numerical CSVs and still plots are always
produced. `--no-silo` omits the large snapshots. Gnuplot is required for plots;
VisIt and FFmpeg are not required by this launcher. C++ helper tools build on
first use, including a dry-run if necessary; dry-run never builds or starts the
application or creates its results batch.

Individual cases and higher resolutions use the same launcher:

```sh
radiation_results/run_ensman.sh blast 2 --output radiation_results/results/blast-full-l2
radiation_results/run_ensman.sh supercritical 2 --output radiation_results/results/supercritical-full-l2
radiation_results/run_ensman.sh subcritical 2 --output radiation_results/results/subcritical-full-l2
# Plan a later 64^3 / 128^3 study without starting it:
radiation_results/run_ensman.sh 3 4 --dry-run
```

Use `--build PATH`, `--threads N`, `--jobs N`, or `--snapshots N` to change those
settings. Do not run the individual-case commands concurrently without dividing
the shared resource budget. `--time SECONDS` is available for deliberate short
checks; it overrides every selected duration and preserves the early-time label.

### Cost of retaining the physical light speed

With zero grid rotation and the default factor 1, the transport bound is
`dt <= min(cfl,0.25) dx/(3c)`. At level 2 with INX=8 and these templates:

| Case | Requested physical duration | Light-speed step ceiling | Minimum step count from that ceiling |
|---|---:|---:|---:|
| Radiating blast | 22000 s | 1.73731 s | 12664 |
| Supercritical shock | 7500 s | 0.00608060 s | 1233432 |
| Subcritical shock | 38300 s | 0.00608060 s | 6298726 |

These are **timestep estimates, not wall-clock predictions**; tighter bounds or
output scheduling can add steps. Each extra refinement level doubles these
counts and increases the uniform-grid cell count by eight. Level 2 is a coarse
first full-duration run, not a convergence result; even these shock runs are
very expensive. The table uses the default physical light speed.

With a factor below 1, replace `c` by `chat` in this transport bound. The
transport-only step counts decrease by that factor; gas/source restrictions
and output scheduling can still require more steps. This is not a prediction
of wall-clock speedup or a justification of the chosen approximation.

### Existing-batch recovery and plot updates

```sh
radiation_results/run_ensman.sh --resume radiation_results/results/ensman-full-l2 --dry-run
radiation_results/run_ensman.sh --resume radiation_results/results/ensman-full-l2
# Rebuild plots/pages only from completed data:
radiation_results/results.sh plot radiation_results/results/ensman-full-l2
# Refresh only the HTML, reusing existing plots:
radiation_results/results.sh pages radiation_results/results/ensman-full-l2
```

Batch resume verifies saved source/executable hashes, reuses completed runs, and
starts runs whose directories do not yet exist. **It does not continue an
interrupted simulation from a Silo checkpoint.** The existing runner refuses an
incomplete run directory rather than overwriting it. A stopped case therefore
needs a new batch, or a separately verified manual application restart. Keep its
files. The radiation checkpoint regression covers the streaming wave at factors
1 and 0.1; all four fields agree with uninterrupted evolution to relative
L-infinity discrepancy at most 3.15×10⁻¹⁶ over 4096 cells. Checkpoints restore
their saved factor, with actual version-1 files restoring the legacy value 1.
Checkpoint continuation of these coupled shock/blast cases has not been
validated. Do not interpret `--resume` as a promise of that coverage.

## Running and interpreting local results

Use `radiation_results/run.sh subcritical`, `supercritical`, or `blast` with an
application build configured with radiation enabled. The runner uses the CGS
configuration without the rescaling applied to the four radiation-only cases.
`--time` changes the requested final time; runs shorter than the case default
are labeled **early-time implementation checks** in their web pages and metadata.
For example:

```sh
radiation_results/run.sh subcritical 1 --build cmake-build-hpx2rc-radiation \
  --time 0.01 --snapshots 3 --threads 2 --jobs 2 --no-build \
  --no-open --no-movies --no-silo
```

The physical light speed is retained by default. The explicit transport timestep can make
full-duration shock evolution expensive; an early-time run is not evidence
that the developed shock or blast has been validated. The blast's deposition
width must be spatially resolved before interpreting its detailed profiles.

Pages show numerical gas/radiation temperature, density, velocity, radiation
energy, signed radiation flux, and reduced-flux profiles. The application emits a final central-plane CSV independently
of analytic solution output. The reader checks the final time, mesh coverage,
positive density and gas temperature, nonnegative radiation energy and
temperature, and |F|≤cE. These checks do not substitute for conservation,
resolution studies, or independent physical reference comparisons.

There is no supplied analytic reference for any of these coupled problems.
The report does not generate analytic error norms or convergence orders for them.
Synthetic results-tool fixtures are explicitly identified; they exercise the
reporting workflow, not the radiation hydrodynamics solver. All website generation
is local. No upload, publication, commit, or push is part of this workflow.
