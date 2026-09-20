# Ensman reference follow-up

## Status

The reference search recovered published **shock reproductions**, not the
original Ensman numerical data. Both Ensman descriptors remain conditional and
unimplemented. Their top-level execution parameters stay null: the nested
`published_reproductions` entries are source evidence, not runnable input files.
No numerical pass, reference curve, tolerance, new boundary condition, or new
frequency group is supplied by this documentation change.

The original citation is Lisa Ensman, *Test Problems for Radiation and
Radiation-Hydrodynamics Codes*, ApJ 424, 275-291 (1994),
[DOI 10.1086/173889](https://doi.org/10.1086/173889).
The ADS article/full-text endpoints and DOI did not yield readable full text
during this follow-up. Bibliographic metadata alone cannot establish a problem
definition.

## Verified shock reproduction definitions

[Kolb et al. (2013), section 4.5](https://arxiv.org/pdf/1309.5231)
defines a quasi-1D PLUTO FLD reproduction with a 7e10 cm longitudinal extent,
3.418e7 cm transverse extents, and 2048 x 4 x 4 cells. Initially,
rho=7.78e-10 g/cm^3 and Tgas=Trad=10 K; E=a_R T^4, gamma=7/5, mu=1, and both
reported extinction products are 3.1e-10 cm^-1. The lower longitudinal boundary
reflects, the upper is zero-gradient, and transverse boundaries are periodic.
Their Minerbo-FLD equations omit radiation advection and pressure work. Thus
Figure 7 and Table 1 are comparative results, not exact M1 references.

| Variant | Signed inflow u (cm/s) | Comparison time (s) |
|---|---:|---:|
| Subcritical | -6e5 | 3.8e4 |
| Supercritical | -2e6 | 7.5e3 |

These pairings are independently reported by
[Melon Fuksman et al. (2021), section IV.1](https://arxiv.org/html/2005.01785v2),
which uses grey M1 and physical light speed, with the same uniform material
state and no scattering. Its domain is [0,7e10] cm. Compare at s=x-u*t, not by
shifting a numerical shock to a fitted position. The paper's F_r is physical
flux divided by c (equation 3), so its reduced flux equals F/(cE).

There are unresolved details in that M1 reference: section IV.1 specifies the
reflecting left boundary but not the right boundary; prose states 2048 zones
while figure captions state 2400. Equation (8) retains pressure-tensor and
selected beta^2 source terms absent from this repository's isotropic source
approximation. Its gas EOS uses the atomic mass unit; its displayed definition
of a_R also requires a normalization cross-check against actual reference
inputs. These differences must not be hidden by rescaling a plotted curve.

[Hayes & Norman (2003), section 6.3](https://arxiv.org/pdf/astro-ph/0207260)
explains another important distinction: Ensman used a thin spherical shell at
large radius to approximate a plane. Their reproduction instead uses a finite
cylinder and a temperature gradient (10 K at the outer boundary, increasing
0.25 K per interior zone). Neither that geometry nor that grid-dependent
temperature prescription may be silently combined with the uniform Cartesian
setups above. They also show that closure and finite transverse extent change
the supercritical precursor.

## Contract needed before an executable shock test

The following is implementation analysis, not additional published data:

1. Select one identified reproduction and keep its geometry, IC, boundaries,
   velocity/time pair, EOS and source convention together. Do not label a hybrid
   setup as an original Ensman run. Resolve missing details from a versioned
   reference input or its author before execution.
2. Resolve opacity evolution, not just an initial number. The cited product
   chi=kappa*rho has units cm^-1. A constant volume extinction is representable
   with `radiation.opacity.units=1/cm`; constant mass opacity uses `cm2/g` and
   changes chi with density. Dividing chi by the initial density and then holding
   that mass opacity fixed is not evidence of equivalence after compression.
   Equal Planck/transport means with no scattering fit the current grey model
   once this prescription is verified.
3. Record c=2.99792458e10 cm/s and c_hat/c=1 for the physical-c comparison.
   Preserve `a_R=4*sigma_SB/c`, E and gas energy in erg/cm^3, physical F in
   erg/(cm^2 s), and Q=F/c in erg/cm^3. The repository uses its own recorded
   `physcon().kb`, `physcon().mh` and `physcon().sigma`; compare those constants
   with the chosen reference, including whether mu multiplies atomic or
   hydrogen mass. Do not force reference temperatures through an unexplained
   normalization adjustment.
4. Specify the initial lab-frame flux. Comoving isotropy in moving LTE implies
   Q approximately (4/3)E*v/c at first order; merely writing Tgas=Trad does not
   distinguish this from lab-frame Q=0. A source implementation must state which
   convention its published initialization actually uses.
5. Exercise the production coupled hydro solver, EOS, thermal exchange,
   momentum feedback and boundary machinery. The serial radiation fixture with
   a prescribed material state cannot validate a shock or its precursor. A
   working supported full application build/run is still required.
6. Obtain versioned reference arrays with units, frame, closure, equations,
   resolution and extraction provenance, or an independently converged solver
   for the same equations. Digitized published curves, if used, need stated
   digitization uncertainty and are comparisons rather than exact solutions.
   Set acceptance criteria before examining the new results. Include budgets
   for actual boundary energy/momentum fluxes; a reflecting wall exerts force,
   so domain gas-plus-radiation momentum is not generally conserved by itself.

Existing source-cell conservation and transport checks remain valid within
their documented scope. They do not remove these coupled-hydro gates.

## Equilibrium-sphere identity remains unresolved

No accessible primary definition established what the descriptor named
`radiation.ensman.equilibrium_sphere` should execute. This is not a claim that
Ensman contains no equilibrium problem. Radius, density, temperature, opacity,
luminosity, time, and boundaries therefore remain unknown rather than borrowed
from another test.

A separately located
[Chatzopoulos & Weide (2019), section IV.5](https://arxiv.org/html/1712.10091v2)
uses sigmoid density/temperature profiles and an initially nonequilibrium
radiation field to study a radiating sphere. That section does not identify its
setup as the requested Ensman equilibrium sphere; it is not a substitute.
Likewise, the repository's Gaussian luminosity bulb has only its stated
diffusion-limit reference and must retain its existing identity.

Resolving this remaining gate requires the relevant original section/figure
or an explicitly agreed, fully specified replacement benchmark. The metadata
keeps the distinction visible and cannot promote either case to passed.
