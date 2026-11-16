# Comoving flux-limited diffusion

The optional `fld` model evolves one comoving radiation energy density per cell.
It is separate from the existing M1 model selected by `radiation`. The fluid and
radiation use the same octree topology and hydro timestep. Radiation diffusion
and material heating are integrated implicitly after the explicit hydro stage.

## Equations and units

Write the stored radiation energy density as \(E_0\), gas internal energy density
as \(U=C_VT\), and radiation equilibrium energy as \(B=a_rT^4\).
The material can be ideal gas with constant specific heat, or the cold WD EOS
plus fully ionized ideal thermal gas. Only thermal energy exchanges with
radiation. See [the WD model and setup](dwd_fld.md) for the energy split,
composition, physical units, floors and new opacity providers.

The default `legacy` opacity provider retains the original power law below.
Its input scattering opacity \(\kappa_s\) is **per unit mass** (\(\mathrm{cm^2\,g^{-1}}\)
in cgs units). The inverse-length coefficients used in transport are:

\[
\alpha=\alpha_*\rho^pT^q,\qquad \sigma=\rho\kappa_s,\qquad
\chi=\alpha+\sigma.
\]

Here \(\kappa_s\) is constant; scattering extinction varies with density.
The absorption power law defines \(\alpha\) per unit length, with no additional
density factor. If a mass absorption law is \(\kappa_a=\kappa_{a*}\rho^mT^q\),
set \(p=m+1\) to obtain \(\alpha=\rho\kappa_a\). Elastic scattering
contributes to transport and radiation force, but not to thermal exchange. The
constants `fld_c`, `fld_ar`, and `fld_cv` apply to code-unit fixtures. Physical
mode derives light speed and the radiation constant from the finalized unit
mapping, and the fully ionized material model computes composition-dependent
heat capacity. In the equations below, `alpha` always means Planck absorption
and `chi` always means total Rosseland extinction; their equality apart from
scattering is specific to the legacy provider.
The implicit solve requires positive finite total extinction in every cell.

The diffusion-ordered nonrelativistic equations are

\[
\begin{aligned}
\partial_t\rho+\nabla\cdot(\rho\boldsymbol v)&=0,\\
\rho\frac{D\boldsymbol v}{Dt}&=-\nabla p-\lambda\nabla E_0,\\
\partial_tU+\nabla\cdot(U\boldsymbol v)+p\nabla\cdot\boldsymbol v
  &=c\alpha(E_0-B),\\
\partial_tE_0+\nabla\cdot(E_0\boldsymbol v)
  +\mathbf P_0:\nabla\boldsymbol v
  &=\nabla\cdot(D\nabla E_0)+c\alpha(B-E_0).
\end{aligned}
\]

The gas total-energy equation also includes radiation-force work
\(-\lambda\boldsymbol v\cdot\nabla E_0\). This work is distinct from gas thermal
exchange and from the full tensor radiation pressure work.

The closure is

\[
R=\frac{|\nabla E_0|}{\chi E_0},\qquad
\lambda=\frac{2+R}{6+3R+R^2},\qquad D=\frac{c\lambda}{\chi},\qquad
f=\lambda+\lambda^2R^2,
\]
\[
\mathbf P_0=E_0\left[\frac{1-f}{2}\mathbf I+
              \frac{3f-1}{2}\boldsymbol n\boldsymbol n\right],
\qquad \boldsymbol n=\frac{\nabla E_0}{|\nabla E_0|}.
\]

At zero gradient the pressure is isotropic, \(\mathbf P_0=E_0\mathbf I/3\).
The closure implementation evaluates the streaming limit without forming an
overflowing \(R^2\). In an optically thick region the radiation equation becomes
\(DE_0/Dt+(4/3)E_0\nabla\cdot\boldsymbol v=\cdots\). Outside that limit,
replacing tensor pressure work by a scalar divergence loses shear work.

This is a comoving FLD model under diffusion ordering, not a complete
arbitrary-optical-depth relativistic moment system. Gas plus comoving radiation
energy is not generally the conserved lab-frame total energy in a moving flow.
Absorption/emission exchange and closed-boundary diffusion do conserve their
discrete combined energy in the static tests.

## Storage and explicit integration

`grid` and its peer `fld_grid` share one
`std::shared_ptr<std::vector<real>>` containing \(E_0\) in hydro cell ordering,
including ghost cells. `grid` owns explicit radiation transport and pressure
work. `fld_grid` owns implicit-stage scratch fields and the reaction-diffusion
multigrid block. It does not own a `grid` or duplicate the accepted radiation
state. The Silo variable for this model is `Erad`.

The explicit radiation advection uses the existing hydro reconstruction and HLL
flux implementation. It adds the radiation acoustic contribution to the same
wave-speed estimate:

\[
\Delta c_n^2=\lambda\frac{E_0+P_{0,nn}}{\rho}.
\]

In isotropic diffusion this is \(4E_0/(9\rho)\). The radiation scalar advects
with \(E_0v_n\), rather than an enthalpy flux; its pressure work is evaluated
separately. For each interior cell, the velocity derivative is computed from
its six axial neighbors:

\[
(\nabla\boldsymbol v)_{ij}=
\frac{v_i(\boldsymbol x+\Delta x\boldsymbol e_j)
      -v_i(\boldsymbol x-\Delta x\boldsymbol e_j)}{2\Delta x}.
\]

The contraction uses the cell pressure tensor and all nine derivatives. This
retains off-diagonal work in a trace-free shear. A rigid rotation has an
antisymmetric velocity gradient and therefore does no work against a symmetric
radiation pressure tensor.

Constant rotation about the z axis uses the existing rotating-mesh transport:
the advective velocity is \(\boldsymbol v-\boldsymbol\Omega\times\boldsymbol x\),
while stored momentum and gas radiation-force work use inertial velocity. The
central pressure-work gradient uses the inertial cell velocities.

## Implicit thermal and diffusion update

Density and velocity are held fixed during a backward-Euler radiation stage.
Let \((E^*,U^*)\) denote the state after the explicit update. The nonlinear
residuals are

\[
R_E=E-E^*-\Delta t\,L_D E-\Delta t\,c\alpha(B-E),\qquad
R_T=U(T)-U^*+\Delta t\,c\alpha(B-E).
\]

The finite-volume diffusion operator uses shared face fluxes,

\[
(L_D E)_i=\frac1{V_i}\sum_{f=i|j}
 \frac{A_fD_f}{d_{ij}}(E_j-E_i).
\]

Face diffusivities are formed from adjacent cells. Physical radiation boundaries
use the `fld_boundary` condition described below. At coarse-fine boundaries the
coarse flux is replaced by the area-weighted fine flux before residuals are
evaluated. Covered parent
cells are multigrid storage and do not contribute to physical energy totals.

At each nonlinear iteration, opacity and limiter coefficients are lagged while
the thermal emission \(a_rT^4\) is linearized. Define

\[
C=\rho c_v,\quad A=c\alpha,\quad b=4Aa_rT^3,\quad H=C+\Delta t\,b.
\]

Eliminating the cell-local temperature increment leaves the scalar solve

\[
\left[I-\Delta t L_D+\frac{\Delta t AC}{H}\right]\delta E
=-R_E-\frac{\Delta t b}{H}R_T,
\qquad
\delta T=\frac{-R_T+\Delta t A\delta E}{H}.
\]

This is a matrix-free reaction-diffusion operator with a positive diagonal
reaction term. The Scorpio-derived distributed multigrid machinery restricts
residuals and coefficients, reconciles interface fluxes, and transfers signed
coarse corrections. The nonlinear iteration refreshes opacity and limiter and
checks an invertibly row-scaled pair of the coupled residuals. This scaling
avoids multiplying equilibrium roundoff by a very large absorption rate; the
sum of the scaled equations is the unscaled total-energy residual. Positivity
damping and a residual line
search keep radiation energy and gas temperature admissible. Reaching an
iteration or cycle limit is a reported failure, not an accepted timestep.

Opacity derivatives are deliberately excluded from the inner linear operator.
Including them would replace \(b\) by
\(A[(q+4)B-qE]/T\), which can be negative for general exponents. Refreshing the
coefficients in the outer iteration still solves the state-dependent-opacity
equations when the nonlinear residual converges.

The accepted material update uses the opposite radiation exchange and the same
reconciled diffusion flux. A rejected implicit stage leaves the shared accepted
radiation vector unchanged.

The fluid's dual-energy variable is \(\tau=U^{1/\gamma}\), so it is
\(U^{3/5}\) when \(\gamma=5/3\). Temperature recovery uses the same
`dual_energy_sw1` condition as hydro pressure: when the internal energy obtained
by subtracting kinetic from total energy is too small, it recovers
\(U=\tau^\gamma\). The accepted thermal increment \(\Delta U\) is added to
the evolved gas total energy, while \(\tau\) is reset to
\((U+\Delta U)^{1/\gamma}\). Mechanical radiation-force work is not added to
this thermal source.

In differential form the thermal contribution to the advected tau equation is
\(Q_\tau=U^{1/\gamma-1}c\alpha(E_0-B)/\gamma\), or
\((3/5)U^{-2/5}c\alpha(E_0-B)\) for \(\gamma=5/3\). The implemented
backward-Euler step applies the solved energy increment and then recomputes
tau, rather than explicitly stepping this differential source.

## Outer radiation boundary

`fld_boundary=reflecting` is the default and sets the normal diffusive flux to
zero on every exterior face. `fld_boundary=vacuum` instead specifies zero
incident radiation through a modified Marshak (Sánchez–Pomraning) condition.
It controls implicit radiation diffusion; the hydro boundary options continue
to control fluid transport and the explicit radiation advection.

For outward normal \(\boldsymbol n\), the continuum vacuum condition is

\[
F_{\rm out}=-D\,\partial_n E_0=\beta(R)cE_b,\qquad
\beta(R)=\frac{\log\cosh R}{R\tanh R},\qquad
\beta(0)=\frac12,\quad\beta(\infty)=1.
\]

Here \(E_b\) is radiation energy density at the exterior face. The boundary
coefficient corresponds to twice Castro's `FLDalpha`; Castro documents this
as boundary condition 105. See its [coefficient implementation](https://github.com/AMReX-Astro/Castro/blob/main/Source/radiation/fluxlimiter.H)
and [radiation boundary documentation](https://amrex-astro.github.io/Castro/docs/radiation.html#boundary-conditions).

Octo-Tiger uses a positive finite-volume Robin flux across the half-cell
distance \(h=\Delta x/2\), with adjacent interior energy \(E_c\):

\[
E_b=\frac{E_c}{1+\beta c h/D},\qquad
F_{\rm out}=H E_c,\qquad
H=\frac{c}{1/\beta+\chi_R h/\lambda},\qquad
D=\frac{c\lambda}{\chi_R}.
\]

This combines diffusion across the interior half cell with the vacuum face
condition. It preserves \(E_b\ge0\) and \(0\le F_{\rm out}\le cE_c\) for
nonnegative interior energy, without constructing negative ghost energy.
It is a half-cell finite-volume discretization of the condition, not a copy of
Castro's boundary stencil. Both \(R=|\nabla E_0|/(\chi_R E_0)\) and the rational
Levermore–Pomraning limiter use the adjacent interior cell's gradient estimate:
an inward one-sided derivative in each direction touching a physical boundary,
and a centered derivative otherwise. This includes edges and corners; clamped
ghosts do not halve the normal derivative used in the boundary closure.
Boundary coefficients are
lagged during each linear solve and refreshed in the outer nonlinear iteration.

Each accepted `FLD_STEP` reports `escaped_diffusion_energy`, the timestep
integral of the outward flux over physical leaf faces, and
`boundary_diffusion_luminosity`, that energy divided by the implicit timestep.
The six fields `escape_x_minus`, `escape_x_plus`, `escape_y_minus`,
`escape_y_plus`, `escape_z_minus`, and `escape_z_plus` give the per-face energy
contributions. These are per-step code energies and code power, respectively;
they are not cumulative totals. `boundary_vacuum=0/1` records the selected
condition. Reflecting boundaries have zero diffusive escape.

For a static implicit stage, the final gas-plus-radiation energy plus the
escaped diffusive energy balances the initial gas-plus-radiation energy to the
solver tolerance. This escape diagnostic excludes explicit advective flux and
does not represent a directional photosphere luminosity or a spectrum.

## Timestep and options

Implicit diffusion and thermal exchange remove their explicit stability limits.
The backward-Euler source/diffusion step and its sequential split from hydro
are first order in time.
The explicit fluid/radiation stage still requires its radiation-modified HLL CFL
and a bound on explicit expansion work. A large stable timestep may fail to
resolve the physical radiation transient. For an implicit-only run
(`hydro=off`), specify a finite positive `hard_dt`.

The FLD options are:

| Option | Meaning |
| --- | --- |
| `fld` | Enable the comoving FLD model; use `radiation=off`. |
| `fld_boundary` | `reflecting` (default, zero diffusive flux) or `vacuum` (zero incident radiation on all exterior faces). |
| `fld_absorption` | Coefficient \(\alpha_*\). |
| `fld_density_exponent` | Exponent \(p\). |
| `fld_temperature_exponent` | Exponent \(q\). |
| `fld_scattering` | Constant scattering opacity per unit mass; local extinction is `rho * fld_scattering`. |
| `fld_cv`, `fld_c`, `fld_ar` | Heat capacity, light speed, radiation constant. |
| `fld_initial_energy` | Uniform initial \(E_0\). |
| `fld_initial_pulse` | Amplitude of an added centered Gaussian of width `0.2*xscale`. |
| `fld_tolerance` | Nonlinear residual tolerance. |
| `fld_max_iterations` | Nonlinear iteration ceiling. |
| `fld_max_cycles` | Multigrid cycle ceiling per linear solve. |

Each accepted implicit step reports `FLD_STEP`, including residual, iteration
counts, leaf/cell counts, radiation and thermal volume integrals before and after
the implicit step, minimum admissible states, and `shared=1` when every peer has
the same radiation buffer as its fluid grid.

## Verification

The dependency-free numerical tests can be built separately:

```sh
cmake -S test_problems/fld -B /tmp/octotiger_fld_numerics -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/octotiger_fld_numerics --parallel 1
ctest --test-dir /tmp/octotiger_fld_numerics --parallel 1 --output-on-failure
```

These tests cover opacity units and limiting cases, density-dependent scattering
optical depths on uniform and coarse/fine faces, flux bounds, pressure trace
and eigenvalues, an analytic variable-coefficient reaction-diffusion operator,
a hundredfold diffusion-coefficient jump with continuous flux, closed-boundary
conservation, and coarse-fine flux-defect signs. Explicit tests use affine
velocity fields, anisotropic shear, rigid rotation, directional signal speeds,
and the same HLL helper used by the production solver. The actual `fld_grid`
thermal wrapper is checked against independent scalar backward-Euler roots,
including coupling strengths of \(10^{12}\), positive and negative temperature
exponents, per-cell energy conservation, and shared-state isolation until commit.

Application checks in `test_problems/fld/check_fld.py` run a uniform stiff
thermal exchange against an independent backward-Euler scalar root, a
scattering-only Gaussian diffusion problem, a mixed-resolution AMR case,
uniform evolution on a rotating mesh, Silo restart versus uninterrupted
evolution, refinement/coarsening of a loaded radiation state, and an intentional nonlinear
nonconvergence case. A forced dual-energy-fallback application check verifies
repeated temperature recovery from tau and thermal updates against the same
backward-Euler reference; the numerical helper test separately exercises true
high-Mach cancellation. They inspect
accepted diagnostics and shared-buffer ownership rather than relying only on
process exit status.

`check_boundary.py` checks vacuum escape on uniform and mixed-AMR meshes,
zero-radiation preservation, stiff absorption/emission, physical free-free plus
Thomson WD material, and checkpoint restoration against uninterrupted evolution.
It verifies the per-face flux sum and the gas/radiation/escape energy ledger.
The numerical boundary fixture independently integrates the angular closure,
and the multigrid fixture checks analytic one-cell decay, flux telescoping, FAS
residual consistency, and area restriction of wall conductance.

The seven boundary CTests passed on the CPU Release build on November 16, 2026.
The vacuum AMR fixture used 288 leaves and 147,456 cells for three timesteps;
its maximum nonlinear residual was `9.17e-10`. Across uniform, mixed-AMR,
zero-radiation, stiff-exchange and physical WD-opacity fixtures, the largest
relative gas/radiation/escape energy imbalance was `2.63e-15`. The WD fixture
uses a small physical box to make escape measurable, not a stellar equilibrium.
A preserved checkpoint without boundary metadata restored reflecting behavior,
reported exactly zero escape and reproduced 14 saved diagnostic fields exactly.

The rotating uniform-state fixture enables `inflow_bc`, because a uniform
inertial fluid crosses the rotating box in both directions. The default outflow
condition clips incoming mesh-relative velocities and does not preserve that
reference state. The restart fixture supplies the saved Silo shard directory in
each run's working directory, as required by the loader's relative shard paths.

After correcting scattering to a constant opacity per unit mass, all ten FLD
CTests passed using the CPU path. The mixed-resolution test used 288 leaves
and 147,456 physical cells for three timesteps of 0.01.
Each step took 32–33 nonlinear iterations and 53–55 multigrid cycles; final scaled
residuals ranged from \(7.17\times10^{-10}\) to \(8.01\times10^{-10}\).
The radiation volume integral changed by \(5.55\times10^{-16}\) over the run,
while its peak decreased from 0.47557 after the first step to 0.34689 after the
third. Restart checks preserved the solution when changing from eight leaves
to either 64 or one. An earlier unit-density two-locality HPX/TCP pulse run
matched the single-locality diagnostics to their printed precision. These results validate
the CPU path; they do not establish GPU execution or scaling performance.

The equation conventions follow the standard comoving diffusion equation in
[Krumholz et al. (2007), Eq. 45](https://www.mso.anu.edu.au/~krumholz/publications/2007/krumholz07d.pdf).

## WD material and opacity extensions

`fld_opacity_model` selects `legacy`, `independent`, or `free_free_thomson`.
`fld_material_model` selects `auto`, `constant_cv`, or `fully_ionized`; auto uses
fully ionized thermal capacity for WD. `fld_unit_system` selects `auto`, `code`,
or `physical`; auto selects physical units for WD, fully ionized material, or
physical opacity. Fully ionized temperature is Kelvin. The opacity cutoff is
separate from thermal energy: it never changes emission temperature or injects
energy. Zero thermal and radiation energy are allowed.

AMR scratch transfers restrict capacity and composition number-density moments
conservatively; coarse temperature is restricted thermal energy divided by
restricted capacity. Accepted hydro species and thermal tau retain the existing
hydro AMR transfers. Restart model version 1 stores all physics choices and
coefficients. Older FLD checkpoints explicitly restore legacy opacity, constant
Cv and code units. The optional `fld_boundary` metadata preserves the selected
diffusion boundary without changing the model version. Checkpoints lacking
that field restore `reflecting`, including when the new command line requests
`vacuum`; the checkpoint takes precedence. Diagnostic opacity selection remains
a runtime setting.

See [dwd_fld.md](dwd_fld.md) for equations, options, limitations and bounded tests.
