# Comoving flux-limited diffusion

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
