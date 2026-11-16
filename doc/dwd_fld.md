# Cold white dwarfs with FLD

## Units and opacity

`fld_unit_system=physical` uses the finalized `code_to_g`, `code_to_cm`, and
`code_to_s`, including normalization performed by SCF and restored from a
checkpoint. Temperature is Kelvin. The speed of light and radiation constant
are derived from these units; `fld_c` and `fld_ar` remain parameters for explicit
code-unit fixtures. Physical opacity outputs are inverse code length.

Three providers are available:

| `fld_opacity_model` | Material exchange | Radiation transport |
| --- | --- | --- |
| `legacy` | `fld_absorption * rho^p * T^q` | Absorption plus `rho * fld_scattering` |
| `independent` | Same Planck absorption power law | Independent **total** `fld_rosseland * rho^p_R * T^q_R` |
| `free_free_thomson` | Planck mean of classical electron-ion free-free absorption | Rosseland harmonic mean of the frequency-dependent free-free absorption **plus Thomson scattering** |

For independent means, the exponents are `fld_density_exponent`,
`fld_temperature_exponent`, `fld_rosseland_density_exponent`, and
`fld_rosseland_temperature_exponent`. The Rosseland input is already total
extinction; scattering is not added to it a second time. Thomson scattering
produces no thermal source. `fld_gaunt=0` selects pure Thomson transport in the
physical provider; otherwise the Gaunt factor is a positive constant.

The thermal source is `c * alpha_P * (a_r*T^4 - E0)`. Using the Planck
absorption mean for absorption of `E0` is the gray approximation
`alpha_E = alpha_P`; an arbitrary radiation spectrum need not have that mean.

The physical provider evaluates the harmonic mean of the total monochromatic
coefficient through a growing interpolation cache backed by quadrature. It does
not add separately averaged free-free and scattering coefficients. Its free-free
normalization is

\[
\alpha_\nu=3.692\times10^8\,g_{\rm ff}\,n_e
\left(\sum_i Z_i^2n_i\right)T^{-1/2}\nu^{-3}
\left(1-e^{-h\nu/k_BT}\right)\ {\rm cm}^{-1}.
\]

This is a controlled fully ionized, classical opacity model, not an opacity
table for arbitrary WD material. Bound-bound, bound-free, degeneracy and
Compton corrections are absent.

`fld_opacity_temperature_floor` is a positive Kelvin cutoff required by the
physical provider. It regularizes **only opacity evaluation** at
`max(T,T_cut)`; the source remains `a_r*T^4` at the actual material temperature.
It never supplies gas heat or initial radiation. `opacity_regularized_cells`
counts leaf interior cells affected by the cutoff. A cutoff sensitivity study
is required before interpreting a production result dominated by such cells.

## Frequency quadrature and nonlinear iteration

With \(x=h\nu/(k_BT)\), the physical provider evaluates

\[
\chi_R^{-1}=\frac{15}{4\pi^4}\int_0^\infty
\frac{x^4e^{-x}}{(1-e^{-x})^2}
\frac{dx}{A_{\rm ff}(1-e^{-x})/x^3+\sigma_T n_e}.
\]

The Planck mean is analytic, \(\alpha_P=15A_{\rm ff}/\pi^4\).
Composite Gauss-Legendre integration uses intervals [0,.125], [.125,.5], [.5,2],
[2,8], [8,24], [24,80]. Estimates with 16 and 32 points per interval are compared
at a relative tolerance of 2e-11. Additional 64- and 128-point estimates are
used only if needed; failure to converge raises an error. Spectral factors are
precomputed. The initial comparison costs 288 point evaluations per newly
sampled value; the maximum is 1440.

The universal function `log(chi_R/chi_s)` is cached against
`log(A_ff/chi_s)`, using four-point cubic interpolation on a natural-log grid
with spacing `1/128`. This adapts CosmicTiger's Einstein--Boltzmann background
interpolation to a table that grows on demand. A query outside the available
range samples the missing range and builds new polynomial coefficients,
extending to an aligned one-e-fold boundary. Existing coefficients stay fixed;
an in-range query performs no quadrature. Immutable snapshots permit concurrent
reads and extensions without holding a lock during integration. The cache is
recreated as needed after restart; it is not part of the physical state.

The reusable `octotiger::InterpolatedCache` in
`octotiger/interpolated_cache.hpp` binds a sampling function at
construction and accepts a grid spacing and extension block size. It supports
signed finite coordinates and values. Callers supply any log transformations
and choose spacing appropriate to their function; the interpolator does not
estimate approximation error. `LogRosselandCache` is the thin physical wrapper.
Tests also cover an unrelated positive function in log-log coordinates and a
signed nonmonotone cubic.

Pure Thomson and pure free-free limits are analytic. Ratio calculations and
reconstruction avoid forming potentially overflowing amplitude ratios. The
measured maximum interpolation error over the numerical test's dense log-range
scan is `2.25e-12`, below the `1e-10` test limit. Independent quadrature,
extension order, interval boundaries and concurrent extensions are checked.
There is no arbitrary opacity coefficient cap.
