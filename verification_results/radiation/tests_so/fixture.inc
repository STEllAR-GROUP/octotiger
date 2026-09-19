#include "octotiger/radiation/m1.hpp"
#include "octotiger/radiation/grey_opacity.hpp"
#include "octotiger/radiation/subcycle.hpp"
#include "octotiger/radiation/conservation.hpp"
#include "octotiger/test_problems/radiation/profiles.hpp"
#include "octotiger/timestep.hpp"
#include <array>
#include <atomic>
#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#define PROFILE()
#define OCTOTIGER_EXPORT
using integer=long long;
#ifndef TEST_CELLS
#define TEST_CELLS 8
#endif
constexpr int INX=TEST_CELLS, NRF=4, NDIM=3, NCHILD=8, H_BW=3, RAD_BW=3;
constexpr int RAD_NX=INX+2*RAD_BW, RAD_N3=RAD_NX*RAD_NX*RAD_NX;
constexpr int H_NX=RAD_NX, H_N3=RAD_N3, HS_NX=INX/2+2*H_BW, HS_N3=HS_NX*HS_NX*HS_NX;
constexpr int XDIM=0,YDIM=1,ZDIM=2,er_i=0,fx_i=1,fy_i=2,fz_i=3;
constexpr int rho_i=0,egas_i=1,tau_i=2,sx_i=3,sy_i=4,sz_i=5,spc_i=6;
constexpr int WD=1,MARSHAK=9,RADIATION_EQUILIBRIUM_SPHERE=3;
constexpr Real ZERO=0;
using M1=RadiationM1<Real,NDIM>;
integer rindex(integer i,integer j,integer k) { return k+RAD_NX*(j+RAD_NX*i); }
integer hindex(integer i,integer j,integer k) { return rindex(i,j,k); }
struct options_fixture {
    radiation::GreyOpacity radiationOpacity;
    Real code_to_g=1, code_to_cm=1;
    Real rad_c_ratio=1,rad_cfl=.4,rad_theta=1,rad_opacity=.7,xscale=.5;
    bool rad_subcycling=true,rad_velocity_terms=true,rad_log_subcycles=false;
    bool hydro=true,rad_implicit=false,periodic=true,gravity=false,optimize_local_communication=false;
    integer rad_max_subcycles=1024;
    int problem=0,n_fields=7,n_species=1,eos=0;
    Real dual_energy_sw1=.001;
    std::string rad_energy_mode="thermal";
};
options_fixture& opts() { static options_fixture o; return o; }
struct constants_fixture { Real c=7,sigma=7./4,mh=1,kb=2./3; };
constants_fixture& physcon() { static constants_fixture p; return p; }
Real ztwd_energy(Real) { return 0; }
struct grid {
    std::vector<std::vector<Real>> U=std::vector<std::vector<Real>>(7,std::vector<Real>(H_N3));
    std::vector<std::vector<Real>> X=std::vector<std::vector<Real>>(3,std::vector<Real>(H_N3));
    Real dx=1./INX;
    static Real get_fgamma() { return 5./3; }
    static Real get_omega() { return 0; }
    Real get_dx() const { return dx; }
    auto const& get_X() const { return X; }
    auto& get_field(int f) { return U[f]; }
};
template<class T> using specie_state_t=std::array<T,5>;
void mean_ion_weight(specie_state_t<Real> const&,Real& mmw,Real& X,Real& Z) { mmw=1;X=.7;Z=.02; }
Real kappa_p(Real rho,Real,Real,Real,Real,Real) { return rho*opts().rad_opacity; }
Real kappa_R(Real rho,Real,Real,Real,Real,Real) { return rho*opts().rad_opacity; }
bool radiationFixedMediumProblem() { return opts().problem==3; }
radiationTests::Parameters radiationTestParameters() {
    radiationTests::Parameters p;p.c=physcon().c;p.length=2*opts().xscale;return p;
}
std::vector<Real> marshak_wave_analytic(Real,Real,Real,Real time) {
    std::vector<Real> u(11);u[7]=1+time;u[8]=physcon().c*(.2+.1*time);return u;
}
struct silo_var_t {};
namespace geo {
constexpr int MINUS=0, PLUS=1;
struct direction {};
struct dimension {
    int index;dimension(int v):index(v){} operator int() const { return index; }
    static std::array<dimension,3> full_set() { return {0,1,2}; }
};
struct octant {};
struct face {
    int index;face(int v):index(v){} operator int() const { return index; }
    int get_dimension() const { return index/2; }
    int get_side() const { return index%2; }
    static std::array<face,6> full_set() { return {0,1,2,3,4,5}; }
};
}
#define private public
