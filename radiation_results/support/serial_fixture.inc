
#include "octotiger/math/Debug.hpp"
#include "octotiger/radiation/m1.hpp"
#include "octotiger/radiation/conservation.hpp"
#include "octotiger/test_problems/radiation/reference.hpp"
#include "test_problems/radiation/hydro_benchmarks.hpp"
#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#define PROFILE()
#define OCTOTIGER_EXPORT
using integer=long long;
constexpr int INX=TEST_CELLS,NRF=4,NDIM=3,NCHILD=8,H_BW=3,RAD_BW=3;
using M1=RadiationM1<Real,NDIM>;
constexpr int RAD_NX=INX+2*RAD_BW,RAD_N3=RAD_NX*RAD_NX*RAD_NX,H_NX=RAD_NX,H_N3=RAD_N3;
constexpr int HS_NX=INX/2+2*H_BW,HS_N3=HS_NX*HS_NX*HS_NX;
constexpr int XDIM=0,YDIM=1,ZDIM=2,MARSHAK=9,RADIATION_EQUILIBRIUM_SPHERE=3;
constexpr int RADIATION_STREAMING_WAVE=0,RADIATION_STREAMING_FRONT=1,RADIATION_GAUSSIAN_PULSE=2;
constexpr int RADIATION_SUBCRITICAL_SHOCK=18,RADIATION_SUPERCRITICAL_SHOCK=19,RADIATION_BLAST_WAVE=20,IDEAL=0;
constexpr int rho_i=0,egas_i=1,tau_i=2,spc_i=3;
constexpr int sx_i=4,sy_i=5,sz_i=6,lx_i=7,ly_i=8,lz_i=9;
integer rindex(integer i,integer j,integer k){return k+RAD_NX*(j+RAD_NX*i);}
integer hindex(integer i,integer j,integer k){return rindex(i,j,k);}
struct options_fixture {
    Real cfl=.4,xscale=1,stop_time=.2,omega=0,radiation_reduced_c_factor=1;
    Real code_to_g=1,code_to_cm=1,code_to_s=1;
    Real radTestChi=5,radTestWidth=.2,radTestBackground=1,radTestAmplitude=.01,radTestLuminosity=.01;
    int problem=0,n_fields=15,max_level=0,eos=IDEAL;
    bool radiation=true,hydro=false,gravity=false,unigrid=true,periodic=true,rad_implicit=false;
    bool reflect_bc=false,inflow_bc=false;
    std::vector<Real> atomic_mass{1},atomic_number{1};
    std::string radReference;
};
options_fixture& opts(){static options_fixture o;return o;}
struct constants_fixture {Real c=1;};
constants_fixture& physcon(){static constants_fixture p;return p;}

std::vector<Real> marshak_wave_analytic(Real,Real,Real,Real){throw std::runtime_error("Unused Marshak fixture");}
struct silo_var_t{};
namespace geo {
constexpr int MINUS=0,PLUS=1;
struct face {
    int index;
    face(int value):index(value){}
    operator int() const {return index;}
    int get_dimension() const {return index/2;}
    int get_side() const {return index%2;}
};
struct direction{};struct dimension{};struct octant{};
}
#define private public
