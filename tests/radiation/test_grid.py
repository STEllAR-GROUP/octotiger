#!/usr/bin/env python3
"""Compile the actual rad_grid compute methods with small application fixtures.

This exercises the production loop bodies and header layout without HPX, Silo,
or the full hydro solver. It does not test distributed HPX actions. Both equal
and unequal ghost widths are checked to catch hydro/radiation index confusion.
Run: python3 tests/radiation/test_grid.py [--sanitize]
"""
import os, pathlib, subprocess, tempfile, sys
root = pathlib.Path(__file__).resolve().parents[2]
cpp = (root/'src/radiation/rad_grid.cpp').read_text()
header = (root/'octotiger/radiation/rad_grid.hpp').read_text()
header = '\n'.join(l for l in header.splitlines() if not l.startswith('#include'))
opacities = (root/'octotiger/radiation/opacities.hpp').read_text()
opacities = '\n'.join(l for l in opacities.splitlines() if not l.startswith('#include'))

def method(start):
    begin=cpp.index(start); brace=cpp.index('{',begin); level=1; end=brace+1
    while level:
        if cpp[end]=='{': level+=1
        elif cpp[end]=='}': level-=1
        end+=1
    return cpp[begin:end]

bodies='\n'.join(method(x) for x in [
    'Real radiation_gas_internal(', 'void rad_grid::allocate()',
    'void rad_grid::set_dx(', 'void rad_grid::set_X(',
    'void rad_grid::compute_mmw(', 'Real rad_grid::max_timestep(',
    'void rad_grid::sanity_check()', 'void rad_grid::compute_flux(',
    'void rad_grid::advance(', 'void rad_grid::rad_imp(',
    'void rad_grid::complete_rad_amr_boundary()',
    'std::vector<Real> rad_grid::get_subset(',
    'rad_grid::rad_grid(Real _dx)', 'rad_grid::rad_grid()'])

fixture=r'''
#include "octotiger/radiation/m1.hpp"
#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <random>
#define PROFILE()
using integer = long long;
constexpr integer NRF=4, NDIM=3, INX=8, NCHILD=8;
constexpr integer RAD_BW=RAD_WIDTH, H_BW=HYDRO_WIDTH;
constexpr integer RAD_NX=INX+2*RAD_BW, RAD_N3=RAD_NX*RAD_NX*RAD_NX;
constexpr integer H_NX=INX+2*H_BW, H_N3=H_NX*H_NX*H_NX;
constexpr integer HS_NX=INX/2+2*H_BW, HS_N3=HS_NX*HS_NX*HS_NX;
constexpr integer HS_DNX=HS_NX*HS_NX, HS_DNY=HS_NX, HS_DNZ=1;
constexpr int XDIM=0,YDIM=1,ZDIM=2,WD=1,MARSHAK=2,spc_i=0;
constexpr int RADIATION_TEST=3,RADIATION_DIFFUSION=4,RADIATION_COUPLING=5;
constexpr Real MARSHAK_OPAC=100;
integer rindex(integer i,integer j,integer k) {return k+RAD_NX*(j+RAD_NX*i);}
integer hindex(integer i,integer j,integer k) {return k+H_NX*(j+H_NX*i);}
integer hSindex(integer i,integer j,integer k) {return k+HS_NX*(j+HS_NX*i);}
struct options_fixture {Real cfl=.4,dual_energy_sw1=.001;int eos=0,problem=RADIATION_COUPLING,n_species=1;};
options_fixture& opts() {static options_fixture o;return o;}
struct constants_fixture {Real c=17,sigma=17./4,kb=1,mh=1;};
constants_fixture& physcon() {static constants_fixture p;return p;}
struct grid {static Real get_fgamma() {return 5./3;}};
Real ztwd_energy(Real rho) {return .1*rho;}
template<class T> using specie_state_t=std::array<T,1>;
void mean_ion_weight(const specie_state_t<Real>&,Real& mmw,Real& X,Real& Z) {mmw=1;X=.7;Z=.02;}
struct silo_var_t {};
namespace geo {struct face{};struct direction{};struct dimension{};struct octant{};}
#define private public
'''
checks=r'''
#undef private
void require(bool x,const char* what) {if(!x)throw std::runtime_error(what);}
void near(Real x,Real y,const char* what) {require(std::abs(x-y)<3e-11*std::max({1.,std::abs(x),std::abs(y)}),what);}
void fill_bounds(rad_grid& g) {
    for(integer i=0;i<RAD_NX;++i)for(integer j=0;j<RAD_NX;++j)for(integer k=0;k<RAD_NX;++k) {
        const auto wrap=[](integer x){return RAD_BW+(x-RAD_BW+INX)%INX;};
        for(int f=0;f<4;++f)g.U[f][rindex(i,j,k)]=g.U[f][rindex(wrap(i),wrap(j),wrap(k))];
    }
}
int main() {
    try {
        const Real dx=.1,c=physcon().c;
        near(temperature(2.,6.,1.,1.4),1.2,"opacity temperature uses gamma");
        near(dB_p_de(2.,6.,1.,1.4),4*B_p(2.,6.,1.,1.4)/6,"emission derivative");
        opts().problem=MARSHAK;
        near(dB_p_de(2.,6.,1.),c/(4*pi_R),"Marshak emission derivative");
        opts().problem=RADIATION_COUPLING;
        rad_grid g(dx);
        std::vector<std::vector<Real>> X(3,std::vector<Real>(H_N3));
        for(integer i=0;i<H_NX;++i)for(integer j=0;j<H_NX;++j)for(integer k=0;k<H_NX;++k) {
            const auto h=hindex(i,j,k);X[0][h]=(i-H_BW+.5)*dx;X[1][h]=(j-H_BW+.5)*dx;X[2][h]=(k-H_BW+.5)*dx;
        }
        g.set_X(X);
        const Real dt=g.max_timestep(0);
        for(auto& row:g.U)std::fill(row.begin(),row.end(),0);
        std::fill(g.U[0].begin(),g.U[0].end(),2);
        g.compute_flux(0);g.advance(dt,0);
        for(Real E:g.U[0])near(E,2,"constant-state transport");
        // Exercise every face and every direction with a periodic oblique beam.
        const Real n=1/std::sqrt(3.);
        Real sum0=0;
        for(integer i=RAD_BW;i<RAD_BW+INX;++i)for(integer j=RAD_BW;j<RAD_BW+INX;++j)for(integer k=RAD_BW;k<RAD_BW+INX;++k) {
            const auto r=rindex(i,j,k);
            const Real E=2+.2*std::sin(2*pi_R*(i+j+k-3*RAD_BW)/INX);
            g.U[0][r]=E;sum0+=E;
            for(int d=1;d<4;++d)g.U[d][r]=c*E*n;
        }
        for(int step=0;step<20;++step) {fill_bounds(g);g.compute_flux(0);g.advance(dt,0);}
        Real sum=0;
        for(integer i=RAD_BW;i<RAD_BW+INX;++i)for(integer j=RAD_BW;j<RAD_BW+INX;++j)for(integer k=RAD_BW;k<RAD_BW+INX;++k) {
            const auto r=rindex(i,j,k);sum+=g.U[0][r];
            radiation_m1::check_state({g.U[0][r],g.U[1][r],g.U[2][r],g.U[3][r]},c);
        }
        near(sum,sum0,"3D periodic energy conservation");
        // Stress realizability at discontinuous beam directions and large contrasts.
        std::mt19937_64 random(92342);
        std::uniform_real_distribution<Real> uniform(-1,1);
        for (int trial=0;trial<12;++trial) {
            for(integer i=RAD_BW;i<RAD_BW+INX;++i)for(integer j=RAD_BW;j<RAD_BW+INX;++j)for(integer k=RAD_BW;k<RAD_BW+INX;++k) {
                const auto r=rindex(i,j,k);
                const Real E=std::exp(8*uniform(random));
                radiation_m1::vector f{uniform(random),uniform(random),uniform(random)};
                const Real norm=std::hypot(f[0],f[1],f[2]);
                const Real mag=trial%2==0?1:std::abs(uniform(random));
                g.U[0][r]=E;
                for(int d=0;d<3;++d)g.U[d+1][r]=c*E*mag*f[d]/norm;
            }
            for(int step=0;step<3;++step) {fill_bounds(g);g.compute_flux(0);g.advance(dt,0);}
        }
        // Coupling uses different array strides when H_BW != RAD_BW.
        std::vector<Real> egas(H_N3,100),tau(H_N3,std::pow(100.,3./5)),sx(H_N3,0),sy=sx,sz=sx,rho(H_N3,10);
        std::vector<std::vector<Real>> species(1,rho);
        g.compute_mmw(species);
        for(integer r=0;r<RAD_N3;++r) {g.U[0][r]=4;g.U[1][r]=2;g.U[2][r]=-1;g.U[3][r]=0;}
        g.rad_imp(egas,tau,sx,sy,sz,rho,dt);
        constexpr integer D=H_BW-RAD_BW;
        for(integer i=RAD_BW;i<RAD_BW+INX;++i)for(integer j=RAD_BW;j<RAD_BW+INX;++j)for(integer k=RAD_BW;k<RAD_BW+INX;++k) {
            const auto r=rindex(i,j,k),h=hindex(i+D,j+D,k+D);
            near(egas[h]+g.U[0][r],104,"grid coupled energy");
            near(sx[h]+g.U[1][r]/c/c,2/c/c,"grid coupled x momentum");
            near(sy[h]+g.U[2][r]/c/c,-1/c/c,"grid coupled y momentum");
            near(std::pow(tau[h],5./3)+.5*(sx[h]*sx[h]+sy[h]*sy[h])/rho[h],egas[h],"grid dual energy");
        }
        near(egas[0],100,"gas ghosts untouched by source");
        // AMR packing takes hydro-coordinate bounds but reads radiation strides.
        // Include the wider hydro envelope to exercise unused halo padding too.
        for (int f=0;f<4;++f) for (integer r=0;r<RAD_N3;++r) g.U[f][r]=10000*f+r;
        const auto packed=g.get_subset({0,0,0},{H_NX,H_NX,H_NX});
        require(packed.size()==4*H_N3,"AMR envelope size");
        std::size_t packed_index=0;
        for (int f=0;f<4;++f) for(integer i=0;i<H_NX;++i)for(integer j=0;j<H_NX;++j)for(integer k=0;k<H_NX;++k) {
            const auto map=[](integer x){return std::clamp(x-D,integer(0),RAD_NX-1);};
            near(packed[packed_index++],10000*f+rindex(map(i),map(j),map(k)),"AMR radiation packing stride");
        }
        // Eight AMR child states preserve a coarse volume average and realizability.
        for(integer r=0;r<HS_N3;++r) {g.has_coarse[r]=1;g.is_coarse[r]=0;}
        const auto center=hSindex(H_BW,H_BW,H_BW);g.is_coarse[center]=1;
        for(int f=0;f<4;++f) std::fill(g.Ushad[f].begin(),g.Ushad[f].end(),f==0?4.:0.);
        g.Ushad[1][center]=c*3.99;
        g.Ushad[0][center-HS_DNX]=2;g.Ushad[0][center+HS_DNX]=8;
        g.complete_rad_amr_boundary();
        std::array<Real,4> mean{};
        for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int d=0;d<2;++d) {
            auto r=rindex(RAD_BW+a,RAD_BW+b,RAD_BW+d);
            radiation_m1::state u;
            for(int f=0;f<4;++f) {u[f]=g.U[f][r];mean[f]+=u[f]/8;}
            radiation_m1::check_state(u,c);
        }
        for(int f=0;f<4;++f)near(mean[f],g.Ushad[f][center],"AMR conservative interpolation");
        std::cout<<"Actual rad_grid kernels passed: H_BW="<<H_BW<<", RAD_BW="<<RAD_BW<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
'''
with tempfile.TemporaryDirectory(prefix='radiation-grid-') as tmp:
    path=pathlib.Path(tmp)
    (path/'test.cpp').write_text(fixture+opacities+'\n'+header+'\n'+bodies+checks)
    for h,r in [(3,3),(4,2)]:
        flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if '--sanitize' in sys.argv else (['-O3','-DNDEBUG'] if '--release' in sys.argv else ['-O2'])
        subprocess.run([os.environ.get('CXX','g++'),'-std=c++23',*flags,'-I'+str(root),f'-DHYDRO_WIDTH={h}',f'-DRAD_WIDTH={r}',str(path/'test.cpp'),'-o',str(path/'test')],check=True)
        subprocess.run([str(path/'test')],check=True)
