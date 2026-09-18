#!/usr/bin/env python3
"""Run the actual rad_grid transport/source methods on a serial uniform mesh.

This provides local numerical threshold calibration without HPX. Distributed
boundary actions are tested only by the full Octo-TIGER CTest cases.
"""
import argparse, pathlib, subprocess, tempfile
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path,required=True)
p.add_argument('--cells',type=int,default=32);p.add_argument('--cxx',default='g++')
p.add_argument('--sanitize',action='store_true');p.add_argument('--output',type=pathlib.Path)
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
cpp=(root/'src/radiation/rad_grid.cpp').read_text()
header='\n'.join(l for l in (root/'octotiger/radiation/rad_grid.hpp').read_text().splitlines() if not l.startswith('#include'))
def method(start):
    begin=cpp.index(start);end=cpp.index('{',begin)+1;depth=1
    while depth:
        if cpp[end]=='{':depth+=1
        elif cpp[end]=='}':depth-=1
        end+=1
    return cpp[begin:end]
bodies='\n'.join(method(s) for s in ['void rad_grid::allocate()', 'void rad_grid::set_dx(',
 'void rad_grid::set_X(', 'Real rad_grid::max_timestep(', 'void rad_grid::compute_flux(',
 'void rad_grid::advance(', 'void rad_grid::sanity_check()', 'void rad_grid::regression_source(',
 'void rad_grid::set_physical_boundaries(', 'rad_grid::rad_grid(Real _dx)', 'rad_grid::rad_grid()'])
app_header='\n'.join(l for l in (root/'octotiger/test_problems/radiation.hpp').read_text().splitlines() if not l.startswith(('#include','#pragma once')))
app_cpp='\n'.join(l for l in (root/'src/test_problems/radiation/radiation.cpp').read_text().splitlines() if not l.startswith('#include'))
fixture=r'''
#include "octotiger/radiation/m1.hpp"
#include "octotiger/test_problems/radiation/reference.hpp"
#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#define PROFILE()
#define OCTOTIGER_EXPORT
using integer=long long;
constexpr int INX=TEST_CELLS,NRF=4,NDIM=3,NCHILD=8,H_BW=3,RAD_BW=3;
constexpr int RAD_NX=INX+2*RAD_BW,RAD_N3=RAD_NX*RAD_NX*RAD_NX,H_NX=RAD_NX,H_N3=RAD_N3;
constexpr int HS_NX=INX/2+2*H_BW,HS_N3=HS_NX*HS_NX*HS_NX;
constexpr int XDIM=0,YDIM=1,ZDIM=2,MARSHAK=9,RADIATION_EQUILIBRIUM_SPHERE=3;
constexpr int RADIATION_STREAMING_WAVE=0,RADIATION_STREAMING_FRONT=1,RADIATION_GAUSSIAN_PULSE=2;
constexpr int rho_i=0,egas_i=1,tau_i=2,spc_i=3;
integer rindex(integer i,integer j,integer k){return k+RAD_NX*(j+RAD_NX*i);}
integer hindex(integer i,integer j,integer k){return rindex(i,j,k);}
struct options_fixture {
    Real cfl=.4,xscale=1,stop_time=.2,omega=0;
    Real rad_test_chi=5,rad_test_width=.2,rad_test_background=1,rad_test_amplitude=.01,rad_test_luminosity=.01;
    int problem=0,n_fields=15,max_level=0;
    bool radiation=true,hydro=false,gravity=false,unigrid=true,periodic=true,rad_implicit=false;
    std::string rad_reference;
};
options_fixture& opts(){static options_fixture o;return o;}
struct constants_fixture {Real c=1;};
constants_fixture& physcon(){static constants_fixture p;return p;}

std::vector<Real> marshak_wave_analytic(Real,Real,Real,Real){throw std::runtime_error("Unused Marshak fixture");}
struct silo_var_t{};
namespace geo {using face=int;struct direction{};struct dimension{};struct octant{};}
#define private public
'''
checks=r'''
#undef private
int main(int argc,char** argv) {
 try {
    auto const ref=radiation_tests::reference_data::read(argv[1]);
    if(ref.n!=INX) throw std::runtime_error("fixture/reference size mismatch");
    auto const& p=ref.p; physcon().c=p.c;
    opts().rad_reference=argv[1];opts().stop_time=p.time;opts().xscale=p.length/2;
    opts().rad_test_chi=p.chi;opts().rad_test_width=p.width;opts().rad_test_background=p.background;
    opts().rad_test_amplitude=p.amplitude;opts().rad_test_luminosity=p.luminosity;
    double const dx=p.length/INX;
    std::vector<std::vector<Real>> X(3,std::vector<Real>(H_N3));
    for(int i=0;i<H_NX;++i)for(int j=0;j<H_NX;++j)for(int k=0;k<H_NX;++k){
        auto r=hindex(i,j,k);X[0][r]=-p.length/2+(i-H_BW+.5)*dx;
        X[1][r]=-p.length/2+(j-H_BW+.5)*dx;X[2][r]=-p.length/2+(k-H_BW+.5)*dx;
    }
    const char* names[]{"STREAMING_WAVE","STREAMING_FRONT","GAUSSIAN_PULSE","EQUILIBRIUM_SPHERE"};
    for(int test=0;test<4;++test){
        opts().problem=test;opts().periodic=test!=3;opts().rad_implicit=test>=2;
        validate_radiation_test();rad_grid g(dx);g.set_X(X);
        auto analytic=[&](radiation_tests::point x,double t){
            auto const full=t==0?radiation_regression_init(x[0],x[1],x[2],dx):
                radiation_regression_analytic(x[0],x[1],x[2],t);
            radiation_tests::state u;
            for(int f=0;f<4;++f)u[f]=full[opts().n_fields+f];
            return u;
        };
        for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k){
            auto r=rindex(i,j,k);auto u=analytic({X[0][r],X[1][r],X[2][r]},0);
            for(int f=0;f<4;++f)g.U[f][r]=u[f];
        }
        double t=0;
        while(t<p.time){
            if(test==3){for(int face=0;face<6;++face)g.set_physical_boundaries(face,t);}
            else {
                auto wrap=[](int x){return RAD_BW+(x-RAD_BW+INX)%INX;};
                for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
                    for(int f=0;f<4;++f)g.U[f][rindex(i,j,k)]=g.U[f][rindex(wrap(i),wrap(j),wrap(k))];
            }
            double const dt=std::min(g.max_timestep(0),p.time-t);
            g.compute_flux(0);g.advance(dt,0);
            if(test>=2)g.regression_source(dt);
            t+=dt;
        }
        std::array<double,4> l1{},l2{},linf{};
        for(int i=RAD_BW;i<RAD_BW+INX;++i)for(int j=RAD_BW;j<RAD_BW+INX;++j)for(int k=RAD_BW;k<RAD_BW+INX;++k){
            auto r=rindex(i,j,k);auto u=analytic({X[0][r],X[1][r],X[2][r]},t);
            for(int f=0;f<4;++f){double e=std::abs(u[f]-g.U[f][r]);l1[f]+=e;l2[f]+=e*e;linf[f]=std::max(linf[f],e);}
        }
        std::cout<<"RADIATION_TEST_FINISHED RADIATION_"<<names[test]<<" t="<<std::setprecision(17)<<t<<'\n';
        const char* fields[]{"er","fx","fy","fz"};
        for(int f=0;f<4;++f)std::cout<<fields[f]<<" "<<std::scientific<<l1[f]/(INX*INX*INX)<<" "<<std::sqrt(l2[f]/(INX*INX*INX))<<" "<<linf[f]<<'\n';
    }
 } catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
'''
with tempfile.TemporaryDirectory(prefix='radiation-regression-') as tmp:
    folder=pathlib.Path(tmp);src=folder/'test.cpp';exe=folder/'test'
    src.write_text(fixture+app_header+'\n'+app_cpp+'\n'+header+'\n'+bodies+'\n'+checks)
    flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if a.sanitize else ['-O3']
    subprocess.run([a.cxx,'-std=c++23',*flags,'-I'+str(root),'-DTEST_CELLS='+str(a.cells),str(src),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe),str(a.reference.resolve())],check=True,text=True,stdout=subprocess.PIPE)
    print(result.stdout,end='')
    if a.output:a.output.write_text(result.stdout)
