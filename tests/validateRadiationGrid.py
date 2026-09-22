#!/usr/bin/env python3
"""Compile and exercise production M1 transport loops without HPX.

Options, geometry, opacity values, and distributed infrastructure are fixtures.
The rad_grid declaration and allocation/reconstruction/flux/update/source/AMR
methods are read from the working tree on every invocation; no numerical method
is copied into this test. These serial checks do not replace application or MPI
runs, nor validate physical opacity models.

Usage: python3 tests/validateRadiationGrid.py
       python3 tests/validateRadiationGrid.py --build-dir /tmp/m1-grid-check

An explicit build directory preserves generated sources, binaries, and GCC's
release vectorization report. CXX can select another GCC-compatible compiler.
"""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


root = Path(__file__).resolve().parents[1]
methods = (
    "void rad_grid::allocate()",
    "void rad_grid::set_dx(",
    "void rad_grid::set_X(",
    "Real rad_grid::maxTimestep(",
    "void rad_grid::compute_flux(",
    "void rad_grid::set_physical_boundaries(",
    "void rad_grid::advance(",
    "void rad_grid::sanity_check()",
    "Real radiationGasInternal(",
    "void rad_grid::rad_imp(",
    "void rad_grid::compute_mmw(",
    "void rad_grid::prepareSources(",
    "void rad_grid::finishSources(",
    "radiationConservation::Totals rad_grid::takeConservation()",
    "std::vector<Real> rad_grid::get_restrict()",
    "std::vector<Real> rad_grid::get_flux_restrict(",
    "void rad_grid::set_flux_restrict(",
    "void rad_grid::complete_rad_amr_boundary()",
    "rad_grid::rad_grid(Real _dx)",
    "rad_grid::rad_grid()",
)


def extractMethod(source, signature):
    """The selected definitions use balanced braces, including in comments."""
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source.count("\n", 0, begin) + 1, source[begin:end]


fixture = r'''
#include "octotiger/radiation/m1.hpp"
#include "octotiger/radiation/conservation.hpp"
#include "octotiger/radiation/grey_opacity.hpp"
#include "octotiger/test_problems/radiation/profiles.hpp"
#include "octotiger/math/Debug.hpp"
#include <array>
#include <atomic>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define PROFILE()
using integer = long long;
constexpr int INX=8, NRF=4, NDIM=3, H_BW=3, RAD_BW=3, NCHILD=8;
constexpr int RAD_NX=INX+2*RAD_BW, RAD_N3=RAD_NX*RAD_NX*RAD_NX;
constexpr int H_NX=RAD_NX, H_N3=RAD_N3;
constexpr int HS_NX=INX/2+2*H_BW, HS_N3=HS_NX*HS_NX*HS_NX;
constexpr int HS_DNX=HS_NX*HS_NX, HS_DNY=HS_NX, HS_DNZ=1;
constexpr int XDIM=0, YDIM=1, ZDIM=2;
constexpr int WD=1, MARSHAK=2;
constexpr int RADIATION_EQUILIBRIUM_SPHERE=3;
constexpr int rho_i=0,egas_i=1,tau_i=2,sx_i=3,sy_i=4,sz_i=5,spc_i=6;
constexpr Real ZERO=0;
constexpr Real lightSpeed=2.99792458e10;
using M1=RadiationM1<Real,NDIM>;
integer rindex(integer i,integer j,integer k) { return k+RAD_NX*(j+RAD_NX*i); }
integer hindex(integer i,integer j,integer k) { return rindex(i,j,k); }
integer hSindex(integer i,integer j,integer k) { return k+HS_NX*(j+HS_NX*i); }
struct OptionsFixture {
    radiation::GreyOpacity radiationOpacity;
    Real radCfl=.25,radCRatio=1,radTheta=1,radOpacity=-1;
    Real code_to_g=1,code_to_cm=1,xscale=.5,dual_energy_sw1=.001;
    int eos=0,problem=0,n_fields=7,n_species=1;
    bool rad_implicit=false,radVelocityTerms=true;
    std::string radEnergyMode="thermal";
};
OptionsFixture& opts() { static OptionsFixture o; return o; }
struct ConstantsFixture { Real c=lightSpeed,sigma=lightSpeed/4,mh=1,kb=1; };
ConstantsFixture& physcon() { static ConstantsFixture p; return p; }
struct grid { static Real get_fgamma() { return 5.0/3.0; } };
Real ztwd_energy(Real) { throw std::runtime_error("WD EOS is not part of this fixture"); }
Real absorptionOpacity=0,totalOpacity=.8;
Real kappa_p(Real,Real,Real,Real,Real,Real) { return absorptionOpacity; }
Real kappa_R(Real,Real,Real,Real,Real,Real) { return totalOpacity; }
template<class T> using specie_state_t=std::array<T,1>;
void mean_ion_weight(const specie_state_t<Real>&,Real& mmw,Real& X,Real& Z) {mmw=1;X=1;Z=0;}
bool radiationFixedMediumProblem() { return false; }
radiationTests::Parameters radiationTestParameters() {
    radiationTests::Parameters p;p.c=physcon().c;return p;
}
std::vector<Real> marshakWaveAnalytic(Real,Real,Real,Real) {
    throw std::runtime_error("Marshak boundaries are not part of this fixture");
}
struct silo_var_t {};
namespace geo {
struct direction {}; struct octant {};
struct face { int index;face(int v):index(v){} operator int() const {return index;} };
struct dimension {
    int index;
    constexpr dimension(int value=0):index(value) {}
    constexpr operator int() const { return index; }
    static std::array<dimension,3> full_set() { return {dimension(0),dimension(1),dimension(2)}; }
};
}
#define private public
'''


checks = r'''
#undef private
// Fixture profiles deliberately use physical grid fields, not normalized M1 states.
using State=std::array<Real,NRF>;
using Point=std::array<Real,NDIM>;
using Totals=std::array<long double,NRF>;
constexpr Real cellWidth=7.5e9;
constexpr Real domainWidth=INX*cellWidth;

void require(bool condition,const std::string& label) {
    if(!condition) throw std::runtime_error(label);
}
void near(long double actual,long double expected,long double scale,
          const std::string& label,long double tolerance=3e-12L) {
    if(!std::isfinite(actual)||std::abs(actual-expected)>tolerance*std::abs(scale)) {
        std::ostringstream message;
        message<<std::setprecision(20)<<label<<": actual="<<actual
               <<" expected="<<expected<<" scale="<<scale;
        throw std::runtime_error(message.str());
    }
}
template<class Function> void interior(Function function) {
    for(int i=RAD_BW;i<RAD_BW+INX;++i)
        for(int j=RAD_BW;j<RAD_BW+INX;++j)
            for(int k=RAD_BW;k<RAD_BW+INX;++k) function(rindex(i,j,k));
}
template<class Function> void initialize(rad_grid& grid,Function profile,Real shift=0) {
    std::vector<std::vector<Real>> coordinates(NDIM,std::vector<Real>(H_N3));
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k) {
        auto r=rindex(i,j,k);
        Point point{(i-RAD_BW+.5-INX/2.)*cellWidth+shift,
                    (j-RAD_BW+.5-INX/2.)*cellWidth,
                    (k-RAD_BW+.5-INX/2.)*cellWidth};
        State state=profile(point);
        for(int d=0;d<NDIM;++d) coordinates[d][r]=point[d];
        for(int f=0;f<NRF;++f) grid.U[f][r]=state[f];
    }
    grid.set_dx(cellWidth);
    grid.set_X(coordinates);
}
void periodic(rad_grid& grid) {
    auto wrap=[](int i) { return RAD_BW+(i-RAD_BW+INX)%INX; };
    for(int i=0;i<RAD_NX;++i)for(int j=0;j<RAD_NX;++j)for(int k=0;k<RAD_NX;++k)
        for(int f=0;f<NRF;++f)
            grid.U[f][rindex(i,j,k)]=grid.U[f][rindex(wrap(i),wrap(j),wrap(k))];
}
Totals totals(const rad_grid& grid) {
    Totals result{};
    interior([&](auto r) {
        for(int f=0;f<NRF;++f) result[f]+=static_cast<long double>(grid.U[f][r]);
    });
    return result;
}
void checkRealizability(const rad_grid& grid,const std::string& label) {
    interior([&](auto r) {
        Real const energy=grid.U[0][r];
        require(std::isfinite(energy)&&energy>=0,label+": energy is nonnegative and finite");
        long double flux2=0;
        for(int f=1;f<NRF;++f) {
            require(std::isfinite(grid.U[f][r]),label+": finite flux");
            if(energy==0) require(grid.U[f][r]==0,label+": vacuum has zero flux");
            else {
                long double reduced=static_cast<long double>(grid.U[f][r])/lightSpeed/energy;
                flux2+=reduced*reduced;
            }
        }
        require(flux2<=1+3e-12L,label+": |F| <= c E");
    });
}
void compareTotals(const Totals& actual,const Totals& expected,const std::string& label) {
    for(int f=0;f<NRF;++f)
        near(actual[f],expected[f],std::max(1.L,expected[0])*(f?lightSpeed:1),
             label+" field "+std::to_string(f));
}

State smoothPrimitives(Point point) {
    Point phase{};
    for(int d=0;d<NDIM;++d) phase[d]=2*std::numbers::pi*point[d]/domainWidth;
    Real const h=1.5+.2*std::cos(phase[0])+.1*std::sin(phase[1]);
    Point beta{.25+.12*std::sin(phase[0]),-.16+.11*std::cos(phase[1]),.1*std::sin(phase[2])};
    Real beta2=0;
    for(Real value:beta) beta2+=value*value;
    // Independent analytic Hanawa relation: E=(3+beta^2)H/4, F=c H beta.
    return State{.25*(3+beta2)*h,lightSpeed*h*beta[0],lightSpeed*h*beta[1],lightSpeed*h*beta[2]};
}
void periodicTransport(const std::string& name) {
    rad_grid grid;
    initialize(grid,[&](Point x) {
        if(name=="smooth") return smoothPrimitives(x);
        if(name=="isotropic") {
            Real energy=1+.25*std::cos(2*std::numbers::pi*x[0]/domainWidth)
                         +.15*std::sin(2*std::numbers::pi*x[1]/domainWidth);
            return State{energy,0,0,0};
        }
        Point mode=name=="oblique"?Point{2,-3,1}:Point{-1,0,0};
        Real phase=0,norm2=0;
        for(int d=0;d<NDIM;++d) { phase+=mode[d]*x[d]; norm2+=mode[d]*mode[d]; }
        Real const energy=1+.25*std::cos(2*std::numbers::pi*phase/domainWidth);
        Real const scale=lightSpeed*energy/std::sqrt(norm2);
        return State{energy,scale*mode[0],scale*mode[1],scale*mode[2]};
    });
    Totals const initial=totals(grid);
    for(int step=0;step<24;++step) {
        periodic(grid);
        Real const dt=.71*grid.maxTimestep(0);
        require(std::isfinite(dt)&&dt>0,"positive finite timestep");
        require(dt<=cellWidth/(12*lightSpeed),"multidimensional transport CFL");
        grid.compute_flux(dt,0);
        grid.advance(dt,0);
        checkRealizability(grid,name);
        compareTotals(totals(grid),initial,name+" periodic conservation");
    }
    if(name=="streaming") interior([&](auto r) {
        near(grid.U[1][r],-lightSpeed*grid.U[0][r],lightSpeed*grid.U[0][r],"axis beam F=-cE");
        near(grid.U[2][r],0,lightSpeed*grid.U[0][r],"axis beam Fy");
        near(grid.U[3][r],0,lightSpeed*grid.U[0][r],"axis beam Fz");
    });
}
void uniformStates() {
    std::array<State,3> states{State{0,0,0,0},State{2,0,0,0},State{2,.6*lightSpeed,-.2*lightSpeed,.3*lightSpeed}};
    for(const auto& state:states) {
        rad_grid grid(cellWidth);
        initialize(grid,[&](Point) { return state; });
        Real const dt=.71*grid.maxTimestep(0);
        grid.compute_flux(dt,0);
        grid.advance(dt,0);
        checkRealizability(grid,"uniform");
        interior([&](auto r) {
            for(int f=0;f<NRF;++f)
                near(grid.U[f][r],state[f],(f?lightSpeed:1)*std::max(Real(1),state[0]),"uniform unchanged");
        });
    }
}
void rotation() {
    rad_grid grid(cellWidth);
    State const initial{2,.8*lightSpeed,-.3*lightSpeed,.2*lightSpeed};
    initialize(grid,[&](Point) { return initial; });
    Real const omega=.13;
    require(grid.maxTimestep(omega)<grid.maxTimestep(0),"grid motion decreases timestep");
    Real const dt=.71*grid.maxTimestep(omega);
    Real angle=0;
    for(int step=0;step<24;++step) {
        // Fill all ghosts with the current spatially constant state.
        periodic(grid);
        grid.compute_flux(dt,omega);
        grid.advance(dt,omega);
        angle+=omega*dt;
        Real const cs=std::cos(angle),sn=std::sin(angle);
        interior([&](auto r) {
            near(grid.U[0][r],initial[0],initial[0],"rotation energy");
            near(grid.U[1][r],cs*initial[1]+sn*initial[2],lightSpeed*initial[0],"rotation Fx");
            near(grid.U[2][r],cs*initial[2]-sn*initial[1],lightSpeed*initial[0],"rotation Fy");
            near(grid.U[3][r],initial[3],lightSpeed*initial[0],"rotation Fz");
            Real magnitude=std::hypot(grid.U[1][r],grid.U[2][r],grid.U[3][r]);
            near(magnitude,std::hypot(initial[1],initial[2],initial[3]),lightSpeed*initial[0],"rotation magnitude");
        });
        checkRealizability(grid,"rotation");
    }
}
Totals outwardFlux(const rad_grid& grid,int omitFace) {
    Totals result{};
    for(int face=0;face<6;++face) {
        if(face==omitFace) continue;
        int const axis=face/2,side=face%2;
        for(int a=RAD_BW;a<RAD_BW+INX;++a)for(int b=RAD_BW;b<RAD_BW+INX;++b) {
            std::array<int,3> index{};
            index[axis]=RAD_BW+side*INX;
            index[(axis+1)%3]=a;
            index[(axis+2)%3]=b;
            auto r=rindex(index[0],index[1],index[2]);
            for(int f=0;f<NRF;++f)
                result[f]+=(side?1.L:-1.L)*grid.flux[axis][f][r];
        }
    }
    return result;
}
void adjacentBlocks() {
    rad_grid left(cellWidth),right(cellWidth);
    initialize(left,smoothPrimitives,-domainWidth/2);
    initialize(right,smoothPrimitives,domainWidth/2);
    Totals initial=totals(left),other=totals(right);
    for(int f=0;f<NRF;++f) initial[f]+=other[f];
    Real const dt=.71*left.maxTimestep(0);
    left.compute_flux(dt,0); right.compute_flux(dt,0);
    for(int j=RAD_BW;j<RAD_BW+INX;++j)for(int k=RAD_BW;k<RAD_BW+INX;++k)
        for(int f=0;f<NRF;++f)
            near(left.flux[0][f][rindex(RAD_BW+INX,j,k)],right.flux[0][f][rindex(RAD_BW,j,k)],
                 f?lightSpeed*lightSpeed:lightSpeed,"matching block interface");
    Totals out=outwardFlux(left,1),outRight=outwardFlux(right,0);
    left.advance(dt,0); right.advance(dt,0);
    Totals final=totals(left),finalRight=totals(right);
    for(int f=0;f<NRF;++f)
        final[f]+=finalRight[f]+static_cast<long double>(dt)/cellWidth*(out[f]+outRight[f]);
    compareTotals(final,initial,"external boundary flux accounts for two-block update");
    checkRealizability(left,"left block"); checkRealizability(right,"right block");
}
void physicalStorageUnits() {
    rad_grid grid(cellWidth);
    State const state{2,.6*lightSpeed,-.2*lightSpeed,.3*lightSpeed};
    initialize(grid,[&](Point) { return state; });
    grid.compute_flux(.71*grid.maxTimestep(0),0);
    Point const q{.6,-.2,.3};
    Real const q2=q[0]*q[0]+q[1]*q[1]+q[2]*q[2];
    Real const h=(2*state[0]+std::sqrt(4*state[0]*state[0]-3*q2))/3;
    Point const beta{q[0]/h,q[1]/h,q[2]/h};
    Real const pressure=.25*h*(1-q2/(h*h));
    interior([&](auto r) {
        // VL stores E,Q=F/c, not the removed Hanawa primitive arrays. For
        // uniform data, predictor and reconstructed face states are exact.
        for(int f=0;f<NRF;++f) {
            Real const normalized=state[f]/(f?lightSpeed:1);
            near(grid.Uhalf[f][r],normalized,state[0],"physical storage to predictor E,Q");
            for(int side=0;side<2;++side)
                near(grid.faces[side][f][r],normalized,state[0],"physical storage to face E,Q");
        }
        for(int normal=0;normal<NDIM;++normal) {
            near(grid.flux[normal][0][r],state[normal+1],lightSpeed*state[0],"physical energy flux");
            for(int d=0;d<NDIM;++d) {
                Real const tensor=h*beta[normal]*beta[d]+(normal==d?pressure:0);
                near(grid.flux[normal][d+1][r],lightSpeed*lightSpeed*tensor,
                     lightSpeed*lightSpeed*state[0],"physical F transport flux");
            }
        }
    });
    auto restricted=grid.get_restrict();
    int const coarseCount=INX*INX*INX/NCHILD;
    for(int f=0;f<NRF;++f)for(int r=0;r<coarseCount;++r)
        near(restricted[f*coarseCount+r],state[f],(f?lightSpeed:1)*state[0],"volume restriction stays physical");
    // Existing face restriction/reflux messages remain in physical units too.
    std::array<integer,NDIM> const lo{RAD_BW,RAD_BW,RAD_BW},hi{RAD_BW+1,RAD_BW+2,RAD_BW+2};
    auto face=grid.get_flux_restrict(lo,hi,geo::dimension(XDIM));
    require(face.size()==NRF,"one restricted face");
    integer const sample=rindex(RAD_BW,RAD_BW,RAD_BW);
    for(int f=0;f<NRF;++f)
        near(face[f],grid.flux[XDIM][f][sample],(f?lightSpeed*lightSpeed:lightSpeed)*state[0],
             "face restriction stays physical");
    std::array<integer,NDIM> const oneHi{RAD_BW+1,RAD_BW+1,RAD_BW+1};
    for(int f=0;f<NRF;++f) grid.flux[XDIM][f][sample]=0;
    grid.set_flux_restrict(face,lo,oneHi,geo::dimension(XDIM));
    for(int f=0;f<NRF;++f)
        near(grid.flux[XDIM][f][sample],face[f],(f?lightSpeed*lightSpeed:lightSpeed)*state[0],
             "reflux setter stays physical");
}
void sourceStorageUnits() {
    State const state{2,.8*lightSpeed,-.3*lightSpeed,.2*lightSpeed};
    for(bool absorption:{false,true}) {
        rad_grid radiation(cellWidth);
        initialize(radiation,[&](Point) { return state; });
        Real const gamma=grid::get_fgamma();
        std::vector<std::vector<Real>> gas(7,std::vector<Real>(H_N3));
        gas[egas_i].assign(H_N3,5);gas[tau_i].assign(H_N3,std::pow(5,1/gamma));
        gas[rho_i].assign(H_N3,1);gas[spc_i].assign(H_N3,1);
        auto& egas=gas[egas_i];auto& tau=gas[tau_i];auto& rho=gas[rho_i];
        auto& sx=gas[sx_i];auto& sy=gas[sy_i];auto& sz=gas[sz_i];
        absorptionOpacity=absorption?.3:0;
        totalOpacity=.8;
        Real const dt=.5/(lightSpeed*totalOpacity);
        auto const initialBudget=radiation.takeConservation();
        // The production driver splits thermal exchange from subcycled
        // momentum damping. Exercise both halves in the same order.
        opts().rad_implicit=true;
        radiation.compute_mmw(gas);
        radiation.rad_imp(egas,tau,sx,sy,sz,rho,dt);
        periodic(radiation);
        // The fixture refreshes the uniform material halo after thermal exchange.
        auto const sample=rindex(RAD_BW,RAD_BW,RAD_BW);
        Real const thermalGas=egas[sample],thermalTau=tau[sample];
        for(int i=0;i<H_NX;++i)for(int j=0;j<H_NX;++j)for(int k=0;k<H_NX;++k)
            if(i<H_BW||i>=H_BW+INX||j<H_BW||j>=H_BW+INX||k<H_BW||k>=H_BW+INX) {
                auto const h=hindex(i,j,k);
                egas[h]=thermalGas;tau[h]=thermalTau;
            }
        radiation.prepareSources(gas);
        radiation.compute_flux(dt,0);
        radiation.advance(dt,0);
        radiation.finishSources(gas);
        auto const sourceBudget=radiation.takeConservation();
        auto const drainedBudget=radiation.takeConservation();
        for(int f=0;f<NRF;++f) {
            long double const scale=initialBudget.value[0]*(f?lightSpeed:1);
            near(sourceBudget.value[f]-initialBudget.value[f]-sourceBudget.source[f],0,scale,
                 "matter coupling radiation source budget uses physical fields");
            near(sourceBudget.boundary[f],0,scale,"matter coupling has no boundary transport");
            near(drainedBudget.source[f],0,scale,"matter source budget drains once");
        }
        interior([&](auto r) {
            near(egas[r]+radiation.U[0][r],5+state[0],5+state[0],"source total energy");
            Point const momentum{sx[r],sy[r],sz[r]};
            for(int d=0;d<NDIM;++d) {
                long double const initialMomentum=static_cast<long double>(state[d+1])/lightSpeed/lightSpeed;
                long double const finalMomentum=momentum[d]+static_cast<long double>(radiation.U[d+1][r])/lightSpeed/lightSpeed;
                near(finalMomentum,initialMomentum,state[0]/lightSpeed,"source total physical momentum");
                if(!absorption)
                    near(radiation.U[d+1][r],state[d+1]/1.5,lightSpeed*state[0],"physical F scattering damping");
            }
            require(std::isfinite(tau[r])&&tau[r]>0,"source tau remains finite and positive");
        });
        checkRealizability(radiation,"source storage conversion");
    }
    absorptionOpacity=0;
    opts().rad_implicit=false;
}
void coarseFineStorageUnits() {
    // One flagged coarse cell populates eight fine ghost cells. All eight must
    // average to the physical coarse state, including when the cone limiter acts.
    for(bool streaming:{false,true}) {
        rad_grid grid(cellWidth);
        for(auto& flag:grid.has_coarse) flag=1;
        for(auto& flag:grid.is_coarse) flag=0;
        int const ci=2,cj=4,ck=4;
        integer const coarse=hSindex(ci,cj,ck);
        grid.is_coarse[coarse]=1;
        for(int i=0;i<HS_NX;++i)for(int j=0;j<HS_NX;++j)for(int k=0;k<HS_NX;++k) {
            Real const energy=2+.1*(i-ci)+.05*(j-cj)+.04*(k-ck);
            Real const angle=.14*(i-ci)+.07*(j-cj)-.05*(k-ck);
            Real const factor=streaming?1:.7;
            State const state{energy,factor*lightSpeed*energy*std::cos(angle),
                                    factor*lightSpeed*energy*std::sin(angle),0};
            for(int f=0;f<NRF;++f) grid.Ushad[f][hSindex(i,j,k)]=state[f];
        }
        grid.complete_rad_amr_boundary();
        Totals sum{};
        for(int child=0;child<NCHILD;++child) {
            integer const r=rindex(2*ci-H_BW+((child>>2)&1),2*cj-H_BW+((child>>1)&1),
                                   2*ck-H_BW+(child&1));
            for(int f=0;f<NRF;++f) sum[f]+=grid.U[f][r];
            Real const magnitude=std::hypot(grid.U[1][r],grid.U[2][r],grid.U[3][r]);
            require(grid.U[0][r]>=0&&magnitude<=lightSpeed*grid.U[0][r]*(1+3e-12),
                    "AMR child remains physically realizable");
        }
        for(int f=0;f<NRF;++f)
            near(sum[f]/NCHILD,grid.Ushad[f][coarse],(f?lightSpeed:1)*grid.Ushad[0][coarse],
                 "AMR children preserve physical parent mean");
    }
}
int main() {
    try {
        uniformStates();
        for(const auto& name:{"streaming","oblique","isotropic","smooth"}) periodicTransport(name);
        rotation(); adjacentBlocks();
        physicalStorageUnits(); sourceStorageUnits(); coarseFineStorageUnits();
        std::cout<<"PASS: uniform/vacuum, axis/oblique streaming, isotropic, smooth H/beta, rotation, adjacent blocks, physical storage/flux/reflux, source coupling/budgets, AMR means\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
'''


def generatedSource():
    sourcePath = root / "src/radiation/rad_grid.cpp"
    headerPath = root / "octotiger/radiation/rad_grid.hpp"
    source = sourcePath.read_text()
    # Preserve header line numbering for useful compiler diagnostics.
    header = "\n".join(
        "" if line.lstrip().startswith("#include") else line
        for line in headerPath.read_text().splitlines()
    )
    opacityPath = root / "octotiger/radiation/opacities.hpp"
    opacity = opacityPath.read_text()
    pieces = [fixture]
    for signature in ("inline Real radiationAbsorption(", "inline Real radiationTransport("):
        line, method = extractMethod(opacity, signature)
        pieces.extend([f'\n#line {line} "{opacityPath}"', method])
    pieces.extend([f'\n#line 1 "{headerPath}"', header])
    for signature in methods:
        line, method = extractMethod(source, signature)
        pieces.extend([f'\n#line {line} "{sourcePath}"', method])
    pieces.extend(['\n#line 1 "gridChecks.cpp"', checks])
    return "\n".join(pieces)


def validate(folder, modes):
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError("CXX must select an available C++23 compiler")
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "radiationGrid.cpp"
    source.write_text(generatedSource())
    flags = {
        "debug": ["-O0", "-g"],
        "release": ["-O3", "-DNDEBUG", f"-fopt-info-vec-all={folder / 'vectorization.txt'}"],
        "ubsan": ["-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
    }
    for mode in modes:
        if mode == "release":
            (folder / "vectorization.txt").unlink(missing_ok=True)
        executable = folder / f"radiationGrid-{mode}"
        command = compiler + ["-std=c++23", "-Wall", "-Wextra"] + flags[mode]
        command += ["-I" + str(root), str(source), "-o", str(executable)]
        subprocess.run(command, check=True)
        print(f"{mode}: ", end="", flush=True)
        subprocess.run([str(executable)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--mode", choices=("debug", "release", "ubsan"), action="append")
    arguments = parser.parse_args()
    modes = arguments.mode or ["debug", "release", "ubsan"]
    if arguments.build_dir:
        validate(arguments.build_dir.resolve(), modes)
    else:
        with tempfile.TemporaryDirectory(prefix="radiation-grid-") as folder:
            validate(Path(folder), modes)


if __name__ == "__main__":
    main()
