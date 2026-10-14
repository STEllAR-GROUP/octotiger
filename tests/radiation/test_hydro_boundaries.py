#!/usr/bin/env python3
"""Execute the production shock boundary branches and opacity functions.

Only the new radiation-hydro branches are selected from the application source;
the numerical assignments are not copied here. Geometry and field containers
are lightweight fixtures so the check also runs without HPX or Silo.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def block(source, signature):
    begin = source.index(signature)
    end = source.index("{", begin) + 1
    depth = 1
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source[begin:end]


FIXTURE = r'''
#include "octotiger/math/Debug.hpp"
#include "test_problems/radiation/hydro_benchmarks.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define ENABLE_THREAD_DEBUG()
using integer = long long;
constexpr int NDIM=3, INX=8, H_BW=3, RAD_BW=3, H_NX=14, RAD_NX=14;
constexpr int H_DNX=196,H_DNY=14,H_DNZ=1,H_DN[3]={196,14,1};
constexpr int XDIM=0,YDIM=1,ZDIM=2,NRF=4,er_i=0,fx_i=1,fy_i=2,fz_i=3;
constexpr int rho_i=0,egas_i=1,sx_i=2,sy_i=3,sz_i=4,tau_i=5,lx_i=6,ly_i=7,lz_i=8;
constexpr int RADIATION_SUBCRITICAL_SHOCK=18,RADIATION_SUPERCRITICAL_SHOCK=19,RADIATION_BLAST_WAVE=20;
constexpr int RADIATION_COUPLED_WAVE=21;
constexpr int MARSHAK=7,RADIATION_TEST=9,RADIATION_DIFFUSION=12,RADIATION_COUPLING=13;
constexpr Real MARSHAK_OPAC=1;
namespace geo {
constexpr int MINUS=0,PLUS=1;
struct face {
    int number;
    int get_dimension() const { return number/2; }
    int get_side() const { return number%2; }
    operator int() const { return number; }
};
}
int index(int i,int j,int k) { return 196*i+14*j+k; }
integer rindex(integer i,integer j,integer k) { return index(i,j,k); }
struct Options {
    int n_fields=11, problem=RADIATION_SUBCRITICAL_SHOCK;
    Real xscale=3.5e10,code_to_g=1,code_to_cm=1,radTestChi=1;
    std::string radiation_model="m1";
};
Options& opts() { static Options value; return value; }
struct Constants { Real c=2.99792458e10,kb=1.380658e-16,mh=1.6733e-24,sigma=5.67051e-5; };
Constants& physcon() { static Constants value; return value; }
bool radiationShockProblem() { return true; }
std::vector<Real> radiationHydroInit(Real x,Real,Real,Real dx) {
    if(x!=opts().xscale||dx!=0) { throw std::runtime_error("Upstream reservoir must sample its exact boundary state"); }
    std::vector<Real> result(opts().n_fields+NRF);
    for(size_t i=0;i<result.size();++i) { result[i]=1e6+13*i; }
    return result;
}
struct grid {
    std::vector<std::vector<Real>> U{11,std::vector<Real>(14*14*14)};
    std::vector<std::vector<Real>> X{3,std::vector<Real>(14*14*14)};
    void set_physical_boundaries(const geo::face&,real);
};
struct rad_grid {
    std::vector<std::vector<Real>> U{NRF,std::vector<Real>(14*14*14)};
    void set_physical_boundaries(geo::face,Real);
};
void require(bool condition,const char* label) {
    if(!condition) { throw std::runtime_error(label); }
}
void near(Real actual,Real expected,const char* label) {
    if(!std::isfinite(actual)||std::abs(actual-expected)>1e-13*std::abs(expected)) {
        throw std::runtime_error(label);
    }
}
'''

CHECKS = r'''
void boundaries() {
    for(int face=0;face<6;++face) {
        grid hydro;
        rad_grid radiation;
        for(int i=0;i<14;++i) {
            for(int j=0;j<14;++j) {
                for(int k=0;k<14;++k) {
                    int const p=index(i,j,k);
                    hydro.X[0][p]=i-.5;
                    hydro.X[1][p]=j-2.5;
                    hydro.X[2][p]=k-3.5;
                    for(int f=0;f<opts().n_fields;++f) { hydro.U[f][p]=1+10000*f+p; }
                    for(int f=0;f<NRF;++f) { radiation.U[f][p]=(1+10000*f+p)*(f%2?-1:1); }
                }
            }
        }
        const auto oldHydro=hydro.U, oldRadiation=radiation.U;
        hydro.set_physical_boundaries({face},0);
        radiation.set_physical_boundaries({face},0);
        const auto reservoir=radiationHydroInit(opts().xscale,0,0,0);
        for(int i=0;i<14;++i) {
            for(int j=0;j<14;++j) {
                for(int k=0;k<14;++k) {
                    const int location[3]={i,j,k};
                    const int normal=face/2,coordinate=location[normal];
                    const bool ghost=face%2?coordinate>=11:coordinate<3;
                    const int target=index(i,j,k);
                    int sourceCoordinates[3]={i,j,k};
                    sourceCoordinates[normal]=face==0?5-coordinate:(face%2?10:3);
                    const int source=ghost?index(sourceCoordinates[0],sourceCoordinates[1],sourceCoordinates[2]):target;
                    std::vector<Real> expected(opts().n_fields);
                    for(int f=0;f<opts().n_fields;++f) {
                        expected[f]=ghost&&face==1?reservoir[f]:oldHydro[f][source];
                    }
                    if(ghost&&face==0) { expected[sx_i]=-expected[sx_i]; }
                    if(ghost) {
                        expected[lx_i]=hydro.X[1][target]*expected[sz_i]-hydro.X[2][target]*expected[sy_i];
                        expected[ly_i]=hydro.X[2][target]*expected[sx_i]-hydro.X[0][target]*expected[sz_i];
                        expected[lz_i]=hydro.X[0][target]*expected[sy_i]-hydro.X[1][target]*expected[sx_i];
                    }
                    for(int f=0;f<opts().n_fields;++f) {
                        require(hydro.U[f][target]==expected[f],"hydro ghost/interior values, mirror depth, reservoir and angular momentum");
                    }
                    for(int f=0;f<NRF;++f) {
                        Real wanted=ghost&&face==1?reservoir[opts().n_fields+f]:oldRadiation[f][source];
                        if(ghost&&face==0&&f==fx_i) { wanted=-wanted; }
                        require(radiation.U[f][target]==wanted,"radiation ghost/interior values, mirror depth, reservoir, unclipped transverse flux");
                    }
                }
            }
        }
    }
}
void opacities() {
    for(int problem:{RADIATION_SUBCRITICAL_SHOCK,RADIATION_SUPERCRITICAL_SHOCK,RADIATION_BLAST_WAVE}) {
        opts().problem=problem;
        for(Real mass:{1.,2.3e20}) {
            for(Real length:{1.,7e8}) {
                opts().code_to_g=mass;
                opts().code_to_cm=length;
                for(Real physicalDensity:{7.78e-10,1e-6,1e-15}) {
                    const Real codeDensity=physicalDensity*length*length*length/mass;
                    const Real expected=physicalDensity*(3.115e-10/7.78e-10)*length;
                    near(kappa_R(codeDensity,1.,.5,1.,0.),expected,"Rosseland rho*kappa and inverse-length units");
                    near(kappa_p(codeDensity,1.,.5,1.,0.),expected,"Planck pure absorption matches transport opacity");
                    near(kappa_R(2*codeDensity,5.,2.,.2,.4),2*expected,"constant mass opacity scales with compressed density");
                }
            }
        }
    }
}
int main() {
    try {
        boundaries(); opacities();
        std::cout<<"PASS: production gas/radiation shock boundary assignments on all six faces and opacity units\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
'''


def source():
    hydro_path = ROOT / "src/grid.cpp"
    hydro = block(hydro_path.read_text(), "void grid::set_physical_boundaries(")
    hydro = hydro[:hydro.index("\n\tif (opts().problem == AMR_TEST)")]
    hydro += '\nthrow std::runtime_error("Expected radiation shock branch");\n}'
    radiation_path = ROOT / "src/radiation/rad_grid.cpp"
    radiation = block(radiation_path.read_text(), "void rad_grid::set_physical_boundaries(")
    radiation = radiation[:radiation.index("                if (opts().problem == RADIATION_EQUILIBRIUM_SPHERE)")]
    radiation += '\nthrow std::runtime_error("Expected radiation shock branch");\n}}}}'
    opacity_path = ROOT / "octotiger/radiation/opacities.hpp"
    opacity = opacity_path.read_text()
    opacity_functions = []
    for signature in ("U temperature(", "U kappa_R(", "U kappa_p("):
        opacity_functions.append("template<class U>\n" + block(opacity, signature))
    return "\n".join((FIXTURE, hydro, radiation, *opacity_functions, CHECKS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="radiation-hydro-boundaries-") as temporary:
        folder = Path(temporary)
        path = folder / "boundaries.cpp"
        path.write_text(source())
        command = shlex.split(args.cxx) + ["-std=c++20", "-O2", "-Wall", "-Wextra", "-I" + str(ROOT)]
        if args.sanitize:
            command += ["-fsanitize=undefined", "-fno-sanitize-recover=all"]
        executable = folder / "boundaries"
        subprocess.run(command + [str(path), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
