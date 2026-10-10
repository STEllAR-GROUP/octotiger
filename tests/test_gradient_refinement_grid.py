#!/usr/bin/env python3
"""Compile the actual grid::refine_me with bounded synthetic grid storage.

The fixture uses different hydro/radiation halo widths to verify coordinate
conversion, and exercises all face-buffer widths without HPX or an application.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

FIXTURE = r'''
#include "octotiger/amr/gradient.hpp"
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#define ENABLE_THREAD_DEBUG()
using integer = int;
using real = double;
constexpr int NDIM=3, INX=8, H_BW=4, RAD_BW=3;
constexpr int H_NX=INX+2*H_BW, RAD_NX=INX+2*RAD_BW, REFINE_BW=2;
constexpr int H_DNX=H_NX*H_NX, H_DNY=H_NX, H_DNZ=1;
constexpr int XDIM=0, YDIM=1, ZDIM=2, rho_i=0;
int hindex(int i,int j,int k) { return (i*H_NX+j)*H_NX+k; }
struct Options {
    bool unigrid=false, radiation=true;
    int n_fields=1, gradient_refine_buffer=0;
    std::string gradient_refine_field="none", gradient_refine_mode="absolute";
    double gradient_refine_threshold=1, gradient_refine_floor=1e-8, gradient_refine_coarsen_factor=.8;
};
Options options;
Options& opts() { return options; }
int legacyCalls=0;
auto get_refine_test() {
    return [](int level,int maximum,double,double,double,const std::vector<double>&,
        const std::array<std::vector<double>,3>&) {
        ++legacyCalls;
        return options.unigrid && level<maximum;
    };
}
struct rad_grid {
    std::vector<double> values=std::vector<double>(RAD_NX*RAD_NX*RAD_NX,0);
    double get_field(int f,int i,int j,int k) const {
        if(f!=0 || i<0 || i>=RAD_NX || j<0 || j>=RAD_NX || k<0 || k>=RAD_NX) {
            throw std::out_of_range("Radiation refinement sample outside allocated halo");
        }
        return values.at((i*RAD_NX+j)*RAD_NX+k);
    }
    void set(int i,int j,int k,double value) { values.at((i*RAD_NX+j)*RAD_NX+k)=value; }
};
struct grid {
    static constexpr int min_level=1,max_level=2;
    std::vector<std::vector<double>> U{std::vector<double>(H_NX*H_NX*H_NX,1)};
    std::array<std::vector<double>,3> X;
    std::shared_ptr<rad_grid> rad_grid_ptr=std::make_shared<rad_grid>();
    double dx=.25;
    bool leaf=true;
    grid() { for(auto& coordinate:X) { coordinate.resize(H_NX*H_NX*H_NX); } }
    bool get_leaf() const { return leaf; }
    bool refine_me(integer,integer) const;
};
int checks=0;
void require(bool condition,const char* message) {
    ++checks;
    if(!condition) { throw std::runtime_error(message); }
}
template<class F> void rejects(F f,const char* message) {
    bool failed=false;
    try { f(); } catch(const std::exception&) { failed=true; }
    require(failed,message);
}
'''

TEST = r'''
int main() {
  try {
    grid g;
    require(!g.refine_me(1,0)&&legacyCalls>0,"Default none retains the legacy criterion");
    options.unigrid=true;
    options.gradient_refine_field="density";
    require(g.refine_me(1,0),"Unigrid overrides selected gradient");
    options.unigrid=false;
    legacyCalls=0;
    require(g.refine_me(0,0),"Minimum refinement level is mandatory");
    require(!g.refine_me(2,0),"Gradient cannot refine beyond maximum level");
    require(!g.refine_me(1,0)&&legacyCalls==0,"Selected gradient replaces legacy criterion");
    const int h=H_BW+3;
    g.U[0][hindex(h,h,h)]=1.225; // Edge gradient 0.9 per code length.
    require(!g.refine_me(1,0),"Leaf below threshold stays coarse");
    g.leaf=false;
    require(g.refine_me(1,0),"Refined parent retains the feature in the hysteresis band");
    g.leaf=true;
    g.U[0][hindex(h,h,h)]=1.5;
    require(g.refine_me(1,0),"Density peak refines");
    options.gradient_refine_field="radiation_energy";
    require(!g.refine_me(1,0),"Radiation criterion ignores hydrodynamic gradients");
    g.rad_grid_ptr->set(RAD_BW+3,RAD_BW+3,RAD_BW+3,.5);
    require(g.refine_me(1,0),"Radiation energy uses radiation coordinates with unequal halos");
    options.radiation=false;
    rejects([&]{g.refine_me(1,0);},"Reject radiation selection without radiation enabled");
    options.radiation=true;
    g.rad_grid_ptr.reset();
    rejects([&]{g.refine_me(1,0);},"Reject absent radiation storage");
    g.rad_grid_ptr=std::make_shared<rad_grid>();
    // Only buffer=2 reaches this third ghost-cell feature via a neighbor edge.
    g.rad_grid_ptr->set(0,RAD_BW+3,RAD_BW+3,.5);
    for(int buffer=0;buffer<=2;++buffer) {
        options.gradient_refine_buffer=buffer;
        require(g.refine_me(1,0)==(buffer==2),"Configured face buffer selects exact halo depth");
    }
    g.rad_grid_ptr=std::make_shared<rad_grid>();
    g.rad_grid_ptr->set(0,0,RAD_BW+3,.5);
    require(!g.refine_me(1,0),"Spatial buffer excludes edge and corner centers");
    // A nontriggering sweep must keep all index accesses in bounds for each width.
    for(int buffer=0;buffer<=2;++buffer) {
        options.gradient_refine_buffer=buffer;
        g.rad_grid_ptr=std::make_shared<rad_grid>();
        require(!g.refine_me(1,0),"Full radiation halo scan stays within allocation");
    }
    options.gradient_refine_field="density";
    options.gradient_refine_mode="relative";
    options.gradient_refine_buffer=0;
    std::fill(g.U[0].begin(),g.U[0].end(),2);
    g.U[0][hindex(h,h,h)]=2*std::exp(.5);
    require(g.refine_me(1,0),"Production relative logarithmic sensor is selected");
    options.gradient_refine_threshold=3;
    require(!g.refine_me(1,0),"Actual dx sets relative gradient dimensions");
    options.gradient_refine_field="bad";
    rejects([&]{g.refine_me(1,0);},"Unknown selected field is rejected");
    options.gradient_refine_field="density";
    options.gradient_refine_mode="bad";
    rejects([&]{g.refine_me(1,0);},"Unknown selected mode is rejected");
    std::cout<<"PASS: "<<checks<<" actual grid refinement wiring checks\n";
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n'; return 1;
  }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--fast-math', action='store_true')
    args = parser.parse_args()
    source = (ROOT / 'src/grid.cpp').read_text()
    begin = source.index('bool grid::refine_me(')
    end = source.index('\nvoid grid::rho_mult(', begin)
    production = source[begin:end]
    with tempfile.TemporaryDirectory(prefix='octotiger-gradient-wiring-') as tmp:
        cpp = Path(tmp) / 'check.cpp'
        binary = Path(tmp) / 'check'
        cpp.write_text(FIXTURE + '\n' + production + '\n' + TEST)
        command = [args.cxx, '-std=c++20', '-O1', '-g', '-I', str(ROOT), str(cpp), '-o', str(binary)]
        if args.sanitize:
            command += ['-fsanitize=undefined', '-fno-sanitize-recover=all']
        if args.fast_math:
            command += ['-ffast-math']
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
