#!/usr/bin/env python3
"""Compare production subcycling against repeated CFL-sized driver calls.

Serial fixture delivery only; this does not validate HPX/MPI or Silo restart.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from verification_results.radiation.validate_so import productionSource

check = r'''
int main() {
 try {
  for(double ratio:{1.,.03})for(double chi:{0.,.7}) {
    reset();physcon().c=2.99792458e10;
    opts().radCRatio=ratio;opts().rad_implicit=chi>0;
    opts().radVelocityTerms=false;opts().radEnergyMode="equilibrium";
    opts().radiationOpacity.model="grey";opts().radiationOpacity.units="1/cm";
    opts().radiationOpacity.absorption=0;opts().radiationOpacity.scattering=chi;
    node_server subcycled,unsplit;
    for(auto node:{&subcycled,&unsplit}) {
      initialize(*node->rad_grid_ptr,*node->grid_ptr,1,.7,100,0);
      interiors([&](int h){
        double const E=1+.1*std::cos(2*std::numbers::pi*node->grid_ptr->X[0][h]);
        node->rad_grid_ptr->U[0][h]=E;node->rad_grid_ptr->U[1][h]=.7*physcon().c*E;
      });
      node->current_time=0;
      node->dt_.radiationDt=node->rad_grid_ptr->maxTimestep(0);
    }
    double const end=3.2*subcycled.dt_.radiationDt;
    radiation::SubcyclePlan const plan(end,subcycled.dt_.radiationDt,true,1024);
    require(plan.count==4,"four radiation substeps");
    subcycled.compute_radiation(end,0);
    opts().radSubcycling=false;
    for(std::size_t step=0;step<plan.count;++step) {
      unsplit.current_time=plan.offset(step);
      unsplit.compute_radiation(plan.dt(step),0);
    }
    interiors([&](int h){
      for(int f=0;f<4;++f)near(subcycled.rad_grid_ptr->U[f][h],unsplit.rad_grid_ptr->U[f][h],f==0?1:physcon().c,"subcycled vs CFL-step radiation");
      for(int f:{egas_i,sx_i,sy_i,sz_i})near(subcycled.grid_ptr->U[f][h],unsplit.grid_ptr->U[f][h],100,"subcycled vs CFL-step gas");
    });
    require(subcycled.hcycle==(chi>0?14:13),"subcycles do not advance hydro epochs");
    require(unsplit.hcycle==(chi>0?17:13),"separate gas steps have separate material refreshes");
    require(subcycled.boundaries.size()==5 && unsplit.boundaries.size()==8,"expected distinct boundary call counts");
    for(auto const& event:subcycled.audit)require(event.halo && event.gas,"valid halo and preserved gas ghosts");
    std::cout<<"PASS subcycle/reference cgs ratio="<<ratio<<" chi_scattering="<<chi<<" nsub=4 tolerance=3e-12\n";
  }
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--build-type', dest='buildType', choices=['Debug', 'Release', 'RelWithDebInfo'], default='Release')
    args = parser.parse_args()
    flags = {'Debug': ['-O0', '-g'], 'Release': ['-O2', '-DNDEBUG'], 'RelWithDebInfo': ['-O2', '-g', '-DNDEBUG']}[args.buildType]
    with tempfile.TemporaryDirectory(prefix='final-radiation-') as tmp:
        folder = Path(tmp)
        checks = (root/'verification_results/radiation/tests_so/checks.inc').read_text().split('int main() {')[0]
        src = folder/'check.cpp'; src.write_text(productionSource(root)+'\n'+checks+'\n'+check)
        exe = folder/'check'
        subprocess.run([args.cxx, '-std=c++23', *flags, '-I'+str(root), str(src), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__': main()
