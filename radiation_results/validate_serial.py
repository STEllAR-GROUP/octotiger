#!/usr/bin/env python3
"""Run the actual rad_grid transport/source methods on a serial uniform mesh.

This emits conservation histories, final slices and volume errors without HPX (validation only). Distributed
boundary actions are tested only by the full Octo-TIGER CTest cases.
"""
import argparse, pathlib, subprocess, tempfile
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--reference',type=pathlib.Path,required=True)
p.add_argument('--cells',type=int,default=32);p.add_argument('--cxx',default='g++')
p.add_argument('--sanitize',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--compile-only',action='store_true',help='Compile the production-method harness without opening the reference')
a=p.parse_args()
project=a.root.resolve()
nested=project/'src/octotiger'
root=nested if (nested/'CMakeLists.txt').is_file() else project
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
 'void rad_grid::set_X(', 'Real rad_grid::maxTimestep(', 'void rad_grid::compute_flux(',
 'void rad_grid::advance(',
 'void rad_grid::sanity_check()', 'void rad_grid::applyRegressionSource(',
 'radiationConservation::Totals rad_grid::takeConservation(', 'void rad_grid::accountBoundaryFlux(',
 'void rad_grid::set_physical_boundaries(', 'rad_grid::rad_grid(Real _dx)', 'rad_grid::rad_grid()'])
app_header='\n'.join(l for l in (root/'test_problems/radiation.hpp').read_text().splitlines() if not l.startswith(('#include','#pragma once')))
app_cpp='\n'.join(l for l in (root/'src/test_problems/radiation/radiation.cpp').read_text().splitlines() if not l.startswith('#include'))
support=pathlib.Path(__file__).resolve().parent/'support'
fixture='#include "'+str(support/'plot_output.hpp')+'"\n'+(support/'serial_fixture.inc').read_text()
checks=(support/'serial_checks.inc').read_text()
with tempfile.TemporaryDirectory(prefix='radiation-regression-') as tmp:
    folder=pathlib.Path(tmp);src=folder/'test.cpp';exe=folder/'test'
    src.write_text(fixture+app_header+'\n'+app_cpp+'\n'+header+'\n'+bodies+'\n'+checks)
    flags=['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if a.sanitize else ['-O3']
    subprocess.run([a.cxx,'-std=c++23',*flags,'-I'+str(root),'-DTEST_CELLS='+str(a.cells),str(src),'-o',str(exe)],check=True)
    if a.compile_only:
        print('Serial production-method harness compiled successfully')
    else:
        result=subprocess.run([str(exe),str(a.reference.resolve()),str(a.output.resolve())],check=True,text=True,stdout=subprocess.PIPE)
        print(result.stdout,end='')
        print('Serial harness output:',a.output)
