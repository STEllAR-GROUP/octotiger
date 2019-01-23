#!/bin/bash
set -x
if [ ! -f gravity_test.chk ]; then
 wget http://phys.lsu.edu/~dmarcel/gravity_test.chk
fi
if [ ! -f gravity_original.silo ]; then
wget http://phys.lsu.edu/~dmarcel/gravity_original.silo
fi
../octotiger -t8 --xscale=4.031496063 --problem=dwd --eos=ideal --maxlevel=8 --variable_omega=1 --core_fine --stopstep=0 --restart=./gravity_test.chk -Ihpx.stacks.use_guard_pages=0 --output=gravity_test
../silocmp gravity_test.silo gravity_original.silo 
