#!/bin/bash
set -x
if [ ! -f gravity_test.chk ]; then
 wget http://phys.lsu.edu/~dmarcel/gravity_test.chk
fi
if [ ! -f gravity_original.silo ]; then
wget http://phys.lsu.edu/~dmarcel/gravity_original.silo
fi
../octotiger -t8 -Xscale=10.0 -Xscale=4.031496063 -Problem=dwd -Eos=ideal -Max_level=8 -VariableOmega=1 -CoreRefine -Stopstep=0 -Restart=./gravity_test.chk -Ihpx.stacks.use_guard_pages=0 -Output=gravity_test
../silocmp gravity_test.silo gravity_original.silo 
