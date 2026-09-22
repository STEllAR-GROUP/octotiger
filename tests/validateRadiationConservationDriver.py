#!/usr/bin/env python3
"""Verify production radiation conservation collection and driver lifecycle.

Extracts the actual tree reduction, sampling callback and pre-regrid drain block.
A small fake-future tree checks leaf-only totals, exactly-once interval drains,
cumulative budgets across remapping, CSV schema and duplicate/final samples.
On supported platforms fake HPX waits also reject a live FpeGuard.
This serial test does not replace a distributed HPX application run.

Usage: python3 tests/validateRadiationConservationDriver.py
       python3 tests/validateRadiationConservationDriver.py --build-dir /tmp/rad-driver
CXX may select a different C++20 compiler.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]


def extractBlock(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


fixture = r'''
#include "octotiger/radiation/conservation.hpp"
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using Real=double;
constexpr int NCHILD=8;
using radiationConservation::Totals;
void require(bool value) { if(!value) throw std::runtime_error("Node conservation verification failed"); }
void requireNoGuard() {
#if HAS_FEENABLEEXCEPT
 require(fegetexcept()==0);
#endif
}
template<class T> struct future {
 T data;
 T get() { requireNoGuard(); return data; }
};
#define GET(x) ((x).get())
struct Grid {
 Totals totals;
 Totals takeConservation() {
  Totals result=totals;
  totals.boundary={}; totals.source={};
  return result;
 }
};
struct node_server;
struct node_client {
 node_server* node=nullptr;
 future<Totals> collectRadiationConservation() const;
};
struct node_server {
 bool is_refined=false;
 Grid* rad_grid_ptr=nullptr;
 std::array<node_client,NCHILD> children;
 Totals collectRadiationConservation();
};
future<Totals> node_client::collectRadiationConservation() const {
 requireNoGuard();
 return {node->collectRadiationConservation()};
}
'''

context = r'''

struct Options { bool radiation=true; std::string data_dir; };
Options& opts(){ static Options value; return value; }
void runSamples(node_server &root, Grid &leaf, const std::string& directory) {
 opts().data_dir=directory;
 double current_time=0;
 auto collectRadiationConservation=[&](){requireNoGuard();return root.collectRadiationConservation();};
'''

checks = r'''

 // Duplicate time must neither emit a row nor drain pending budgets.
 leaf.totals.boundary[0]=3;
 sampleRadiation();
 require(leaf.totals.boundary[0]==3);
 current_time=1;
 leaf.totals.source[0]=2;
 leaf.totals.value[0]-=1;
 // Exact pre-regrid lifecycle: drain old leaves, then change field totals.
@DRAIN@
 require(leaf.totals.boundary[0]==0 && leaf.totals.source[0]==0);
 leaf.totals.value[0]+=0.5; // Artificial remapping defect must remain visible.
 sampleRadiation();
 current_time=2;
 leaf.totals.source[0]=4;
 leaf.totals.value[0]+=4;
 sampleRadiation(); // Final numerical sample precedes destructive analytics.
 sampleRadiation();
 radiationOutput.close();
 std::ifstream in(directory+"/radiation-conservation.csv");
 std::string line;
 std::getline(in,line);
 require(line=="t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,er_source,fx_source,fy_source,fz_source");
 std::vector<std::vector<long double>> rows;
 while(std::getline(in,line)) {
  std::stringstream parts(line); std::string part; std::vector<long double> row;
  while(std::getline(parts,part,',')) row.push_back(std::stold(part));
  rows.push_back(row);
 }
 require(rows.size()==3);
 for(int i=0;i<3;++i) { require(rows[i].size()==14); require(rows[i][0]==i); require(rows[i][1]==15); }
 require(rows[0][6]==0 && rows[0][10]==0);
 require(rows[1][6]==3 && rows[1][10]==2);
 require(rows[2][6]==3 && rows[2][10]==6);
 require(rows[1][2]-rows[0][2]+rows[1][6]-rows[1][10]==.5L);
 require(rows[2][2]-rows[0][2]+rows[2][6]-rows[2][10]==.5L);
}
int main(int argc,char** argv) {
 require(argc==2);
#if HAS_FEENABLEEXCEPT
 fedisableexcept(FE_ALL_EXCEPT);
#endif
 std::array<Grid,17> grids;
 std::array<node_server,17> nodes;
 for(int i=0;i<17;++i) {
  nodes[i].rad_grid_ptr=&grids[i];
  grids[i].totals.volume=1;
  for(int f=0;f<4;++f) {
   grids[i].totals.value[f]=i*(f+1);
   grids[i].totals.boundary[f]=(i+1)*(f+1);
   grids[i].totals.source[f]=-2*(i+1)*(f+1);
  }
 }
 nodes[0].is_refined=nodes[1].is_refined=true;
 for(int i=0;i<8;++i) { nodes[0].children[i].node=&nodes[i+1]; nodes[1].children[i].node=&nodes[i+9]; }
 grids[0].totals.volume=grids[1].totals.volume=999;
 const auto first=nodes[0].collectRadiationConservation();
 require(first.volume==15);
 for(int f=0;f<4;++f) {
  require(first.value[f]==135*(f+1));
  require(first.boundary[f]==150*(f+1));
  require(first.source[f]==-300*(f+1));
 }
 const auto second=nodes[0].collectRadiationConservation();
 require(second.value==first.value);
 require(second.boundary==radiationConservation::Moments{});
 require(second.source==radiationConservation::Moments{});
 require(grids[0].totals.boundary[0]==1 && grids[1].totals.boundary[0]==2);
 std::filesystem::create_directories(argv[1]);
 runSamples(nodes[0],grids[2],argv[1]);
 requireNoGuard();
 std::cout<<"Actual collector and sampling code passed leaf-only, interval drain, regrid ledger, CSV, duplicate-time, final-sample and FPE scope checks\n";
}
'''

def productionParts():
    nodeSource = (root / "src/node_server_actions_2.cpp").read_text()
    collector = extractBlock(nodeSource, "radiationConservation::Totals node_server::collectRadiationConservation()")
    driverSource = (root / "src/node_server_actions_3.cpp").read_text()
    driver = extractBlock(driverSource, "void node_server::execute_solver(")
    sampleStart = driver.index("radiationConservation::Ledger radiationLedger;")
    loopStart = driver.index("while (current_time", sampleStart)
    sampling = driver[sampleStart:loopStart]
    if "sampleRadiation();" not in sampling:
        raise AssertionError("Missing initial radiation conservation sample")
    loop = extractBlock(driver, "while (current_time")
    if loop.index("sampleRadiation();") > loop.index("output_all("):
        raise AssertionError("Conservation intervals must drain before output/checkpointing")
    regrid = loop.index("ngrids = regrid(me.get_gid(), omega, new_floor, false);")
    drainStart = loop.rfind("if (opts().radiation)", 0, regrid)
    if drainStart < 0:
        raise AssertionError("Missing pre-regrid conservation drain")
    drain = extractBlock(loop[drainStart:regrid], "if (opts().radiation)")
    finalStart = driver.index(loop) + len(loop)
    finalTail = driver[finalStart:]
    if finalTail.index("sampleRadiation();") > finalTail.index("compare_analytic();"):
        raise AssertionError("Final conservation sample must precede analytic field replacement")
    return collector, sampling, drain


def run(buildDir):
    collector, sampling, drain = productionParts()
    buildDir.mkdir(parents=True, exist_ok=True)
    source = buildDir / "conservationDriver.cpp"
    executable = buildDir / "conservationDriver"
    source.write_text(fixture + collector + context + sampling + checks.replace("@DRAIN@", drain))
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(compiler + ["-std=c++20", "-O2", "-I", str(root), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(buildDir / "output")], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.build_dir:
        run(args.build_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="octotiger-conservation-driver-") as directory:
            run(Path(directory))


if __name__ == "__main__":
    main()
