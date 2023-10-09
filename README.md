# Octo-Tiger  

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ebc6d3e2e4f0407aa6a80dfc4fd03b97)](https://www.codacy.com/gh/STEllAR-GROUP/octotiger?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=STEllAR-GROUP/octotiger&amp;utm_campaign=Badge_Grade) [![CITATION-cff](https://github.com/STEllAR-GROUP/octotiger/actions/workflows/cff-validator.yml/badge.svg)](https://github.com/STEllAR-GROUP/octotiger/actions/workflows/cff-validator.yml)  [![DOI](https://zenodo.org/badge/73526736.svg)](https://zenodo.org/badge/latestdoi/73526736)

![Logo](https://stellar-group.org/wp-content/uploads/2020/11/octotigerlogoArtboard-github.png)

From <https://doi.org/10.1145/3204919.3204938>:
> Octo-Tiger is an astrophysics program simulating the evolution of star systems
> based on the fast multipole method on adaptive Octrees. It was implemented using
> high-level C++ libraries, specifically HPX and Vc, which allows its use on
> different hardware platforms.

## Build Status [master]


#### Jenkins - All CPU / GPU node-level tests for the 8 major build configurations:
|   	|   	|
|---	|---	|
| CPU/GPU Tests with Kokkos, CUDA, HIP, SYCL | [![Build Status](https://rostam.cct.lsu.edu/jenkins/buildStatus/icon?job=Octo-Tiger+Node-Level%2Fmaster&config=nodelevel)](https://rostam.cct.lsu.edu/jenkins/job/Octo-Tiger%20Node-Level/job/master/) |

#### Jenkins - Special machine tests:

|   	|   	|
|---	|---	|
|  KNL Kokkos HPX Backend / SIMD tests 	| [![Build Status](https://simsgs.informatik.uni-stuttgart.de/jenkins/buildStatus/icon?job=Octo-Tiger+KNL%2Fmaster&config=knlbuild)](https://simsgs.informatik.uni-stuttgart.de/jenkins/view/Octo-Tiger%20and%20Dependencies/job/Octo-Tiger%20KNL/job/master/)  	|
| Development environment tests 	|  [![Build Status](https://simsgs.informatik.uni-stuttgart.de/jenkins/buildStatus/icon?job=Octo-Tiger+DEV%2Fmaster&config=devbuild)](https://simsgs.informatik.uni-stuttgart.de/jenkins/view/Octo-Tiger%20and%20Dependencies/job/Octo-Tiger%20DEV/job/master/)

## Quick Reference
  * **Building Octo-Tiger**:
    - Option 1: Use the [Octo-Tiger Spack Package](https://github.com/G-071/octotiger-spack) (recommended)
    - Option 2: Use the [Octo-Tiger buildscripts](https://github.com/STEllAR-GROUP/OctoTigerBuildChain)

  * **Where to get help**:

  IRC Channel `#ste||ar` on [libera.chat](libera.chat)

  * **Where to file issues**:

  [Octo-Tiger Issue Tracker](https://github.com/STEllAR-GROUP/octotiger/issues)

  * **Documentation**:
  
  The [documentation](https://stellar-group.github.io/octotiger/doc/html/) of the master branch.

## Citing

In publications, please use the following publication to cite Octo-Tiger:

*  Dominic C. Marcello, Sagiv Shiber, Orsola De Marco, Juhan Frank, Geoffrey C. Clayton, Patrick M. Motl, Patrick Diehl, Hartmut Kaiser, "Octo-Tiger: A New, 3D Hydrodynamic Code for Stellar Mergers that uses HPX Parallelisation", accepted for publication in the Monthly Notices of the Royal Astronomical Society, 2021

For more publications, please review Octo-Tigers' [documentation](https://octotiger.stellar-group.org/papers/).


# Funding

## Allocations
 
* Porting Octo-Tiger, an astrophysics program simulating the evolution of star systems based on the fast multipole method on adaptive Octrees, Testbed, Ookami, Stony Brook University
* Porting Octo-Tiger, an astrophysics program simulating the evolution of star systems based on the fast multipole method on adaptive Octrees, Testbed, Fugaku, RIKEN Center for Computational Science 


## License
Distributed under the Boost Software License, Version 1.0. (See 
<http://www.boost.org/LICENSE_1_0.txt>)
