# Octo-Tiger [![link](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master.svg?style=shield)](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master)  [![Codacy Badge](https://app.codacy.com/project/badge/Grade/ebc6d3e2e4f0407aa6a80dfc4fd03b97)](https://www.codacy.com/gh/STEllAR-GROUP/octotiger?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=STEllAR-GROUP/octotiger&amp;utm_campaign=Badge_Grade) [![DOI](https://zenodo.org/badge/73526736.svg)](https://zenodo.org/badge/latestdoi/73526736)



![Logo](https://stellar-group.org/wp-content/uploads/2020/11/octotigerlogoArtboard-github.png)

From <https://doi.org/10.1145/3204919.3204938>:
> Octo-Tiger is an astrophysics program simulating the evolution of star systems
> based on the fast multipole method on adaptive Octrees. It was implemented using
> high-level C++ libraries, specifically HPX and Vc, which allows its use on
> different hardware platforms.

## Build Status

Current status of the [CircleCI](https://circleci.com/gh/STEllAR-GROUP/octotiger) continuous
integration service for the master branch:
[![link](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master.svg?style=shield)](https://circleci.com/gh/STEllAR-GROUP/octotiger/tree/master)

> **Note for maintainers**: The base Docker image used by CircleCI needs to be built
> and updated manually. Neither HPX nor any of the other dependencies update
> automatically. Relevant files are under
> [`tools/docker/base_image`](tools/docker/base_image).

## Quick Reference

  * **Where to get help**:

  IRC Channel `#ste||ar` on [freenode.net](https://freenode.net/)

  * **Where to file issues**:

  [Octo-Tiger Issue Tracker](https://github.com/STEllAR-GROUP/octotiger/issues)

  * **Wiki pages**:

  [Octo-Tiger Wiki](https://github.com/STEllAR-GROUP/octotiger/wiki)

  * **Documentation**:
  
  The [documentation](https://stellar-group.github.io/octotiger/doc/html/) of the master branch.

## Citing

In publications, please use the following publication to cite Octo-Tiger:

*  Dominic C. Marcello, Sagiv Shiber, Orsola De Marco, Juhan Frank, Geoffrey C. Clayton, Patrick M. Motl, Patrick Diehl, Hartmut Kaiser, "Octo-Tiger: A New, 3D Hydrodynamic Code for Stellar Mergers that uses HPX Parallelisation", accepted for publication in the Monthly Notices of the Royal Astronomical Society, 2021

For more publications, refer to Octo-Tigers' [documentation](https://stellar-group.github.io/octotiger/doc/html/md_content_publications.html).


# Funding

## Allocations
 
* Merger-Simulations using High-Level Abstractions, Production, Piz Daint, CSCS Swiss National Supercomputing Centre
* Porting Octo-Tiger, an astrophysics program simulating the evolution of star systems based on the fast multipole method on adaptive Octrees, Testbed, Ookami, Stony Brook University


## License
Distributed under the Boost Software License, Version 1.0. (See 
<http://www.boost.org/LICENSE_1_0.txt>)
