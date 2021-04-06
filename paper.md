---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Dominic
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Gregor
    affiliation: 2
  - name: Patrick
    affiliation: 2
  - name: Geoff C. Clayton
    affiliation: 2
  - name: Juhan Frank
    affiliation: 2
  - name: Kevin Huck
    affiliation: 2
  - name: Hartmut Kaiser
    affiliation: 2
  - name: Orsola De Marco
    affiliation: 2
  - name: Sagiv
    affiliation: 2
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution 2
   index: 2
date: 13 August 2017
bibliography: paper.bib
---

# Summary


## Implementation details

### HPX

[@Kaiser2020] [@daiss2019piz]

### Kokkos and CUDA integration

# Statement of need

Octotiger is designed to solve fluid dynamics and self-gravity for astrophysical applications. While there are many astrophysical fluid dynamics codes, Octotiger specializes in resolving the fluid dynamics of the early mass transfer and merging processes. Only so we will be able to accurately model the light properties of dynamical transients, such as mergers and other strong binary interactions, where the properties of the photospheric layers dominate the lightcurve properties. Achieving sufficient resolution invariably means large problems that cannot be calculated within reasonable wall-clock times. To expedite the calculation, Octotiger uses a fast, asynchronous many-task parallelization technique, HPX [@Kaiser2020], that allows efficient scaling to tens of thousands of cores, utilizing CPU and GPU architectures simultaneously [@daiss2019piz]. Additionally, Octotiger makes some choices at the hydrodynamics and gravity solver level to achieve an accurate solution, including  a fully three-dimensional reconstruction at cell faces and machine precision conservation of energy in both gravity and hydro solvers. Also, Octotiger’s gravity solver conserves angular momentum to machine precision. While the inclusion of hydrodynamics worsens the conservation, the use of a frame rotating at the orbital frequency allows for superior overall conservation properties with a low diffusion level.

# Acknowledgements

This work was supported by National Science Foundation Award 1814967. The numerical work was carried out using the computational resources (QueenBee2) of the Louisiana Optical Network
Initiative (LONI); Louisiana State University’s High Performance Computing (LSU HPC); resources of the National Energy Research Scientific Computing Center, a U.S. Departmentof Energy Office of Science User Facility operated underContract No. DE-AC02-05CH11231; and by Lilly Endowment,  Inc.,through its support for the Indiana University PervasiveTechnology Institute.

# References
