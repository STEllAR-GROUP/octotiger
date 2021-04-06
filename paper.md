---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution 2
   index: 2
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary


## Implementation details

### HPX

### Kokkos and CUDA integration

# Statement of need

Octotiger is designed to solve fluid dynamics and self-gravity for astrophysical applications. While there are many astrophysical fluid dynamics codes, Octotiger specializes in resolving the fluid dynamics of the early mass transfer and merging processes. Only so we will be able to accurately model the light properties of dynamical transients, such as mergers and other strong binary interactions, where the properties of the photospheric layers dominate the lightcurve properties. Achieving sufficient resolution invariably means large problems that cannot be calculated within reasonable wall-clock times. To expedite the calculation, Octotiger uses a fast, asynchronous many-task parallelization technique, HPX, that allows efficient scaling to tens of thousands of cores, utilizing CPU and GPU architectures simultaneously. Additionally, Octotiger makes some choices at the hydrodynamics and gravity solver level to achieve an accurate solution, including  a fully three-dimensional reconstruction at cell faces and machine precision conservation of energy in both gravity and hydro solvers. Also, Octotiger’s gravity solver conserves angular momentum to machine precision. While the inclusion of hydrodynamics worsens the conservation, the use of a frame rotating at the orbital frequency allows for superior overall conservation properties with a low diffusion level.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
